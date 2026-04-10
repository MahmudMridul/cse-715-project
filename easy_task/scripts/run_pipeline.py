#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clustering import pca_kmeans, run_gmm, run_kmeans
from src.dataset import (
    DECADE_LABEL_NAMES,
    get_year_labels,
    make_train_loader,
    prepare_data,
)
from src.evaluation import compute_metrics, metrics_to_dataframe
from src.vae import VAE, TrainHistory, encode_mu, train_vae


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_2d(
    emb: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: Path,
    label_names: list[str] | None = None,
    cmap: str = "tab10",
    legend: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    unique_labels = np.unique(labels)
    if label_names is not None and legend:
        for lbl in unique_labels:
            mask = labels == lbl
            name = label_names[lbl] if lbl < len(label_names) else f"Cluster {lbl}"
            ax.scatter(
                emb[mask, 0], emb[mask, 1],
                label=name, alpha=0.6, s=8, linewidths=0,
            )
        ax.legend(fontsize=7, markerscale=3, loc="best")
    else:
        scatter = ax.scatter(
            emb[:, 0], emb[:, 1],
            c=labels, cmap=cmap, alpha=0.6, s=8, linewidths=0,
        )
        plt.colorbar(scatter, ax=ax, label="cluster")
    ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_loss_curves(
    history: TrainHistory,
    out_path: Path,
    title: str = "VAE Training Loss",
) -> None:
    """Plot train and validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history.train_losses) + 1)
    ax.plot(epochs, history.train_losses, label="Train Loss", color="steelblue", linewidth=1.5)
    if history.val_losses:
        ax.plot(epochs, history.val_losses, label="Validation Loss", color="tomato", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_reconstructions(
    model: VAE,
    X_scaled: np.ndarray,
    feature_columns: list[str],
    device: torch.device,
    out_path: Path,
    n_samples: int = 6,
    seed: int = 42,
) -> None:
    """Side-by-side bar chart of original vs reconstructed features."""
    model.eval()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_scaled), size=n_samples, replace=False)
    X_sample = X_scaled[idx]

    with torch.no_grad():
        x_t = torch.from_numpy(X_sample).to(device)
        recon, _, _ = model(x_t)
        recon_np = recon.cpu().numpy()

    n_feats = len(feature_columns)
    x = np.arange(n_feats)
    width = 0.38

    fig, axes = plt.subplots(n_samples, 1, figsize=(14, 3 * n_samples))
    if n_samples == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.bar(x - width / 2, X_sample[i], width, label="Original", alpha=0.82, color="steelblue")
        ax.bar(x + width / 2, recon_np[i], width, label="Reconstructed", alpha=0.82, color="tomato")
        ax.set_xticks(x)
        ax.set_xticklabels(feature_columns, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Std. value", fontsize=7)
        ax.set_title(f"Sample {i + 1}", fontsize=9)
        if i == 0:
            ax.legend(fontsize=8)

    fig.suptitle("VAE Reconstruction: Original vs Reconstructed (standardised features)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_recon_per_feature(
    model: VAE,
    X_scaled: np.ndarray,
    feature_columns: list[str],
    device: torch.device,
    out_path: Path,
    n_samples: int = 1000,
    seed: int = 42,
) -> None:
    """Bar chart of mean absolute reconstruction error per feature."""
    model.eval()
    rng = np.random.default_rng(seed)
    n_use = min(n_samples, len(X_scaled))
    idx = rng.choice(len(X_scaled), size=n_use, replace=False)
    X_sample = X_scaled[idx]

    with torch.no_grad():
        x_t = torch.from_numpy(X_sample).to(device)
        recon, _, _ = model(x_t)
        recon_np = recon.cpu().numpy()

    mae = np.abs(X_sample - recon_np).mean(axis=0)

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(feature_columns)), mae, alpha=0.85, color="steelblue")
    ax.set_xticks(range(len(feature_columns)))
    ax.set_xticklabels(feature_columns, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean Absolute Error (standardised units)")
    ax.set_title(f"Per-Feature Reconstruction Error (VAE, n={n_use})")
    for bar, v in zip(bars, mae):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, f"{v:.3f}",
                ha="center", va="bottom", fontsize=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_vs_k(
    summary_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Line plot comparing VAE vs PCA silhouette / CH / DB across k values."""
    vae = summary_df[summary_df["method"] == "vae_kmeans"].copy()
    pca = summary_df[summary_df["method"] == "pca_kmeans"].copy()

    if vae.empty or pca.empty:
        return

    best_latent = int(vae.sort_values("silhouette_mean", ascending=False).iloc[0]["latent_dim"])
    best_beta = float(vae.sort_values("silhouette_mean", ascending=False).iloc[0]["beta"])
    vae_best = vae[(vae["latent_dim"] == best_latent) & (vae["beta"] == best_beta)].sort_values("n_clusters")
    pca_best = pca[pca["latent_dim"] == best_latent].sort_values("n_clusters")

    metrics = [
        ("silhouette_mean", "Silhouette Score (\u2191)"),
        ("calinski_harabasz_mean", "Calinski-Harabasz Index (\u2191)"),
        ("davies_bouldin_mean", "Davies-Bouldin Index (\u2193)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, (col, ylabel) in zip(axes, metrics):
        ax.plot(vae_best["n_clusters"], vae_best[col], marker="o", ms=4, label="VAE + K-Means", color="steelblue")
        ax.plot(pca_best["n_clusters"], pca_best[col], marker="s", ms=4, label="PCA + K-Means", color="tomato")
        std_col = col.replace("_mean", "_std")
        if std_col in vae_best.columns:
            ax.fill_between(
                vae_best["n_clusters"],
                vae_best[col] - vae_best[std_col],
                vae_best[col] + vae_best[std_col],
                alpha=0.15, color="steelblue",
            )
        if std_col in pca_best.columns:
            ax.fill_between(
                pca_best["n_clusters"],
                pca_best[col] - pca_best[std_col],
                pca_best[col] + pca_best[std_col],
                alpha=0.15, color="tomato",
            )
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Clustering Quality vs k  (latent_dim={best_latent}, \u03b2={best_beta})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_elbow(
    summary_df: pd.DataFrame,
    out_path: Path,
    method: str = "vae_kmeans",
    latent_dim: int | None = None,
    beta: float | None = None,
) -> None:
    """Elbow / silhouette plot for selecting the optimal k."""
    df = summary_df[summary_df["method"] == method].copy()
    if latent_dim is not None:
        df = df[df["latent_dim"] == latent_dim]
    if beta is not None:
        df = df[df["beta"] == beta]
    if df.empty:
        return
    df = df.sort_values("n_clusters")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Silhouette
    ax = axes[0]
    ax.plot(df["n_clusters"], df["silhouette_mean"], "o-", color="steelblue", ms=4)
    if "silhouette_std" in df.columns:
        ax.fill_between(df["n_clusters"],
                        df["silhouette_mean"] - df["silhouette_std"],
                        df["silhouette_mean"] + df["silhouette_std"],
                        alpha=0.15, color="steelblue")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score vs k")
    ax.grid(True, alpha=0.3)

    # CH Index
    ax = axes[1]
    ax.plot(df["n_clusters"], df["calinski_harabasz_mean"], "o-", color="forestgreen", ms=4)
    ax.set_xlabel("k")
    ax.set_ylabel("Calinski-Harabasz Index")
    ax.set_title("Calinski-Harabasz Index vs k")
    ax.grid(True, alpha=0.3)

    # DB Index
    ax = axes[2]
    ax.plot(df["n_clusters"], df["davies_bouldin_mean"], "o-", color="tomato", ms=4)
    ax.set_xlabel("k")
    ax.set_ylabel("Davies-Bouldin Index")
    ax.set_title("Davies-Bouldin Index vs k")
    ax.grid(True, alpha=0.3)

    label = f"{method}"
    if latent_dim is not None:
        label += f", dim={latent_dim}"
    if beta is not None:
        label += f", \u03b2={beta}"
    fig.suptitle(f"Cluster Selection Analysis ({label})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_supervised_metrics_vs_k(
    summary_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """ARI, NMI, Purity vs k for VAE and PCA."""
    vae = summary_df[summary_df["method"] == "vae_kmeans"].copy()
    pca = summary_df[summary_df["method"] == "pca_kmeans"].copy()
    if vae.empty or pca.empty:
        return

    best_latent = int(vae.sort_values("silhouette_mean", ascending=False).iloc[0]["latent_dim"])
    best_beta = float(vae.sort_values("silhouette_mean", ascending=False).iloc[0]["beta"])
    vae_best = vae[(vae["latent_dim"] == best_latent) & (vae["beta"] == best_beta)].sort_values("n_clusters")
    pca_best = pca[pca["latent_dim"] == best_latent].sort_values("n_clusters")

    sup_metrics = []
    for col in ("ari_mean", "nmi_mean", "purity_mean"):
        if col in vae_best.columns and vae_best[col].notna().any():
            sup_metrics.append(col)

    if not sup_metrics:
        return

    labels_map = {
        "ari_mean": "Adjusted Rand Index",
        "nmi_mean": "Normalized Mutual Information",
        "purity_mean": "Cluster Purity",
    }

    fig, axes = plt.subplots(1, len(sup_metrics), figsize=(5.5 * len(sup_metrics), 4))
    if len(sup_metrics) == 1:
        axes = [axes]

    for ax, col in zip(axes, sup_metrics):
        ax.plot(vae_best["n_clusters"], vae_best[col], "o-", ms=4, label="VAE + K-Means", color="steelblue")
        ax.plot(pca_best["n_clusters"], pca_best[col], "s-", ms=4, label="PCA + K-Means", color="tomato")
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel(labels_map.get(col, col))
        ax.set_title(labels_map.get(col, col))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Supervised Metrics vs k  (latent_dim={best_latent}, \u03b2={best_beta})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_bar(
    vae_metrics: dict,
    pca_metrics: dict,
    out_path: Path,
    k: int,
) -> None:
    """Side-by-side bar chart of all metrics for VAE vs PCA at the best k."""
    metric_names = ["Silhouette\n(\u2191)", "CH Index\n(\u2191, /1000)", "DB Index\n(\u2193)",
                    "ARI\n(\u2191)", "NMI\n(\u2191)", "Purity\n(\u2191)"]
    metric_keys = ["silhouette", "calinski_harabasz", "davies_bouldin", "ari", "nmi", "purity"]

    vae_vals = []
    pca_vals = []
    for key in metric_keys:
        v = vae_metrics.get(key, float("nan"))
        p = pca_metrics.get(key, float("nan"))
        # Scale CH down for display
        if key == "calinski_harabasz":
            v = v / 1000.0
            p = p / 1000.0
        vae_vals.append(v)
        pca_vals.append(p)

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - width / 2, vae_vals, width, label="VAE + K-Means", color="steelblue", alpha=0.85)
    bars2 = ax.bar(x + width / 2, pca_vals, width, label="PCA + K-Means", color="tomato", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=9)
    ax.set_ylabel("Metric Value")
    ax.set_title(f"VAE vs PCA + K-Means Comparison (k={k})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, v in zip(bars1, vae_vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    for bar, v in zip(bars2, pca_vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_latent_distributions(
    mu: np.ndarray,
    out_path: Path,
    latent_dim: int,
) -> None:
    """Histogram of latent dimension distributions."""
    n_dims = min(latent_dim, mu.shape[1])
    cols = min(4, n_dims)
    rows = (n_dims + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.atleast_2d(axes)
    for i in range(n_dims):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.hist(mu[:, i], bins=50, alpha=0.75, color="steelblue", edgecolor="white")
        ax.set_title(f"z_{i}", fontsize=9)
        ax.set_xlabel("Value", fontsize=7)
        ax.set_ylabel("Count", fontsize=7)
    # Hide unused axes
    for i in range(n_dims, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].set_visible(False)
    fig.suptitle("Latent Dimension Distributions (VAE \u03bc)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_beta_comparison(
    summary_df: pd.DataFrame,
    out_path: Path,
    latent_dim: int,
    k: int,
) -> None:
    """Show how beta affects metrics at a fixed latent_dim and k."""
    df = summary_df[
        (summary_df["method"] == "vae_kmeans")
        & (summary_df["latent_dim"] == latent_dim)
        & (summary_df["n_clusters"] == k)
    ].copy().sort_values("beta")
    if df.empty or len(df) < 2:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = [
        ("silhouette_mean", "Silhouette Score"),
        ("calinski_harabasz_mean", "Calinski-Harabasz"),
        ("davies_bouldin_mean", "Davies-Bouldin"),
    ]
    for ax, (col, name) in zip(axes, metrics):
        ax.plot(df["beta"], df[col], "o-", color="steelblue", ms=5)
        ax.set_xlabel("\u03b2 (KL weight)")
        ax.set_ylabel(name)
        ax.set_title(f"{name} vs \u03b2")
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
    fig.suptitle(f"Effect of \u03b2 on Clustering (latent_dim={latent_dim}, k={k})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def subsample_rows(
    X: np.ndarray,
    labels: np.ndarray,
    max_samples: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    if max_samples is None or n <= max_samples:
        return X, labels
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_samples, replace=False)
    return X[idx], labels[idx]


def run_tsne(X: np.ndarray, seed: int, perplexity: float = 30.0) -> np.ndarray:
    n = X.shape[0]
    perp = min(perplexity, max(5, (n - 1) // 3))
    kwargs = dict(
        n_components=2,
        random_state=seed,
        perplexity=perp,
        init="pca",
        learning_rate="auto",
    )
    try:
        tsne = TSNE(**kwargs, n_jobs=-1)
    except TypeError:
        tsne = TSNE(**kwargs)
    return tsne.fit_transform(X)


def run_umap_2d(X: np.ndarray, seed: int) -> np.ndarray | None:
    try:
        import umap
    except ImportError:
        return None
    n_neighbors = min(15, max(5, X.shape[0] - 1))
    reducer = umap.UMAP(
        n_components=2,
        random_state=seed,
        n_neighbors=n_neighbors,
        min_dist=0.1,
    )
    return reducer.fit_transform(X)


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _resolve_data_path(path_str: str) -> Path:
    cwd = Path.cwd()
    p = Path(path_str)
    if p.is_file():
        return p
    alt = cwd / path_str
    if alt.is_file():
        return alt
    alt2 = ROOT / path_str
    if alt2.is_file():
        return alt2
    raise FileNotFoundError(f"Data file not found: {path_str}")


def _select_best_config(summary_df: pd.DataFrame, min_k: int = 3) -> dict[str, object]:
    """Pick best VAE+KMeans config.

    Strategy: filter to k >= min_k (avoid trivial k=2), then pick by best
    mean silhouette, tie-break by lower mean Davies-Bouldin.
    """
    df = summary_df[
        (summary_df["method"] == "vae_kmeans")
        & (summary_df["n_clusters"] >= min_k)
    ].copy()
    df = df.sort_values(
        by=["silhouette_mean", "davies_bouldin_mean"],
        ascending=[False, True],
        kind="mergesort",
    )
    if df.empty:
        raise ValueError("No VAE results found in summary to select best config.")
    row = df.iloc[0].to_dict()
    return {
        "latent_dim": int(row["latent_dim"]),
        "beta": float(row["beta"]),
        "n_clusters": int(row["n_clusters"]),
    }


def _build_comparison_table(
    summary_df: pd.DataFrame,
    best: dict,
) -> pd.DataFrame:
    """Build a side-by-side comparison of VAE vs PCA at the best config's k and latent_dim."""
    k = best["n_clusters"]
    ld = best["latent_dim"]
    beta = best["beta"]

    vae_row = summary_df[
        (summary_df["method"] == "vae_kmeans")
        & (summary_df["latent_dim"] == ld)
        & (summary_df["beta"] == beta)
        & (summary_df["n_clusters"] == k)
    ]
    pca_row = summary_df[
        (summary_df["method"] == "pca_kmeans")
        & (summary_df["latent_dim"] == ld)
        & (summary_df["n_clusters"] == k)
    ]

    metric_cols = [
        ("silhouette_mean", "silhouette_std", "Silhouette Score"),
        ("calinski_harabasz_mean", "calinski_harabasz_std", "Calinski-Harabasz Index"),
        ("davies_bouldin_mean", "davies_bouldin_std", "Davies-Bouldin Index"),
        ("ari_mean", "ari_std", "Adjusted Rand Index"),
        ("nmi_mean", "nmi_std", "Normalized Mutual Information"),
        ("purity_mean", "purity_std", "Cluster Purity"),
    ]

    rows = []
    for mean_col, std_col, name in metric_cols:
        if mean_col not in summary_df.columns:
            continue
        vae_mean = float(vae_row[mean_col].iloc[0]) if not vae_row.empty else float("nan")
        vae_std = float(vae_row[std_col].iloc[0]) if not vae_row.empty and std_col in vae_row.columns else 0.0
        pca_mean = float(pca_row[mean_col].iloc[0]) if not pca_row.empty else float("nan")
        pca_std = float(pca_row[std_col].iloc[0]) if not pca_row.empty and std_col in pca_row.columns else 0.0
        rows.append({
            "Metric": name,
            "VAE + K-Means (mean)": round(vae_mean, 4),
            "VAE + K-Means (std)": round(vae_std, 4),
            "PCA + K-Means (mean)": round(pca_mean, 4),
            "PCA + K-Means (std)": round(pca_std, 4),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="VAE + K-Means pipeline")
    parser.add_argument(
        "--data", type=str, default="data/MillionSongSubset.csv",
        help="Path to CSV (relative to easy_task cwd or absolute)",
    )
    parser.add_argument("--out", type=str, default="results", help="Output directory")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--seeds", type=str, default="42,43,44,45,46",
        help="Comma-separated seeds for multi-run stability",
    )
    parser.add_argument(
        "--latent-dim-list", type=str, default="8,16,32",
        help="Comma-separated latent dimensions to sweep",
    )
    parser.add_argument(
        "--beta-list", type=str, default="0.01,0.05,0.1,0.5,1.0",
        help="Comma-separated beta (KL weight) values to sweep",
    )
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=15)
    parser.add_argument("--kmeans-n-init", type=int, default=30)
    parser.add_argument(
        "--cluster-backend", type=str, default="sklearn",
        choices=["sklearn", "cuml"],
    )
    parser.add_argument("--also-gmm", action="store_true")
    parser.add_argument("--num-hidden-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--kl-warmup-epochs", type=int, default=10)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--clip-outliers", action="store_true")
    parser.add_argument("--power-transform", action="store_true")
    parser.add_argument("--silhouette-subsample", type=int, default=3000)
    parser.add_argument("--plot-pca", action="store_true", default=True)
    parser.add_argument("--viz-max-samples", type=int, default=4000)
    args = parser.parse_args()

    data_path = _resolve_data_path(args.data)

    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = Path.cwd() / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    viz_dir = out_root / "latent_visualization"
    viz_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_int_list(args.seeds)
    latent_dims = parse_int_list(args.latent_dim_list)
    betas = parse_float_list(args.beta_list)
    k_values = list(range(int(args.k_min), int(args.k_max) + 1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sub = args.silhouette_subsample if args.silhouette_subsample > 0 else None
    patience = args.early_stop_patience if args.early_stop_patience > 0 else None

    print(f"Device: {device}")
    print(f"Seeds: {seeds}")
    print(f"Latent dims: {latent_dims}, Betas: {betas}, k: {k_values}")
    print(f"Architecture: hidden={args.hidden_dim}, layers={args.num_hidden_layers}, "
          f"dropout={args.dropout}, warmup={args.kl_warmup_epochs}")

    # Load decade-based pseudo-labels once.
    print("Loading year-based pseudo-labels ...")
    year_labels = get_year_labels(data_path)

    runs: list[dict[str, object]] = []

    for seed in seeds:
        set_seed(seed)
        prepared = prepare_data(
            data_path,
            test_size=0.1,
            random_state=seed,
            clip_outliers=args.clip_outliers,
            use_power_transform=args.power_transform,
        )
        train_loader = make_train_loader(
            prepared.X_train, batch_size=args.batch_size, shuffle=True
        )
        val_loader = make_train_loader(
            prepared.X_val, batch_size=args.batch_size, shuffle=False
        )

        for latent_dim in latent_dims:
            for beta in betas:
                print(f"\n--- seed={seed}, latent_dim={latent_dim}, beta={beta} ---")
                model = VAE(
                    input_dim=prepared.input_dim,
                    hidden_dim=args.hidden_dim,
                    latent_dim=latent_dim,
                    num_hidden_layers=args.num_hidden_layers,
                    dropout=args.dropout,
                ).to(device)
                history = train_vae(
                    model,
                    train_loader,
                    epochs=args.epochs,
                    device=device,
                    lr=args.lr,
                    kl_weight=float(beta),
                    kl_warmup_epochs=int(args.kl_warmup_epochs),
                    val_loader=val_loader,
                    early_stop_patience=patience,
                )

                # Save loss curves for the first seed
                if seed == seeds[0]:
                    plot_loss_curves(
                        history,
                        viz_dir / f"loss_curve_dim{latent_dim}_beta{beta}.png",
                        title=f"VAE Loss (dim={latent_dim}, \u03b2={beta})",
                    )

                mu = encode_mu(model, prepared.X_full, device=device)

                for k in k_values:
                    labels_vae, _ = run_kmeans(
                        mu, k, random_state=seed,
                        n_init=int(args.kmeans_n_init),
                        backend=args.cluster_backend,
                    )
                    m_vae = compute_metrics(
                        mu, labels_vae, labels_true=year_labels,
                        silhouette_sample_size=sub, random_state=seed,
                    )
                    runs.append({
                        "method": "vae_kmeans",
                        "seed": seed,
                        "latent_dim": latent_dim,
                        "beta": float(beta),
                        "n_clusters": k,
                        "silhouette": m_vae["silhouette"],
                        "calinski_harabasz": m_vae["calinski_harabasz"],
                        "davies_bouldin": m_vae["davies_bouldin"],
                        "ari": m_vae.get("ari", float("nan")),
                        "nmi": m_vae.get("nmi", float("nan")),
                        "purity": m_vae.get("purity", float("nan")),
                        "n_samples_silhouette": m_vae["n_samples_metric"],
                    })

                    if args.also_gmm:
                        labels_gmm, _ = run_gmm(mu, n_components=k, random_state=seed)
                        m_gmm = compute_metrics(
                            mu, labels_gmm, labels_true=year_labels,
                            silhouette_sample_size=sub, random_state=seed,
                        )
                        runs.append({
                            "method": "vae_gmm",
                            "seed": seed,
                            "latent_dim": latent_dim,
                            "beta": float(beta),
                            "n_clusters": k,
                            "silhouette": m_gmm["silhouette"],
                            "calinski_harabasz": m_gmm["calinski_harabasz"],
                            "davies_bouldin": m_gmm["davies_bouldin"],
                            "ari": m_gmm.get("ari", float("nan")),
                            "nmi": m_gmm.get("nmi", float("nan")),
                            "purity": m_gmm.get("purity", float("nan")),
                            "n_samples_silhouette": m_gmm["n_samples_metric"],
                        })

            # PCA baseline: once per (seed, latent_dim, k).
            for k in k_values:
                labels_pca, scores_pca_k, _, _ = pca_kmeans(
                    prepared.X_full,
                    n_components=latent_dim,
                    n_clusters=k,
                    random_state=seed,
                    n_init=int(args.kmeans_n_init),
                    backend=args.cluster_backend,
                )
                m_pca = compute_metrics(
                    scores_pca_k, labels_pca, labels_true=year_labels,
                    silhouette_sample_size=sub, random_state=seed,
                )
                runs.append({
                    "method": "pca_kmeans",
                    "seed": seed,
                    "latent_dim": latent_dim,
                    "beta": float("nan"),
                    "n_clusters": k,
                    "silhouette": m_pca["silhouette"],
                    "calinski_harabasz": m_pca["calinski_harabasz"],
                    "davies_bouldin": m_pca["davies_bouldin"],
                    "ari": m_pca.get("ari", float("nan")),
                    "nmi": m_pca.get("nmi", float("nan")),
                    "purity": m_pca.get("purity", float("nan")),
                    "n_samples_silhouette": m_pca["n_samples_metric"],
                })

                if args.also_gmm:
                    labels_gmm, _ = run_gmm(scores_pca_k, n_components=k, random_state=seed)
                    m_gmm = compute_metrics(
                        scores_pca_k, labels_gmm, labels_true=year_labels,
                        silhouette_sample_size=sub, random_state=seed,
                    )
                    runs.append({
                        "method": "pca_gmm",
                        "seed": seed,
                        "latent_dim": latent_dim,
                        "beta": float("nan"),
                        "n_clusters": k,
                        "silhouette": m_gmm["silhouette"],
                        "calinski_harabasz": m_gmm["calinski_harabasz"],
                        "davies_bouldin": m_gmm["davies_bouldin"],
                        "ari": m_gmm.get("ari", float("nan")),
                        "nmi": m_gmm.get("nmi", float("nan")),
                        "purity": m_gmm.get("purity", float("nan")),
                        "n_samples_silhouette": m_gmm["n_samples_metric"],
                    })

    # ------------------------------------------------------------------
    # Aggregate and save results
    # ------------------------------------------------------------------
    runs_df = metrics_to_dataframe(runs)
    runs_path = out_root / "metrics_runs.csv"
    runs_df.to_csv(runs_path, index=False)
    print(f"\nWrote {runs_path}")

    group_cols = ["method", "latent_dim", "beta", "n_clusters"]
    agg_cols = {
        "silhouette": ["mean", "std"],
        "calinski_harabasz": ["mean", "std"],
        "davies_bouldin": ["mean", "std"],
    }
    for col in ("ari", "nmi", "purity"):
        if col in runs_df.columns and runs_df[col].notna().any():
            agg_cols[col] = ["mean", "std"]

    summary = runs_df.groupby(group_cols, dropna=False).agg(agg_cols)
    summary.columns = ["_".join(c).strip() for c in summary.columns.to_flat_index()]
    summary = summary.reset_index()
    summary_path = out_root / "metrics_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    best = _select_best_config(summary, min_k=3)
    print(f"\nBest config: {best}")

    best_df = summary[
        (summary["method"] == "vae_kmeans")
        & (summary["latent_dim"] == best["latent_dim"])
        & (summary["beta"] == best["beta"])
        & (summary["n_clusters"] == best["n_clusters"])
    ].copy()
    legacy_path = out_root / "clustering_metrics.csv"
    best_df.to_csv(legacy_path, index=False)
    print(f"Wrote {legacy_path}")

    # Comparison table
    comparison = _build_comparison_table(summary, best)
    comp_path = out_root / "comparison_table.csv"
    comparison.to_csv(comp_path, index=False)
    print(f"Wrote {comp_path}")
    print("\n=== VAE vs PCA Comparison ===")
    print(comparison.to_string(index=False))

    # ------------------------------------------------------------------
    # Visualisations for the best config
    # ------------------------------------------------------------------
    best_seed = seeds[0]
    set_seed(best_seed)
    prepared = prepare_data(
        data_path, test_size=0.1, random_state=best_seed,
        clip_outliers=args.clip_outliers, use_power_transform=args.power_transform,
    )
    train_loader = make_train_loader(
        prepared.X_train, batch_size=args.batch_size, shuffle=True
    )
    val_loader = make_train_loader(prepared.X_val, batch_size=args.batch_size, shuffle=False)
    model = VAE(
        input_dim=prepared.input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=int(best["latent_dim"]),
        num_hidden_layers=args.num_hidden_layers,
        dropout=args.dropout,
    ).to(device)
    history = train_vae(
        model, train_loader, epochs=args.epochs, device=device,
        lr=args.lr, kl_weight=float(best["beta"]),
        kl_warmup_epochs=int(args.kl_warmup_epochs),
        val_loader=val_loader, early_stop_patience=patience,
    )
    mu = encode_mu(model, prepared.X_full, device=device)
    labels_vae, _ = run_kmeans(
        mu, int(best["n_clusters"]), random_state=best_seed,
        n_init=int(args.kmeans_n_init), backend=args.cluster_backend,
    )

    viz_cap = args.viz_max_samples if args.viz_max_samples > 0 else None

    # ---- Loss curve for best config ----
    plot_loss_curves(history, viz_dir / "loss_curve_best.png",
                     title=f"VAE Training Loss (best: dim={best['latent_dim']}, \u03b2={best['beta']})")
    print(f"Wrote {viz_dir / 'loss_curve_best.png'}")

    # ---- Latent dimension distributions ----
    plot_latent_distributions(mu, viz_dir / "latent_distributions.png",
                              latent_dim=int(best["latent_dim"]))
    print(f"Wrote {viz_dir / 'latent_distributions.png'}")

    # ---- t-SNE / UMAP with cluster colors ----
    mu_v, lab_v = subsample_rows(mu, labels_vae, viz_cap, best_seed)
    tsne_vae = run_tsne(mu_v, best_seed)
    _plot_2d(tsne_vae, lab_v,
             f"t-SNE of VAE Latent Space (K-Means, k={best['n_clusters']})",
             viz_dir / "tsne_vae_kmeans.png")
    print(f"Wrote {viz_dir / 'tsne_vae_kmeans.png'}")

    umap_emb = run_umap_2d(mu_v, best_seed)
    if umap_emb is not None:
        _plot_2d(umap_emb, lab_v,
                 f"UMAP of VAE Latent Space (K-Means, k={best['n_clusters']})",
                 viz_dir / "umap_vae_kmeans.png")
        print(f"Wrote {viz_dir / 'umap_vae_kmeans.png'}")
    else:
        print("UMAP skipped (umap-learn not available)")

    # ---- t-SNE / UMAP with decade colors (pseudo ground truth) ----
    _, year_v = subsample_rows(mu, year_labels, viz_cap, best_seed)
    _plot_2d(tsne_vae, year_v,
             "t-SNE of VAE Latent Space (Decade Labels)",
             viz_dir / "tsne_vae_decades.png",
             label_names=DECADE_LABEL_NAMES, legend=True)
    print(f"Wrote {viz_dir / 'tsne_vae_decades.png'}")

    if umap_emb is not None:
        _plot_2d(umap_emb, year_v,
                 "UMAP of VAE Latent Space (Decade Labels)",
                 viz_dir / "umap_vae_decades.png",
                 label_names=DECADE_LABEL_NAMES, legend=True)
        print(f"Wrote {viz_dir / 'umap_vae_decades.png'}")

    # ---- Reconstruction plots ----
    plot_reconstructions(
        model, prepared.X_full, prepared.feature_columns, device,
        viz_dir / "reconstruction_samples.png", n_samples=6, seed=best_seed,
    )
    print(f"Wrote {viz_dir / 'reconstruction_samples.png'}")

    plot_recon_per_feature(
        model, prepared.X_full, prepared.feature_columns, device,
        viz_dir / "reconstruction_per_feature.png", n_samples=1000, seed=best_seed,
    )
    print(f"Wrote {viz_dir / 'reconstruction_per_feature.png'}")

    # ---- Metrics vs k comparison ----
    plot_metrics_vs_k(summary, viz_dir / "metrics_vs_k.png")
    print(f"Wrote {viz_dir / 'metrics_vs_k.png'}")

    # ---- Supervised metrics vs k ----
    plot_supervised_metrics_vs_k(summary, viz_dir / "supervised_metrics_vs_k.png")
    print(f"Wrote {viz_dir / 'supervised_metrics_vs_k.png'}")

    # ---- Elbow / cluster selection plots ----
    plot_elbow(summary, viz_dir / "elbow_vae.png",
               method="vae_kmeans",
               latent_dim=int(best["latent_dim"]),
               beta=float(best["beta"]))
    print(f"Wrote {viz_dir / 'elbow_vae.png'}")

    plot_elbow(summary, viz_dir / "elbow_pca.png",
               method="pca_kmeans",
               latent_dim=int(best["latent_dim"]))
    print(f"Wrote {viz_dir / 'elbow_pca.png'}")

    # ---- Beta comparison plot ----
    plot_beta_comparison(summary, viz_dir / "beta_comparison.png",
                         latent_dim=int(best["latent_dim"]),
                         k=int(best["n_clusters"]))
    print(f"Wrote {viz_dir / 'beta_comparison.png'}")

    # ---- Comparison bar chart ----
    vae_row = summary[
        (summary["method"] == "vae_kmeans")
        & (summary["latent_dim"] == best["latent_dim"])
        & (summary["beta"] == best["beta"])
        & (summary["n_clusters"] == best["n_clusters"])
    ]
    pca_row = summary[
        (summary["method"] == "pca_kmeans")
        & (summary["latent_dim"] == best["latent_dim"])
        & (summary["n_clusters"] == best["n_clusters"])
    ]
    if not vae_row.empty and not pca_row.empty:
        vae_m = {
            "silhouette": float(vae_row["silhouette_mean"].iloc[0]),
            "calinski_harabasz": float(vae_row["calinski_harabasz_mean"].iloc[0]),
            "davies_bouldin": float(vae_row["davies_bouldin_mean"].iloc[0]),
            "ari": float(vae_row["ari_mean"].iloc[0]) if "ari_mean" in vae_row.columns else float("nan"),
            "nmi": float(vae_row["nmi_mean"].iloc[0]) if "nmi_mean" in vae_row.columns else float("nan"),
            "purity": float(vae_row["purity_mean"].iloc[0]) if "purity_mean" in vae_row.columns else float("nan"),
        }
        pca_m = {
            "silhouette": float(pca_row["silhouette_mean"].iloc[0]),
            "calinski_harabasz": float(pca_row["calinski_harabasz_mean"].iloc[0]),
            "davies_bouldin": float(pca_row["davies_bouldin_mean"].iloc[0]),
            "ari": float(pca_row["ari_mean"].iloc[0]) if "ari_mean" in pca_row.columns else float("nan"),
            "nmi": float(pca_row["nmi_mean"].iloc[0]) if "nmi_mean" in pca_row.columns else float("nan"),
            "purity": float(pca_row["purity_mean"].iloc[0]) if "purity_mean" in pca_row.columns else float("nan"),
        }
        plot_comparison_bar(vae_m, pca_m, viz_dir / "comparison_bar.png", k=int(best["n_clusters"]))
        print(f"Wrote {viz_dir / 'comparison_bar.png'}")

    # ---- PCA baseline visualizations ----
    if args.plot_pca:
        labels_pca, scores_pca, _, _ = pca_kmeans(
            prepared.X_full,
            n_components=int(best["latent_dim"]),
            n_clusters=int(best["n_clusters"]),
            random_state=best_seed,
            n_init=int(args.kmeans_n_init),
            backend=args.cluster_backend,
        )
        pca_v, lab_p = subsample_rows(scores_pca, labels_pca, viz_cap, best_seed + 1)
        tsne_p = run_tsne(pca_v, best_seed + 1)
        _plot_2d(tsne_p, lab_p,
                 f"t-SNE of PCA Scores (K-Means, k={best['n_clusters']})",
                 viz_dir / "tsne_pca_kmeans.png")
        print(f"Wrote {viz_dir / 'tsne_pca_kmeans.png'}")

        umap_p = run_umap_2d(pca_v, best_seed + 1)
        if umap_p is not None:
            _plot_2d(umap_p, lab_p,
                     f"UMAP of PCA Scores (K-Means, k={best['n_clusters']})",
                     viz_dir / "umap_pca_kmeans.png")
            print(f"Wrote {viz_dir / 'umap_pca_kmeans.png'}")

        # PCA with decade colors
        _, year_p = subsample_rows(scores_pca, year_labels, viz_cap, best_seed + 1)
        _plot_2d(tsne_p, year_p,
                 "t-SNE of PCA Scores (Decade Labels)",
                 viz_dir / "tsne_pca_decades.png",
                 label_names=DECADE_LABEL_NAMES, legend=True)
        print(f"Wrote {viz_dir / 'tsne_pca_decades.png'}")

        if umap_p is not None:
            _plot_2d(umap_p, year_p,
                     "UMAP of PCA Scores (Decade Labels)",
                     viz_dir / "umap_pca_decades.png",
                     label_names=DECADE_LABEL_NAMES, legend=True)
            print(f"Wrote {viz_dir / 'umap_pca_decades.png'}")

    print(f"\nDone.  Best config: {best}")
    print(f"All results saved to: {out_root}")


if __name__ == "__main__":
    main()
