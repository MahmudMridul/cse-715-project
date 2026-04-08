#!/usr/bin/env python3
"""Train VAE, run sweeps, cluster, evaluate, and save plots + metrics."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clustering import pca_kmeans, run_gmm, run_kmeans
from src.dataset import get_year_labels, make_train_loader, prepare_data
from src.evaluation import compute_metrics, metrics_to_dataframe
from src.vae import VAE, encode_mu, train_vae


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
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        emb[:, 0],
        emb[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.65,
        s=8,
        linewidths=0,
    )
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    plt.colorbar(scatter, ax=ax, label="cluster")
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
    """Side-by-side bar chart of original vs reconstructed features for n_samples songs."""
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
    # Annotate bars with values
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

    # Use best VAE latent_dim / beta to get a clean per-k curve
    if vae.empty or pca.empty:
        return

    best_latent = int(vae.sort_values("silhouette_mean", ascending=False).iloc[0]["latent_dim"])
    best_beta = float(vae.sort_values("silhouette_mean", ascending=False).iloc[0]["beta"])
    vae_best = vae[(vae["latent_dim"] == best_latent) & (vae["beta"] == best_beta)].sort_values("n_clusters")
    pca_best = pca[pca["latent_dim"] == best_latent].sort_values("n_clusters")

    metrics = [
        ("silhouette_mean", "Silhouette Score (↑)"),
        ("calinski_harabasz_mean", "Calinski-Harabasz Index (↑)"),
        ("davies_bouldin_mean", "Davies-Bouldin Index (↓)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, (col, ylabel) in zip(axes, metrics):
        ax.plot(vae_best["n_clusters"], vae_best[col], marker="o", ms=4, label="VAE + K-Means", color="steelblue")
        ax.plot(pca_best["n_clusters"], pca_best[col], marker="s", ms=4, label="PCA + K-Means", color="tomato")
        if "silhouette_std" in vae_best.columns and "silhouette" in col:
            std_col = col.replace("_mean", "_std")
            ax.fill_between(
                vae_best["n_clusters"],
                vae_best[col] - vae_best[std_col],
                vae_best[col] + vae_best[std_col],
                alpha=0.15,
                color="steelblue",
            )
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Clustering Quality vs k  (latent_dim={best_latent}, β={best_beta})", fontsize=12)
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


def _select_best_config(summary_df: pd.DataFrame) -> dict[str, object]:
    """Pick best VAE+KMeans config by mean silhouette, tie-break by lower mean DB."""
    df = summary_df[summary_df["method"] == "vae_kmeans"].copy()
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="VAE + K-Means pipeline")
    parser.add_argument(
        "--data",
        type=str,
        default="data/MillionSongSubset.csv",
        help="Path to CSV (relative to easy_task cwd or absolute)",
    )
    parser.add_argument("--out", type=str, default="results", help="Output directory")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44,45,46",
        help="Comma-separated seeds for multi-run stability",
    )
    parser.add_argument(
        "--latent-dim-list",
        type=str,
        default="8,16,32",
        help="Comma-separated latent dimensions to sweep",
    )
    parser.add_argument(
        "--beta-list",
        type=str,
        default="0.1,0.5,1,2,4",
        help="Comma-separated beta (KL weight) values to sweep",
    )
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=30)
    parser.add_argument("--kmeans-n-init", type=int, default=30)
    parser.add_argument(
        "--cluster-backend",
        type=str,
        default="sklearn",
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
    parser.add_argument("--plot-pca", action="store_true")
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

    # Load decade-based pseudo-labels once (same for every seed / config).
    print("Loading year-based pseudo-labels …")
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
                model = VAE(
                    input_dim=prepared.input_dim,
                    hidden_dim=args.hidden_dim,
                    latent_dim=latent_dim,
                    num_hidden_layers=args.num_hidden_layers,
                    dropout=args.dropout,
                ).to(device)
                train_vae(
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

                mu = encode_mu(model, prepared.X_full, device=device)

                for k in k_values:
                    labels_vae, _ = run_kmeans(
                        mu,
                        k,
                        random_state=seed,
                        n_init=int(args.kmeans_n_init),
                        backend=args.cluster_backend,
                    )
                    m_vae = compute_metrics(
                        mu,
                        labels_vae,
                        labels_true=year_labels,
                        silhouette_sample_size=sub,
                        random_state=seed,
                    )
                    runs.append(
                        {
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
                        }
                    )

                    if args.also_gmm:
                        labels_gmm, _ = run_gmm(mu, n_components=k, random_state=seed)
                        m_gmm = compute_metrics(
                            mu,
                            labels_gmm,
                            labels_true=year_labels,
                            silhouette_sample_size=sub,
                            random_state=seed,
                        )
                        runs.append(
                            {
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
                            }
                        )

            # PCA baseline: beta is not applicable; evaluated once per (seed, latent_dim, k).
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
                    scores_pca_k,
                    labels_pca,
                    labels_true=year_labels,
                    silhouette_sample_size=sub,
                    random_state=seed,
                )
                runs.append(
                    {
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
                    }
                )

                if args.also_gmm:
                    labels_gmm, _ = run_gmm(scores_pca_k, n_components=k, random_state=seed)
                    m_gmm = compute_metrics(
                        scores_pca_k,
                        labels_gmm,
                        labels_true=year_labels,
                        silhouette_sample_size=sub,
                        random_state=seed,
                    )
                    runs.append(
                        {
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
                        }
                    )

    # ------------------------------------------------------------------
    # Aggregate and save results
    # ------------------------------------------------------------------
    runs_df = metrics_to_dataframe(runs)
    runs_path = out_root / "metrics_runs.csv"
    runs_df.to_csv(runs_path, index=False)
    print(f"Wrote {runs_path}")

    group_cols = ["method", "latent_dim", "beta", "n_clusters"]
    agg_cols = {
        "silhouette": ["mean", "std"],
        "calinski_harabasz": ["mean", "std"],
        "davies_bouldin": ["mean", "std"],
    }
    # Include supervised metrics in aggregation if present
    for col in ("ari", "nmi", "purity"):
        if col in runs_df.columns and runs_df[col].notna().any():
            agg_cols[col] = ["mean", "std"]

    summary = runs_df.groupby(group_cols, dropna=False).agg(agg_cols)
    summary.columns = ["_".join(c).strip() for c in summary.columns.to_flat_index()]
    summary = summary.reset_index()
    summary_path = out_root / "metrics_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    best = _select_best_config(summary)
    best_df = summary[
        (summary["method"] == "vae_kmeans")
        & (summary["latent_dim"] == best["latent_dim"])
        & (summary["beta"] == best["beta"])
        & (summary["n_clusters"] == best["n_clusters"])
    ].copy()
    legacy_path = out_root / "clustering_metrics.csv"
    best_df.to_csv(legacy_path, index=False)
    print(f"Wrote {legacy_path}")

    # ------------------------------------------------------------------
    # Visualisations for the best config
    # ------------------------------------------------------------------
    best_seed = seeds[0]
    set_seed(best_seed)
    prepared = prepare_data(
        data_path,
        test_size=0.1,
        random_state=best_seed,
        clip_outliers=args.clip_outliers,
        use_power_transform=args.power_transform,
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
    train_vae(
        model,
        train_loader,
        epochs=args.epochs,
        device=device,
        lr=args.lr,
        kl_weight=float(best["beta"]),
        kl_warmup_epochs=int(args.kl_warmup_epochs),
        val_loader=val_loader,
        early_stop_patience=patience,
    )
    mu = encode_mu(model, prepared.X_full, device=device)
    labels_vae, _ = run_kmeans(
        mu,
        int(best["n_clusters"]),
        random_state=best_seed,
        n_init=int(args.kmeans_n_init),
        backend=args.cluster_backend,
    )

    viz_cap = args.viz_max_samples if args.viz_max_samples > 0 else None
    mu_v, lab_v = subsample_rows(mu, labels_vae, viz_cap, best_seed)

    # t-SNE latent space
    tsne_vae = run_tsne(mu_v, best_seed)
    _plot_2d(
        tsne_vae,
        lab_v,
        "t-SNE of VAE latent means (K-Means colors)",
        viz_dir / "tsne_vae_kmeans.png",
    )
    print(f"Wrote {viz_dir / 'tsne_vae_kmeans.png'}")

    # UMAP latent space
    umap_emb = run_umap_2d(mu_v, best_seed)
    if umap_emb is not None:
        _plot_2d(
            umap_emb,
            lab_v,
            "UMAP of VAE latent means (K-Means colors)",
            viz_dir / "umap_vae_kmeans.png",
        )
        print(f"Wrote {viz_dir / 'umap_vae_kmeans.png'}")
    else:
        print("UMAP skipped (umap-learn not available)")

    # Reconstruction visualisations
    plot_reconstructions(
        model,
        prepared.X_full,
        prepared.feature_columns,
        device,
        viz_dir / "reconstruction_samples.png",
        n_samples=6,
        seed=best_seed,
    )
    print(f"Wrote {viz_dir / 'reconstruction_samples.png'}")

    plot_recon_per_feature(
        model,
        prepared.X_full,
        prepared.feature_columns,
        device,
        viz_dir / "reconstruction_per_feature.png",
        n_samples=1000,
        seed=best_seed,
    )
    print(f"Wrote {viz_dir / 'reconstruction_per_feature.png'}")

    # Metrics vs k comparison plot
    plot_metrics_vs_k(summary, viz_dir / "metrics_vs_k.png")
    print(f"Wrote {viz_dir / 'metrics_vs_k.png'}")

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
        _plot_2d(
            tsne_p,
            lab_p,
            "t-SNE of PCA scores (K-Means colors)",
            viz_dir / "tsne_pca_kmeans.png",
        )
        umap_p = run_umap_2d(pca_v, best_seed + 1)
        if umap_p is not None:
            _plot_2d(
                umap_p,
                lab_p,
                "UMAP of PCA scores (K-Means colors)",
                viz_dir / "umap_pca_kmeans.png",
            )

    print("\nDone.  Best config:", best)


if __name__ == "__main__":
    main()
