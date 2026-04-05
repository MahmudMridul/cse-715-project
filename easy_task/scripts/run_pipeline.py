#!/usr/bin/env python3
"""Train VAE, cluster, evaluate, and save plots + metrics CSV."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clustering import pca_kmeans, run_kmeans
from src.dataset import make_train_loader, prepare_data
from src.evaluation import compute_metrics, metrics_to_dataframe
from src.vae import VAE, encode_mu, train_vae


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="VAE + K-Means pipeline")
    parser.add_argument(
        "--data",
        type=str,
        default="data/MillionSongSubset.csv",
        help="Path to CSV (relative to easy_task cwd or absolute)",
    )
    parser.add_argument("--out", type=str, default="results", help="Output directory")
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-clusters", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--silhouette-subsample",
        type=int,
        default=3000,
        help="Max samples for Silhouette (None = full)",
    )
    parser.add_argument(
        "--plot-pca",
        action="store_true",
        help="Also save t-SNE/UMAP for PCA+KMeans baseline",
    )
    parser.add_argument(
        "--viz-max-samples",
        type=int,
        default=4000,
        help="Max rows for t-SNE/UMAP (subsample for speed; None = all)",
    )
    args = parser.parse_args()

    cwd = Path.cwd()
    data_path = Path(args.data)
    if not data_path.is_file():
        alt = cwd / args.data
        if alt.is_file():
            data_path = alt
        else:
            alt2 = ROOT / args.data
            if alt2.is_file():
                data_path = alt2
            else:
                raise FileNotFoundError(f"Data file not found: {args.data}")

    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = cwd / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    viz_dir = out_root / "latent_visualization"
    viz_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prepared = prepare_data(data_path, test_size=0.1, random_state=args.seed)
    loader = make_train_loader(
        prepared.X_train, batch_size=args.batch_size, shuffle=True
    )

    model = VAE(
        input_dim=prepared.input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
    ).to(device)

    train_vae(
        model,
        loader,
        epochs=args.epochs,
        device=device,
        lr=args.lr,
        kl_weight=1.0,
    )

    mu = encode_mu(model, prepared.X_full, device=device)

    labels_vae, _ = run_kmeans(
        mu, args.n_clusters, random_state=args.seed
    )

    labels_pca, scores_pca, _, _ = pca_kmeans(
        prepared.X_full,
        n_components=args.latent_dim,
        n_clusters=args.n_clusters,
        random_state=args.seed,
    )

    sub = args.silhouette_subsample if args.silhouette_subsample > 0 else None

    m_vae = compute_metrics(mu, labels_vae, silhouette_sample_size=sub, random_state=args.seed)
    m_pca = compute_metrics(
        scores_pca, labels_pca, silhouette_sample_size=sub, random_state=args.seed
    )

    rows = [
        {
            "method": "vae_kmeans",
            "silhouette": m_vae["silhouette"],
            "calinski_harabasz": m_vae["calinski_harabasz"],
            "n_samples_silhouette": m_vae["n_samples_metric"],
            "latent_dim": args.latent_dim,
            "n_clusters": args.n_clusters,
        },
        {
            "method": "pca_kmeans",
            "silhouette": m_pca["silhouette"],
            "calinski_harabasz": m_pca["calinski_harabasz"],
            "n_samples_silhouette": m_pca["n_samples_metric"],
            "latent_dim": args.latent_dim,
            "n_clusters": args.n_clusters,
        },
    ]
    df = metrics_to_dataframe(rows)
    metrics_path = out_root / "clustering_metrics.csv"
    df.to_csv(metrics_path, index=False)
    print(f"Wrote {metrics_path}")

    viz_cap = args.viz_max_samples if args.viz_max_samples > 0 else None
    mu_v, lab_v = subsample_rows(mu, labels_vae, viz_cap, args.seed)
    tsne_vae = run_tsne(mu_v, args.seed)
    _plot_2d(
        tsne_vae,
        lab_v,
        "t-SNE of VAE latent means (K-Means colors)",
        viz_dir / "tsne_vae_kmeans.png",
    )
    print(f"Wrote {viz_dir / 'tsne_vae_kmeans.png'}")

    umap_emb = run_umap_2d(mu_v, args.seed)
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

    if args.plot_pca:
        pca_v, lab_p = subsample_rows(scores_pca, labels_pca, viz_cap, args.seed + 1)
        tsne_p = run_tsne(pca_v, args.seed + 1)
        _plot_2d(
            tsne_p,
            lab_p,
            "t-SNE of PCA scores (K-Means colors)",
            viz_dir / "tsne_pca_kmeans.png",
        )
        umap_p = run_umap_2d(pca_v, args.seed + 1)
        if umap_p is not None:
            _plot_2d(
                umap_p,
                lab_p,
                "UMAP of PCA scores (K-Means colors)",
                viz_dir / "umap_pca_kmeans.png",
            )


if __name__ == "__main__":
    main()
