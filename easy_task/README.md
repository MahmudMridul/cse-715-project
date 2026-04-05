# Easy task: VAE + clustering (Million Song subset)

Trains a basic variational autoencoder (VAE) on numeric tabular features from [`data/MillionSongSubset.csv`](data/MillionSongSubset.csv), clusters songs with K-Means in the latent mean space, and compares against **PCA + K-Means** using **Silhouette** and **Calinski-Harabasz** scores. Saves 2D **t-SNE** and **UMAP** plots colored by cluster.

## Setup

```bash
cd easy_task
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

From the `easy_task` directory:

```bash
python scripts/run_pipeline.py --data data/MillionSongSubset.csv --out results --latent-dim 16 --hidden-dim 128 --n-clusters 10 --epochs 50 --seed 42
```

Optional flags:

- `--viz-max-samples 4000` — subsample rows (default) for t-SNE/UMAP only (faster); use `0` for all rows.
- `--plot-pca` — also save t-SNE/UMAP for the PCA + K-Means baseline.
- `--silhouette-subsample 3000` — cap points for Silhouette (default); use `0` for the full dataset.

Outputs:

- `results/clustering_metrics.csv` — metrics for `vae_kmeans` and `pca_kmeans`
- `results/latent_visualization/tsne_vae_kmeans.png`
- `results/latent_visualization/umap_vae_kmeans.png`
- With `--plot-pca`: `tsne_pca_kmeans.png`, `umap_pca_kmeans.png`

**Defaults:** `n_clusters=10` is a reasonable exploratory choice for ~10k tracks; tune `--n-clusters` and `--latent-dim` for your experiments.

## Project layout

- `src/dataset.py` — load CSV, impute missing values, standardize (fit on train split)
- `src/vae.py` — PyTorch VAE and training
- `src/clustering.py` — K-Means and PCA baseline
- `src/evaluation.py` — Silhouette and Calinski-Harabasz
- `scripts/run_pipeline.py` — end-to-end pipeline

The CSV provides hand-crafted audio/metadata features (Echo Nest style), not raw waveforms or lyrics.
