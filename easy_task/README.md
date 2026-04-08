# Easy Task: VAE + Clustering on the Million Song Subset

Implements an unsupervised learning pipeline for music clustering:

1. **VAE** (Variational Autoencoder) for latent-feature extraction from 16 tabular Echo Nest features
2. **K-Means** on VAE latent means as the primary clustering method
3. **PCA + K-Means** as the linear baseline
4. **Full metric suite**: Silhouette, Calinski-Harabasz, Davies-Bouldin (unsupervised) + ARI, NMI, Cluster Purity against year-decade pseudo-labels

---

## Dataset

**`data/MillionSongSubset.csv`** — 10 000 songs from the [Million Song Dataset](http://millionsongdataset.com/) with Echo Nest audio features.

### Features used (16)

| Feature | Description |
|---|---|
| Duration | Song length (seconds) |
| KeySignature / KeySignatureConfidence | Estimated musical key and confidence |
| Tempo | BPM |
| TimeSignature / TimeSignatureConfidence | Estimated time signature |
| Year | Release year (0 treated as missing → median-imputed) |
| ArtistFamiliarity | Echo Nest familiarity score |
| Hotness | Echo Nest hotness score |
| end_of_fade_in / start_of_fade_out | Fade-in end and fade-out start (sec) |
| key / keyConfidence | Raw key index and confidence |
| Loudness | Overall loudness (dB) |
| mode / mode_confidence | Major/minor mode indicator |

**Excluded from original 20:**

| Column | Reason |
|---|---|
| Danceability | Identically 0 for all 10 001 songs — zero variance, adds no information |
| Energy | Identically 0 for all 10 001 songs — zero variance, adds no information |
| ArtistLatitude | 63 % missing; median imputation creates a fake US-centric geographic cluster |
| ArtistLongitude | 63 % missing; same issue as Latitude |

**Year=0 fix:** `Year == 0` (53 % of songs) is a sentinel meaning "unknown year" in the MSD — not year 0 CE. It is replaced with NaN before training and filled with the median of known release years by the imputer, preventing the VAE from learning a trivial "has year / no year" split.

---

## Pseudo Ground-Truth Labels (for ARI / NMI / Purity)

Since the dataset has no genre labels, we use **year-decade bins** as proxy labels:

| Label | Decade | Songs |
|---|---|---|
| 0 | Unknown era (Year = 0) | ~5 320 |
| 1 | Pre-1970s | ~100 |
| 2 | 1970s | ~200 |
| 3 | 1980s | ~400 |
| 4 | 1990s | ~500 |
| 5 | 2000s+ | ~2 800 |

Musical style varies substantially by decade; this provides a grounded proxy for cluster alignment evaluation.

---

## Setup

```bash
cd easy_task
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Run

From the `easy_task/` directory:

```bash
python3 scripts/run_pipeline.py --data data/MillionSongSubset.csv --out results
```

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--seeds` | `42,43,44,45,46` | Seeds for stability (5 runs per config) |
| `--latent-dim-list` | `8,16,32` | Latent dimensions to sweep |
| `--beta-list` | `0.1,0.5,1,2,4` | β (KL weight) values; > 1 → β-VAE |
| `--k-min / --k-max` | `2 / 30` | Cluster count sweep range |
| `--epochs` | `50` | Training epochs per VAE |
| `--hidden-dim` | `128` | Hidden layer width |
| `--num-hidden-layers` | `2` | Encoder/decoder depth |
| `--kl-warmup-epochs` | `10` | Linear KL annealing epochs (prevents posterior collapse) |
| `--clip-outliers` | off | Winsorize features to [1 %, 99 %] quantiles |
| `--power-transform` | off | Yeo-Johnson transform before scaling |
| `--also-gmm` | off | Also evaluate GMM on VAE and PCA representations |
| `--plot-pca` | off | Save t-SNE / UMAP for PCA baseline too |
| `--viz-max-samples` | `4000` | Subsample for t-SNE / UMAP (speed) |
| `--silhouette-subsample` | `3000` | Subsample for Silhouette (speed) |

---

## Outputs

| File | Description |
|---|---|
| `results/metrics_runs.csv` | One row per run (method / seed / latent_dim / beta / k + all metrics) |
| `results/metrics_summary.csv` | Mean ± std aggregated across seeds |
| `results/clustering_metrics.csv` | Best VAE config (highest mean silhouette) |
| `results/latent_visualization/tsne_vae_kmeans.png` | t-SNE of VAE latent space, coloured by K-Means cluster |
| `results/latent_visualization/umap_vae_kmeans.png` | UMAP of VAE latent space |
| `results/latent_visualization/reconstruction_samples.png` | Original vs reconstructed features for 6 sample songs |
| `results/latent_visualization/reconstruction_per_feature.png` | Mean absolute reconstruction error per feature |
| `results/latent_visualization/metrics_vs_k.png` | Silhouette / CH / DB of VAE vs PCA baseline across k values |
| `results/latent_visualization/tsne_pca_kmeans.png` | t-SNE of PCA scores (with `--plot-pca`) |
| `results/latent_visualization/umap_pca_kmeans.png` | UMAP of PCA scores (with `--plot-pca`) |

---

## Metrics

### Unsupervised (no labels required)

| Metric | Formula | Better |
|---|---|---|
| Silhouette Score | (b(i) − a(i)) / max(a(i), b(i)) | Higher (max 1) |
| Calinski-Harabasz | tr(B_k)/(k−1) / tr(W_k)/(n−k) | Higher |
| Davies-Bouldin | avg max_j≠i (σ_i + σ_j) / d_ij | Lower |

### Supervised (using year-decade pseudo-labels)

| Metric | Description | Better |
|---|---|---|
| ARI | Adjusted Rand Index — agreement with pseudo-labels, chance-corrected | Higher |
| NMI | Normalised Mutual Information between clusters and pseudo-labels | Higher |
| Purity | Fraction of dominant pseudo-class per cluster | Higher |

---

## Architecture

```
Input (16-dim)
    │
    ▼
Encoder  [Linear(16→128) → ReLU → Dropout(0.1)] × 2
    │
    ▼  (μ, log σ²)  ─── reparameterise ──►  z ~ N(μ, σ²)
    │
    ▼
Decoder  [Linear(latent→128) → ReLU → Dropout(0.1)] × 2 → Linear(128→16)
    │
    ▼
Reconstruction (MSE) + β·KL
```

Loss: `L = MSE(x̂, x) + β · KL(q(z|x) ‖ p(z))`

KL warmup: linear ramp over first 10 epochs prevents posterior collapse.

---

## Project layout

```
easy_task/
├── data/
│   └── MillionSongSubset.csv
├── scripts/
│   └── run_pipeline.py       end-to-end sweep + visualisation
├── src/
│   ├── dataset.py            CSV loading, preprocessing, pseudo-labels
│   ├── vae.py                VAE encoder/decoder, training loop
│   ├── clustering.py         K-Means, PCA, GMM (optional GPU via cuML)
│   └── evaluation.py         Silhouette, CH, DB, ARI, NMI, Purity
├── results/
│   ├── metrics_runs.csv
│   ├── metrics_summary.csv
│   ├── clustering_metrics.csv
│   └── latent_visualization/
├── requirements.txt
└── README.md
```
