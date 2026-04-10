# VAE + Clustering on the Million Song Subset

This project implements an unsupervised learning pipeline for music clustering:

1. **VAE** (Variational Autoencoder) for non-linear latent-feature extraction from 16 features
2. **K-Means** on VAE latent means as the primary clustering method
3. **PCA + K-Means** as the linear dimensionality-reduction baseline
4. **Full metric suite**: Silhouette, Calinski-Harabasz, Davies-Bouldin (unsupervised) + ARI, NMI, Cluster Purity against year-decade pseudo-labels

The pipeline runs a full hyperparameter sweep over latent dimensions, β (KL weight), and number of clusters k, repeated across multiple seeds for stability. All plots and CSVs needed for a report are saved automatically.

---

## Dataset

**`data/MillionSongSubset.csv`** — 10 000 songs from the [Million Song Dataset](http://millionsongdataset.com/).

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
| `--beta-list` | `0.01,0.05,0.1,0.5,1.0` | β (KL weight) values; > 1 → β-VAE |
| `--k-min / --k-max` | `2 / 15` | Cluster count sweep range |
| `--epochs` | `80` | Training epochs per VAE |
| `--hidden-dim` | `128` | Hidden layer width |
| `--num-hidden-layers` | `2` | Encoder/decoder depth |
| `--dropout` | `0.1` | Dropout probability in encoder/decoder layers |
| `--kl-warmup-epochs` | `10` | Linear KL annealing epochs (prevents posterior collapse) |
| `--early-stop-patience` | `0` (off) | Stop early if val loss does not improve for N epochs |
| `--clip-outliers` | off | Winsorize features to [1 %, 99 %] quantiles |
| `--power-transform` | off | Yeo-Johnson transform before scaling |
| `--also-gmm` | off | Also evaluate GMM on VAE and PCA representations |
| `--plot-pca` | on | Save t-SNE / UMAP for PCA baseline |
| `--viz-max-samples` | `4000` | Subsample for t-SNE / UMAP (speed) |
| `--silhouette-subsample` | `3000` | Subsample for Silhouette (speed) |

---

## Outputs

### CSV files

| File | Description |
|---|---|
| `results/metrics_runs.csv` | One row per run (method / seed / latent_dim / beta / k + all metrics) |
| `results/metrics_summary.csv` | Mean ± std aggregated across seeds |
| `results/clustering_metrics.csv` | Best VAE config (highest mean silhouette at k ≥ 3) |
| `results/comparison_table.csv` | Side-by-side VAE vs PCA metrics at the best config's k |
| `results/pipeline_run.log` | Full stdout/stderr log of the pipeline run |

### Plots (`results/latent_visualization/`)

| File | Description |
|---|---|
| `tsne_vae_kmeans.png` | t-SNE of VAE latent space, coloured by K-Means cluster |
| `tsne_vae_decades.png` | t-SNE of VAE latent space, coloured by year-decade pseudo-label |
| `umap_vae_kmeans.png` | UMAP of VAE latent space, coloured by K-Means cluster |
| `umap_vae_decades.png` | UMAP of VAE latent space, coloured by year-decade pseudo-label |
| `tsne_pca_kmeans.png` | t-SNE of PCA scores, coloured by K-Means cluster |
| `tsne_pca_decades.png` | t-SNE of PCA scores, coloured by year-decade pseudo-label |
| `umap_pca_kmeans.png` | UMAP of PCA scores, coloured by K-Means cluster |
| `umap_pca_decades.png` | UMAP of PCA scores, coloured by year-decade pseudo-label |
| `reconstruction_samples.png` | Original vs reconstructed features for 6 sample songs |
| `reconstruction_per_feature.png` | Mean absolute reconstruction error per feature |
| `metrics_vs_k.png` | Silhouette / CH / DB of VAE vs PCA across k values |
| `supervised_metrics_vs_k.png` | ARI / NMI / Purity of VAE vs PCA across k values |
| `elbow_vae.png` | Silhouette, CH, DB elbow plots for the best VAE config |
| `elbow_pca.png` | Silhouette, CH, DB elbow plots for the PCA baseline |
| `beta_comparison.png` | Effect of β on clustering metrics at the best latent_dim and k |
| `comparison_bar.png` | Side-by-side bar chart of all metrics: VAE vs PCA at best k |
| `latent_distributions.png` | Histograms of each latent dimension (VAE μ values) |
| `loss_curve_dim{d}_beta{b}.png` | Train/val loss curve for each (latent_dim, β) combination |

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

## Best Results

Best configuration: **latent_dim=8, β=1.0, k=8** (selected by highest mean Silhouette at k ≥ 3, averaged over 5 seeds).

| Metric | VAE + K-Means | PCA + K-Means |
|---|---|---|
| Silhouette Score | **0.2105 ± 0.0045** | 0.1134 ± 0.0018 |
| Calinski-Harabasz | **1916.35 ± 28.19** | 950.85 ± 5.08 |
| Davies-Bouldin | **1.2619 ± 0.0211** | 1.8323 ± 0.0077 |
| Adjusted Rand Index | 0.0520 ± 0.0010 | 0.0524 ± 0.0011 |
| Normalized Mutual Info | **0.1350 ± 0.0025** | 0.1177 ± 0.0055 |
| Cluster Purity | **0.5933 ± 0.0008** | 0.5901 ± 0.0017 |

VAE substantially outperforms PCA on all unsupervised metrics. Supervised metrics (ARI, NMI, Purity) are similar between methods, reflecting that year-decade is a weak proxy for the underlying structure captured by the latent space.

---

## Architecture

```
Input (16-dim)
    │
    ▼
Encoder  [Linear(16→128) → ReLU → Dropout(p)] × num_hidden_layers
    │
    ▼  (μ, log σ²)  ─── reparameterise ──►  z ~ N(μ, σ²)
    │
    ▼
Decoder  [Linear(latent→128) → ReLU → Dropout(p)] × num_hidden_layers → Linear(128→16)
    │
    ▼
Reconstruction (MSE) + β·KL
```

Loss: `L = MSE(x̂, x) + β · KL(q(z|x) ‖ p(z))`

- **KL warmup:** linear ramp over `--kl-warmup-epochs` (default 10) prevents posterior collapse.
- **β sweep:** values < 1 weight reconstruction more heavily; β = 1 is the standard VAE; β > 1 encourages a more disentangled latent space (β-VAE).
- **Dropout** (default p=0.1) is applied after each hidden ReLU in both encoder and decoder.
- **Validation split:** 10 % of data held out; train/val loss curves are saved for each (latent_dim, β) pair.
- **Early stopping:** optional via `--early-stop-patience` (off by default).
- **Clustering uses μ** (latent means), not sampled z, for deterministic and stable cluster assignments.

---

## Preprocessing

1. Load 16 numeric features from CSV.
2. Set `Year == 0` → NaN (sentinel for missing year).
3. Median-impute all missing values (fit on training split only to avoid leakage).
4. Optional winsorization to [1 %, 99 %] quantiles (`--clip-outliers`).
5. Optional Yeo-Johnson power transform to reduce skewness (`--power-transform`).
6. StandardScaler (fit on training split only).

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
│   ├── vae.py                VAE encoder/decoder, training loop, encode_mu
│   ├── clustering.py         K-Means, PCA+K-Means, GMM (optional GPU via cuML)
│   └── evaluation.py         Silhouette, CH, DB, ARI, NMI, Purity
├── results/
│   ├── metrics_runs.csv
│   ├── metrics_summary.csv
│   ├── clustering_metrics.csv
│   ├── comparison_table.csv
│   └── latent_visualization/
├── requirements.txt
└── README.md
```
