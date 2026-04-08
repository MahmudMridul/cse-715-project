"""Clustering metrics: unsupervised (Silhouette, CH, DB) + supervised (ARI, NMI, Purity)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)


# ---------------------------------------------------------------------------
# Supervised metrics (require ground-truth labels)
# ---------------------------------------------------------------------------

def _cluster_purity(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Cluster purity: fraction of dominant true class across all clusters.

    Purity = (1/n) * sum_k max_j |c_k ∩ t_j|
    """
    n = len(labels_true)
    total_correct = 0
    for cluster_id in np.unique(labels_pred):
        mask = labels_pred == cluster_id
        if not np.any(mask):
            continue
        # Count occurrences of each true label in this cluster
        true_in_cluster = labels_true[mask]
        counts = np.bincount(true_in_cluster.astype(int) + (-true_in_cluster.min() if true_in_cluster.min() < 0 else 0))
        total_correct += int(counts.max())
    return total_correct / n


def compute_supervised_metrics(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
) -> dict[str, float]:
    """Compute ARI, NMI, and Purity when ground-truth (or pseudo) labels are available.

    Parameters
    ----------
    labels_true : integer array of shape (n,)
        Ground-truth or pseudo-ground-truth class labels.
    labels_pred : integer array of shape (n,)
        Predicted cluster assignments.

    Returns
    -------
    dict with keys: 'ari', 'nmi', 'purity'
    """
    ari = float(adjusted_rand_score(labels_true, labels_pred))
    nmi = float(
        normalized_mutual_info_score(labels_true, labels_pred, average_method="arithmetic")
    )
    purity = _cluster_purity(labels_true, labels_pred)
    return {"ari": ari, "nmi": nmi, "purity": purity}


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def compute_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    labels_true: np.ndarray | None = None,
    silhouette_sample_size: int | None = 3000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compute Silhouette, Calinski-Harabasz, Davies-Bouldin, and (optionally)
    ARI, NMI, and Purity on the same space used for clustering.

    For large n, Silhouette is subsampled for speed (deterministic via random_state).
    CH/DB are relatively cheap and are computed on the full set.

    Parameters
    ----------
    X : feature array used for clustering (e.g. VAE latent means)
    labels : cluster assignments from K-Means / GMM
    labels_true : optional pseudo-ground-truth labels; if provided, ARI / NMI /
        Purity are also returned.
    silhouette_sample_size : max samples for Silhouette; None = use all.
    random_state : RNG seed for subsampling.
    """
    n = X.shape[0]
    unique = np.unique(labels)
    if len(unique) < 2:
        result: dict[str, Any] = {
            "silhouette": float("nan"),
            "calinski_harabasz": float("nan"),
            "davies_bouldin": float("nan"),
            "n_samples_metric": 0,
        }
        if labels_true is not None:
            result.update({"ari": float("nan"), "nmi": float("nan"), "purity": float("nan")})
        return result

    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)

    if silhouette_sample_size is not None and n > silhouette_sample_size:
        rng = np.random.default_rng(random_state)
        sil = float("nan")
        n_metric = silhouette_sample_size

        # Guard against rare cases where a uniform subsample contains only one label.
        for _ in range(10):
            idx = rng.choice(n, size=silhouette_sample_size, replace=False)
            labels_s = labels[idx]
            if len(np.unique(labels_s)) >= 2:
                X_s = X[idx]
                sil = float(silhouette_score(X_s, labels_s, random_state=random_state))
                break
    else:
        sil = float(silhouette_score(X, labels, random_state=random_state))
        n_metric = n

    result = {
        "silhouette": float(sil),
        "calinski_harabasz": float(ch),
        "davies_bouldin": float(db),
        "n_samples_metric": n_metric,
    }

    if labels_true is not None:
        result.update(compute_supervised_metrics(labels_true, labels))

    return result


def metrics_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)
