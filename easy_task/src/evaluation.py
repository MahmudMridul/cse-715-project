"""Clustering metrics: Silhouette and Calinski-Harabasz."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, silhouette_score


def compute_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    silhouette_sample_size: int | None = 3000,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Compute Silhouette and Calinski-Harabasz on the same space used for clustering.

    For large n, Silhouette is subsampled for speed (deterministic with random_state).
    """
    n = X.shape[0]
    unique = np.unique(labels)
    if len(unique) < 2:
        return {
            "silhouette": float("nan"),
            "calinski_harabasz": float("nan"),
            "n_samples_metric": 0,
        }

    ch = calinski_harabasz_score(X, labels)

    if silhouette_sample_size is not None and n > silhouette_sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=silhouette_sample_size, replace=False)
        X_s = X[idx]
        labels_s = labels[idx]
        sil = silhouette_score(X_s, labels_s, random_state=random_state)
        n_metric = silhouette_sample_size
    else:
        sil = silhouette_score(X, labels, random_state=random_state)
        n_metric = n

    return {
        "silhouette": float(sil),
        "calinski_harabasz": float(ch),
        "n_samples_metric": n_metric,
    }


def metrics_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)
