"""K-Means and PCA + K-Means baseline."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def run_kmeans(
    X: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    n_init: int = 10,
) -> tuple[np.ndarray, KMeans]:
    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )
    labels = km.fit_predict(X)
    return labels, km


def pca_transform(
    X: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, PCA]:
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)
    return scores, pca


def pca_kmeans(
    X: np.ndarray,
    n_components: int,
    n_clusters: int,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, PCA, KMeans]:
    scores, pca = pca_transform(X, n_components)
    labels, km = run_kmeans(scores, n_clusters, random_state=random_state)
    return labels, scores, pca, km
