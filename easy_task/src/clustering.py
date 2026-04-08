"""Clustering helpers (CPU by default; optional GPU via cuML if installed)."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


def run_kmeans(
    X: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    n_init: int = 30,
    backend: str = "sklearn",
) -> tuple[np.ndarray, KMeans]:
    """
    Run KMeans and return labels + fitted estimator.

    backend:
      - 'sklearn' (default): CPU
      - 'cuml': GPU if RAPIDS cuML is installed (optional)
    """
    if backend == "cuml":
        try:
            from cuml.cluster import KMeans as cuKMeans  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "backend='cuml' requested but cuML is not available"
            ) from e
        km = cuKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init,
        )
        labels = km.fit_predict(X)
        # cuML may return GPU array; convert to numpy for downstream.
        labels = np.asarray(labels)
        return labels, km  # type: ignore[return-value]

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels = km.fit_predict(X)
    return labels, km


def pca_transform(
    X: np.ndarray,
    n_components: int,
    backend: str = "sklearn",
) -> tuple[np.ndarray, PCA]:
    n_features = int(X.shape[1])
    if n_components > n_features:
        # Keep sweeps robust when latent_dim > input feature dimension.
        # PCA can't exceed feature dimension, so we cap automatically.
        print(
            f"[warn] PCA n_components={n_components} > n_features={n_features}; "
            f"capping to {n_features}."
        )
        n_components = n_features
    if backend == "cuml":
        try:
            from cuml.decomposition import PCA as cuPCA  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "backend='cuml' requested but cuML is not available"
            ) from e
        pca = cuPCA(n_components=n_components, random_state=0)
        scores = pca.fit_transform(X)
        return np.asarray(scores), pca  # type: ignore[return-value]

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)
    return scores, pca


def pca_kmeans(
    X: np.ndarray,
    n_components: int,
    n_clusters: int,
    random_state: int = 42,
    n_init: int = 30,
    backend: str = "sklearn",
) -> tuple[np.ndarray, np.ndarray, PCA, KMeans]:
    scores, pca = pca_transform(X, n_components, backend=backend)
    labels, km = run_kmeans(
        scores, n_clusters, random_state=random_state, n_init=n_init, backend=backend
    )
    return labels, scores, pca, km


def run_gmm(
    X: np.ndarray,
    n_components: int,
    random_state: int = 42,
) -> tuple[np.ndarray, GaussianMixture]:
    """CPU Gaussian mixture baseline (often better for non-spherical clusters)."""
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    labels = gmm.fit_predict(X)
    return labels, gmm
