"""Load Million Song subset CSV and build scaled feature matrices for the VAE."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Iterable, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from torch.utils.data import DataLoader, TensorDataset

# Numeric feature columns used for the VAE.
# Excluded from the original 20:
#   - Danceability, Energy: identically 0 for all songs in this MSD subset
#     (Echo Nest did not compute them) → zero-variance, add no information.
#   - ArtistLatitude, ArtistLongitude: 63 % missing; median imputation would
#     create an artificial geographic cluster centred on the US.
FEATURE_COLUMNS = [
    "Duration",
    "KeySignature",
    "KeySignatureConfidence",
    "Tempo",
    "TimeSignature",
    "TimeSignatureConfidence",
    "Year",
    "ArtistFamiliarity",
    "Hotness",
    "end_of_fade_in",
    "key",
    "keyConfidence",
    "Loudness",
    "mode",
    "mode_confidence",
    "start_of_fade_out",
]

# Decade-bin labels used as pseudo ground-truth for ARI / NMI / Purity.
DECADE_LABEL_NAMES = [
    "Unknown era",   # Year == 0 (sentinel for missing)
    "Pre-1970s",     # < 1970
    "1970s",
    "1980s",
    "1990s",
    "2000s+",        # >= 2000
]


@dataclass
class PreparedData:
    """Train/val tensors and full scaled matrix for encoding + clustering."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_full: np.ndarray
    input_dim: int
    feature_columns: list[str]


def load_raw_frame(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df


def _feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[FEATURE_COLUMNS].copy()
    for col in sub.columns:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    # Year == 0 is a sentinel value meaning "unknown year" in the MSD; it does
    # not represent year 0 CE.  Treat it as missing so the imputer fills it
    # with the median of known years rather than letting it dominate clustering.
    if "Year" in sub.columns:
        sub.loc[sub["Year"] == 0, "Year"] = np.nan
    return sub


def _clip_outliers(
    X: np.ndarray,
    columns: Optional[Iterable[int]] = None,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> np.ndarray:
    """
    Simple winsorization: clip selected columns to given quantiles.

    Parameters
    ----------
    X:
        2D array of features (n_samples, n_features).
    columns:
        Iterable of column indices to clip. If None, all columns are clipped.
    lower_quantile / upper_quantile:
        Quantiles to use for clipping (per column).
    """
    X_clipped = X.copy()
    n_features = X_clipped.shape[1]
    if columns is None:
        cols = range(n_features)
    else:
        cols = columns

    for j in cols:
        col = X_clipped[:, j]
        # Ignore NaNs; clipping happens after imputation, so this is defensive.
        finite_mask = np.isfinite(col)
        if not np.any(finite_mask):
            continue
        lo = np.quantile(col[finite_mask], lower_quantile)
        hi = np.quantile(col[finite_mask], upper_quantile)
        X_clipped[:, j] = np.clip(col, lo, hi)
    return X_clipped


def prepare_data(
    csv_path: str | Path,
    test_size: float = 0.1,
    random_state: int = 42,
    clip_outliers: bool = False,
    clip_columns: Optional[Iterable[int]] = None,
    clip_lower_quantile: float = 0.01,
    clip_upper_quantile: float = 0.99,
    use_power_transform: bool = False,
) -> PreparedData:
    df = load_raw_frame(csv_path)
    X = _feature_matrix(df)
    idx = np.arange(len(X))
    train_idx, val_idx = train_test_split(
        idx, test_size=test_size, random_state=random_state
    )
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X.iloc[train_idx])
    X_imputed = imputer.transform(X)

    # Optional winsorization to reduce the impact of extreme outliers
    if clip_outliers:
        X_imputed = _clip_outliers(
            X_imputed,
            columns=clip_columns,
            lower_quantile=clip_lower_quantile,
            upper_quantile=clip_upper_quantile,
        )

    # Optional power transform (Yeo–Johnson) to reduce skewness before scaling.
    if use_power_transform:
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        pt.fit(X_imputed[train_idx])
        X_imputed = pt.transform(X_imputed)

    scaler = StandardScaler()
    scaler.fit(X_imputed[train_idx])
    X_scaled = scaler.transform(X_imputed)

    X_train = X_scaled[train_idx]
    X_val = X_scaled[val_idx]
    input_dim = X_train.shape[1]

    return PreparedData(
        X_train=X_train.astype(np.float32),
        X_val=X_val.astype(np.float32),
        X_full=X_scaled.astype(np.float32),
        input_dim=input_dim,
        feature_columns=list(FEATURE_COLUMNS),
    )


def make_train_loader(
    X_train: np.ndarray,
    batch_size: int = 256,
    shuffle: bool = True,
) -> DataLoader:
    t = torch.from_numpy(X_train)
    ds = TensorDataset(t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def numpy_to_tensor(X: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(X, dtype=np.float32))


def get_year_labels(csv_path: str | Path) -> np.ndarray:
    """Return integer decade-bin labels for every song in the CSV.

    Bins (matches DECADE_LABEL_NAMES):
        0 – Year == 0  (unknown / missing year)
        1 – Year  1 – 1969  (pre-1970s)
        2 – 1970 – 1979
        3 – 1980 – 1989
        4 – 1990 – 1999
        5 – 2000+

    These are used as pseudo ground-truth labels for ARI / NMI / Purity
    evaluation. All 10 000 songs receive a label (songs with Year == 0 get
    label 0, i.e. "Unknown era").
    """
    df = load_raw_frame(csv_path)
    year = pd.to_numeric(df["Year"], errors="coerce").fillna(0).astype(int)
    labels = np.zeros(len(year), dtype=int)
    labels[(year > 0) & (year < 1970)] = 1
    labels[(year >= 1970) & (year < 1980)] = 2
    labels[(year >= 1980) & (year < 1990)] = 3
    labels[(year >= 1990) & (year < 2000)] = 4
    labels[year >= 2000] = 5
    return labels
