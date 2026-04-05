"""Load Million Song subset CSV and build scaled feature matrices for the VAE."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Numeric feature columns (exclude IDs and text fields).
FEATURE_COLUMNS = [
    "Danceability",
    "Duration",
    "KeySignature",
    "KeySignatureConfidence",
    "Tempo",
    "TimeSignature",
    "TimeSignatureConfidence",
    "Year",
    "Energy",
    "ArtistFamiliarity",
    "Hotness",
    "end_of_fade_in",
    "key",
    "keyConfidence",
    "Loudness",
    "mode",
    "mode_confidence",
    "start_of_fade_out",
    "ArtistLatitude",
    "ArtistLongitude",
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
    return sub


def prepare_data(
    csv_path: str | Path,
    test_size: float = 0.1,
    random_state: int = 42,
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
