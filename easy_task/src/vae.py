"""Variational autoencoder for tabular music features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_hidden_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_hidden_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers = []
        in_dim = latent_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_hidden_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            input_dim,
            hidden_dim,
            latent_dim,
            num_hidden_layers=num_hidden_layers,
            dropout=dropout,
        )
        self.decoder = Decoder(
            latent_dim,
            hidden_dim,
            input_dim,
            num_hidden_layers=num_hidden_layers,
            dropout=dropout,
        )
        self.latent_dim = latent_dim

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1.0,
) -> torch.Tensor:
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl


@dataclass
class TrainHistory:
    """Training history returned by train_vae."""
    train_losses: list[float]
    val_losses: list[float]


def train_vae(
    model: VAE,
    train_loader: DataLoader,
    epochs: int,
    device: torch.device,
    lr: float = 1e-3,
    kl_weight: float = 1.0,
    kl_warmup_epochs: int = 0,
    val_loader: DataLoader | None = None,
    early_stop_patience: int | None = None,
) -> TrainHistory:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss: float | None = None
    patience_left = early_stop_patience

    for epoch in tqdm(range(epochs), desc="VAE train"):
        if kl_warmup_epochs and kl_warmup_epochs > 0:
            warmup_factor = min(1.0, (epoch + 1) / float(kl_warmup_epochs))
            current_kl_weight = kl_weight * warmup_factor
        else:
            current_kl_weight = kl_weight

        epoch_loss = 0.0
        n_batches = 0
        for (batch,) in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar, kl_weight=current_kl_weight)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_epoch_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(train_epoch_loss)

        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_batches = 0
                for (val_batch,) in val_loader:
                    val_batch = val_batch.to(device)
                    recon, mu, logvar = model(val_batch)
                    v_loss = vae_loss(
                        recon, val_batch, mu, logvar, kl_weight=current_kl_weight
                    )
                    val_loss += v_loss.item()
                    val_batches += 1
                val_loss = val_loss / max(val_batches, 1)
            val_losses.append(val_loss)
            model.train()

            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_left = early_stop_patience
            elif early_stop_patience is not None:
                patience_left -= 1
                if patience_left <= 0:
                    break

    return TrainHistory(train_losses=train_losses, val_losses=val_losses)


@torch.no_grad()
def encode_mu(model: VAE, X: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    """Return latent means for all rows (stable for clustering)."""
    model.eval()
    n = X.shape[0]
    out = np.empty((n, model.latent_dim), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = torch.from_numpy(X[start:end]).to(device)
        mu, _ = model.encoder(xb)
        out[start:end] = mu.cpu().numpy()
    return out
