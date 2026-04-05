"""Variational autoencoder for tabular music features."""

from __future__ import annotations

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
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(z))
        return self.fc_out(h)


class VAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
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


def train_vae(
    model: VAE,
    train_loader: DataLoader,
    epochs: int,
    device: torch.device,
    lr: float = 1e-3,
    kl_weight: float = 1.0,
) -> list[float]:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []

    for _ in tqdm(range(epochs), desc="VAE train"):
        epoch_loss = 0.0
        n_batches = 0
        for (batch,) in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar, kl_weight=kl_weight)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        history.append(epoch_loss / max(n_batches, 1))

    return history


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
