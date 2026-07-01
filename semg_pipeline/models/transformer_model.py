"""
semg_pipeline/models/transformer_model.py

Transformer Autoencoder for per-channel sEMG anomaly detection.

Architecture (per channel):
    Input projection : Linear(1, d_model=64)
    Positional enc.  : Standard sinusoidal (fixed, not learnable)
    Encoder          : nn.TransformerEncoder
                         d_model=64, nhead=4, num_encoder_layers=2,
                         dim_feedforward=128, dropout=0.1
    Decoder proj.    : Linear(64, 1)  → reconstruction

Input  shape : (batch, window_size, 1)
Output shape : (batch, window_size, 1)
Loss         : MSE
Anomaly score: MSE between input and reconstruction (per window)

One model instance is trained independently per sEMG channel (9 total).
"""

import os
import math
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional

logger = logging.getLogger(__name__)

RANDOM_SEED = 42


def _set_seeds() -> None:
    import random
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Positional encoding
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding (Vaswani et al., 2017).

    Adds fixed position-dependent sine/cosine signals to the embedded
    input so the Transformer can distinguish temporal positions.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)                   # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )                                                     # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                                  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# Model architecture
# ─────────────────────────────────────────────────────────────────────────────

class TransformerAutoencoder(nn.Module):
    """
    Transformer autoencoder for 1D time-series reconstruction.

    The encoder processes the positionally-encoded input through a standard
    TransformerEncoder stack. A linear projection maps the encoder output
    back to the original 1D signal space.
    """

    def __init__(
        self,
        window_size: int = 1920,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Input projection: 1 → d_model
        self.input_proj = nn.Linear(1, d_model)

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_len=window_size + 1, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,      # input shape: (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        # Output projection: d_model → 1
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (batch, window_size, 1)

        Returns
        -------
        torch.Tensor  shape (batch, window_size, 1)
        """
        # Project to d_model
        h = self.input_proj(x)          # (batch, T, 64)

        # Add positional encoding
        h = self.pos_enc(h)             # (batch, T, 64)

        # Transformer encoder (self-attention + FFN)
        h = self.transformer_encoder(h) # (batch, T, 64)

        # Project back to 1D
        recon = self.output_proj(h)     # (batch, T, 1)
        return recon


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper with fit / score interface
# ─────────────────────────────────────────────────────────────────────────────

class TransformerModel:
    """
    High-level wrapper around TransformerAutoencoder.

    One instance should be created per sEMG channel.
    """

    def __init__(
        self,
        channel_name: str,
        window_size: int = 1920,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
        patience: int = 5,
        device: Optional[str] = None,
    ):
        self.channel_name       = channel_name
        self.window_size        = window_size
        self.d_model            = d_model
        self.nhead              = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward    = dim_feedforward
        self.dropout            = dropout
        self.lr                 = lr
        self.epochs             = epochs
        self.batch_size         = batch_size
        self.patience           = patience

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        _set_seeds()
        self.model: Optional[TransformerAutoencoder] = None

    def _build_model(self) -> TransformerAutoencoder:
        return TransformerAutoencoder(
            window_size=self.window_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        ).to(self.device)

    def fit(
        self,
        train_windows: np.ndarray,
        val_windows: Optional[np.ndarray] = None,
    ) -> "TransformerModel":
        """
        Train the Transformer autoencoder on windows for this channel.

        Parameters
        ----------
        train_windows : np.ndarray
            Shape (N_train, window_size, 1)
        val_windows : np.ndarray, optional
            Shape (N_val, window_size, 1) — for early stopping.

        Returns
        -------
        self
        """
        _set_seeds()
        self.model = self._build_model()
        optimizer  = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion  = nn.MSELoss()

        X_train = torch.tensor(train_windows, dtype=torch.float32)
        train_loader = DataLoader(
            TensorDataset(X_train),
            batch_size=self.batch_size,
            shuffle=True,
        )

        has_val = val_windows is not None and len(val_windows) > 0
        if has_val:
            X_val = torch.tensor(val_windows, dtype=torch.float32)
            val_loader = DataLoader(
                TensorDataset(X_val),
                batch_size=self.batch_size,
                shuffle=False,
            )

        best_val_loss    = float("inf")
        patience_counter = 0
        best_state       = None

        for epoch in range(1, self.epochs + 1):
            # ── Training ──
            self.model.train()
            train_loss = 0.0
            for (batch_x,) in train_loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                recon = self.model(batch_x)
                loss  = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(batch_x)
            train_loss /= len(X_train)

            # ── Validation + early stopping ──
            if has_val:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for (batch_x,) in val_loader:
                        batch_x = batch_x.to(self.device)
                        recon = self.model(batch_x)
                        val_loss += criterion(recon, batch_x).item() * len(batch_x)
                val_loss /= len(X_val)

                if epoch % 5 == 0 or epoch == 1:
                    logger.info(
                        f"[Transformer][{self.channel_name}] "
                        f"Epoch {epoch:3d}/{self.epochs} "
                        f"train={train_loss:.6f}  val={val_loss:.6f}"
                    )

                if val_loss < best_val_loss:
                    best_val_loss    = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(
                            f"[Transformer][{self.channel_name}] "
                            f"Early stopping at epoch {epoch}"
                        )
                        break
            else:
                if epoch % 5 == 0 or epoch == 1:
                    logger.info(
                        f"[Transformer][{self.channel_name}] "
                        f"Epoch {epoch:3d}/{self.epochs} "
                        f"train={train_loss:.6f}"
                    )

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        return self

    def score(self, windows: np.ndarray) -> np.ndarray:
        """
        Compute per-window reconstruction MSE.

        Parameters
        ----------
        windows : np.ndarray
            Shape (N, window_size, 1)

        Returns
        -------
        errors : np.ndarray
            Shape (N,)
        """
        if self.model is None:
            raise RuntimeError("Call fit() before score().")

        self.model.eval()
        X = torch.tensor(windows, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X), batch_size=self.batch_size, shuffle=False)

        errors = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                recon   = self.model(batch_x)
                mse     = ((recon - batch_x) ** 2).mean(dim=[1, 2])
                errors.append(mse.cpu().numpy())

        return np.concatenate(errors, axis=0)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> "TransformerModel":
        _set_seeds()
        self.model = self._build_model()
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        return self
