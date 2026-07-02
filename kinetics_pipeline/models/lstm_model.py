"""
kinetics_pipeline/models/lstm_model.py

LSTM Autoencoder for per-channel kinematics and kinetics anomaly detection.

Architecture (per channel):
    Encoder : LSTM(input_size=1, hidden_size=64, num_layers=1, batch_first=True)
              → take last hidden state h_n  shape (batch, 64)
    Bridge  : repeat h_n across window_size time steps
    Decoder : LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
              TimeDistributed(Linear(64, 1))  → reconstruction

Input  shape : (batch, window_size, 1)   — univariate per channel
Output shape : (batch, window_size, 1)
Loss         : MSE
Anomaly score: MSE between input and reconstruction (per window)

One model instance is trained independently per channel (16 total).
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

RANDOM_SEED = 42


def _set_seeds() -> None:
    import random
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


class LSTMAutoencoder(nn.Module):
    """
    Sequence-to-sequence LSTM autoencoder.
    """

    def __init__(self, window_size: int = 180, hidden_size: int = 64):
        super().__init__()
        self.window_size = window_size
        self.hidden_size = hidden_size

        self.encoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        # Encode: → (batch, window_size, hidden) + (h_n, c_n)
        _, (h_n, _) = self.encoder(x)               # h_n: (1, batch, 64)

        # Repeat context vector across all time steps
        context = h_n.squeeze(0)                     # (batch, 64)
        context = context.unsqueeze(1).repeat(1, self.window_size, 1)  # (batch, T, 64)

        # Decode
        dec_out, _ = self.decoder(context)           # (batch, T, 64)
        recon = self.output_layer(dec_out)            # (batch, T, 1)
        return recon


class LSTMModel:
    """
    High-level wrapper around LSTMAutoencoder.
    """

    def __init__(
        self,
        channel_name: str,
        window_size: int = 180,
        hidden_size: int = 64,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
        patience: int = 5,
        device: Optional[str] = None,
    ):
        self.channel_name = channel_name
        self.window_size  = window_size
        self.hidden_size  = hidden_size
        self.lr           = lr
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.patience     = patience

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        _set_seeds()
        self.model: Optional[LSTMAutoencoder] = None

    def fit(
        self,
        train_windows: np.ndarray,
        val_windows: Optional[np.ndarray] = None,
    ) -> "LSTMModel":
        _set_seeds()
        self.model = LSTMAutoencoder(self.window_size, self.hidden_size).to(self.device)
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

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(1, self.epochs + 1):
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
                        f"[LSTM][{self.channel_name}] "
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
                            f"[LSTM][{self.channel_name}] "
                            f"Early stopping at epoch {epoch} (patience={self.patience})"
                        )
                        break
            else:
                if epoch % 5 == 0 or epoch == 1:
                    logger.info(
                        f"[LSTM][{self.channel_name}] "
                        f"Epoch {epoch:3d}/{self.epochs} "
                        f"train={train_loss:.6f}"
                    )

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        return self

    def score(self, windows: np.ndarray) -> np.ndarray:
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
                mse = ((recon - batch_x) ** 2).mean(dim=[1, 2])
                errors.append(mse.cpu().numpy())

        return np.concatenate(errors, axis=0)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> "LSTMModel":
        _set_seeds()
        self.model = LSTMAutoencoder(self.window_size, self.hidden_size).to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        return self
