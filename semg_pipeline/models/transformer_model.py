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

    [Semi-supervised extension]
    Classifier head (optional):
              Mean-pool encoder output across time → (batch, d_model)
              Linear(d_model, 1) + Sigmoid
              Trained jointly: total_loss = recon_loss + lambda_cls * bce_loss

    Mean-pool rationale: anomalies (amplitude scale, time warp, time shift)
    are often localised in time.  Mean-pooling aggregates signal across the
    full sequence without requiring the model to learn an attention location
    (CLS token) or risk fixating on a single spike (max-pool).

Input  shape : (batch, window_size, 1)
Output shape : (batch, window_size, 1)
Loss         : MSE  [+ optional lambda_cls × BCE]
Anomaly score: MSE between input and reconstruction (per window)
               OR sigmoid classifier probability via anomaly_probability()

One model instance is trained independently per sEMG channel (9 total)."""

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
    Transformer autoencoder for 1D time-series reconstruction with optional
    semi-supervised classifier head.

    The encoder processes the positionally-encoded input through a standard
    TransformerEncoder stack. A linear projection maps the encoder output
    back to the original 1D signal space.

    When use_classifier=True, a mean-pool over the encoder's time-axis output
    produces a (batch, d_model) latent vector that feeds into a lightweight
    Linear(d_model, 1)+Sigmoid classifier head.
    """

    def __init__(
        self,
        window_size: int = 1920,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        use_classifier: bool = False,
    ):
        super().__init__()
        self.d_model        = d_model
        self.use_classifier = use_classifier

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

        # Optional classifier head: mean-pooled latent (batch, d_model) → prob (batch, 1)
        if use_classifier:
            self.classifier_head = nn.Sequential(
                nn.Linear(d_model, 1),
                # Sigmoid is applied in anomaly_probability() for BCEWithLogitsLoss
            )
        else:
            self.classifier_head = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Core encode helpers (shared between forward and anomaly_probability)
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode x and return the mean-pooled latent vector.

        Parameters
        ----------
        x : torch.Tensor  shape (batch, window_size, 1)

        Returns
        -------
        latent : torch.Tensor  shape (batch, d_model)
            Mean-pool of encoder output across the time axis.
        """
        h = self.input_proj(x)              # (batch, T, d_model)
        h = self.pos_enc(h)                 # (batch, T, d_model)
        h = self.transformer_encoder(h)     # (batch, T, d_model)
        return h.mean(dim=1)                # (batch, d_model)  — mean-pool

    def classify(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Run the classifier head on a pre-computed latent vector.

        Parameters
        ----------
        latent : torch.Tensor  shape (batch, d_model)

        Returns
        -------
        prob : torch.Tensor  shape (batch, 1)  values in [0, 1]

        Raises
        ------
        RuntimeError  if the model was built without use_classifier=True.
        """
        if self.classifier_head is None:
            raise RuntimeError(
                "classify() called on a model built with use_classifier=False. "
                "Reconstruct TransformerModel with use_classifier=True."
            )
        return self.classifier_head(latent)  # (batch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruction forward pass (unchanged signature).

        Parameters
        ----------
        x : torch.Tensor  shape (batch, window_size, 1)

        Returns
        -------
        torch.Tensor  shape (batch, window_size, 1)
        """
        # Project to d_model
        h = self.input_proj(x)          # (batch, T, d_model)

        # Add positional encoding
        h = self.pos_enc(h)             # (batch, T, d_model)

        # Transformer encoder (self-attention + FFN)
        h = self.transformer_encoder(h) # (batch, T, d_model)

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

    Semi-supervised mode (optional):
        m = TransformerModel(ch_name, use_classifier=True, lambda_cls=0.5)
        m.fit(train_windows, val_windows)   # augmentation happens internally
        probs = m.anomaly_probability(test_windows)   # [0, 1] per window

    All new parameters default to the original unsupervised behaviour so
    existing call sites (notebook, run_pipeline) work without modification.
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
        # ── Semi-supervised options (all backward-compatible defaults) ──
        use_classifier: bool = False,
        lambda_cls: float = 0.5,
        augmentation_fraction: float = 0.20,
        augmentation_severity: str = "moderate",
        augmentation_types: Optional[List[str]] = None,
        pos_weight: Optional[float] = None,
    ):
        self.channel_name          = channel_name
        self.window_size           = window_size
        self.d_model               = d_model
        self.nhead                 = nhead
        self.num_encoder_layers    = num_encoder_layers
        self.dim_feedforward       = dim_feedforward
        self.dropout               = dropout
        self.lr                    = lr
        self.epochs                = epochs
        self.batch_size            = batch_size
        self.patience              = patience
        self.use_classifier        = use_classifier
        self.lambda_cls            = lambda_cls
        self.augmentation_fraction = augmentation_fraction
        self.augmentation_severity = augmentation_severity
        self.augmentation_types    = augmentation_types
        self.pos_weight            = pos_weight
        self._last_pos_weight      = None  # For unit testing

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
            use_classifier=self.use_classifier,
        ).to(self.device)

    def fit(
        self,
        train_windows: np.ndarray,
        val_windows: Optional[np.ndarray] = None,
    ) -> "TransformerModel":
        """
        Train the Transformer autoencoder (and optional classifier head) on windows.

        Parameters
        ----------
        train_windows : np.ndarray
            Shape (N_train, window_size, 1)
        val_windows : np.ndarray, optional
            Shape (N_val, window_size, 1) — for early stopping (reconstruction
            loss only; classifier does not affect early stopping).

        Returns
        -------
        self

        Notes
        -----
        When use_classifier=True the training set is internally augmented:
        a fraction of clean windows are replaced by synthetic anomalies and
        labeled 1 (anomalous). The augmented set is materialised ONCE with
        RANDOM_SEED=42 for reproducibility. The combined loss is:

            total_loss = recon_loss + lambda_cls * bce_loss

        Known limitation: early stopping monitors reconstruction-only val loss,
        so the classifier head may be underfit/overfit when training stops.
        Future work: monitor val_recon + lambda_cls * val_bce, or val_bce alone
        with an augmented val set, once the semi-supervised path is validated.
        """
        _set_seeds()
        self.model = self._build_model()
        optimizer  = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        mse_criterion = nn.MSELoss()

        # ── Build training tensors (augment if semi-supervised) ────────────
        if self.use_classifier:
            from semg_pipeline.augmenter import augment_with_anomalies
            aug_windows, aug_labels = augment_with_anomalies(
                train_windows,
                fraction=self.augmentation_fraction,
                anomaly_types=self.augmentation_types,
                severity=self.augmentation_severity,
            )
            X_train = torch.tensor(aug_windows, dtype=torch.float32)
            Y_train = torch.tensor(aug_labels,  dtype=torch.float32)  # (N,)
            
            # Compute dynamic pos_weight
            actual_anomalies = int(aug_labels.sum())
            total_windows = len(aug_labels)
            if self.pos_weight is not None:
                pw_val = self.pos_weight
            else:
                actual_fraction = actual_anomalies / total_windows if total_windows > 0 else 0
                pw_val = (1.0 - actual_fraction) / actual_fraction if actual_fraction > 0 else 1.0
            
            self._last_pos_weight = pw_val
            pw_tensor = torch.tensor([pw_val], dtype=torch.float32, device=self.device)
            bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)

            logger.info(
                f"[Transformer][{self.channel_name}] Semi-supervised augmentation: "
                f"{actual_anomalies}/{total_windows} anomalous windows "
                f"(fraction={self.augmentation_fraction}, "
                f"severity={self.augmentation_severity}, pos_weight={pw_val:.2f})"
            )
        else:
            X_train = torch.tensor(train_windows, dtype=torch.float32)
            Y_train = None

        if Y_train is not None:
            train_loader = DataLoader(
                TensorDataset(X_train, Y_train),
                batch_size=self.batch_size,
                shuffle=True,
            )
        else:
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
            # ── Training ──────────────────────────────────────────────────
            self.model.train()
            train_loss_total = 0.0

            if self.use_classifier:
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)     # (batch,)

                    optimizer.zero_grad()

                    # Reconstruction loss (full forward pass)
                    recon = self.model(batch_x)            # (batch, T, 1)
                    recon_loss = mse_criterion(recon, batch_x)

                    # Classification loss (mean-pool latent)
                    latent = self.model.encode(batch_x)             # (batch, d_model)
                    prob   = self.model.classify(latent).squeeze(1) # (batch,)
                    cls_loss = bce_criterion(prob, batch_y)          # type: ignore[misc]

                    loss = recon_loss + self.lambda_cls * cls_loss
                    loss.backward()
                    optimizer.step()
                    train_loss_total += loss.item() * len(batch_x)
            else:
                for (batch_x,) in train_loader:
                    batch_x = batch_x.to(self.device)
                    optimizer.zero_grad()
                    recon = self.model(batch_x)
                    loss  = mse_criterion(recon, batch_x)
                    loss.backward()
                    optimizer.step()
                    train_loss_total += loss.item() * len(batch_x)

            train_loss = train_loss_total / len(X_train)

            # ── Validation + early stopping (reconstruction loss only) ───
            # Known limitation: classifier head fitness is not monitored here.
            # See docstring Notes for future improvement path.
            if has_val:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for (batch_x,) in val_loader:
                        batch_x = batch_x.to(self.device)
                        recon = self.model(batch_x)
                        val_loss += mse_criterion(recon, batch_x).item() * len(batch_x)
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
                    best_state = {
                        k: v.cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
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

        This method is unchanged from the original unsupervised implementation.
        It always returns reconstruction error regardless of whether the
        classifier head is enabled.

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

    def anomaly_probability(self, windows: np.ndarray) -> np.ndarray:
        """
        Return the classifier head's sigmoid output for each window.

        Values close to 1 indicate the model believes the window is anomalous.
        This is the classifier-based alternative to reconstruction-error
        thresholding.

        Parameters
        ----------
        windows : np.ndarray
            Shape (N, window_size, 1)

        Returns
        -------
        probs : np.ndarray
            Shape (N,)  — values in [0, 1].

        Raises
        ------
        RuntimeError
            If fit() has not been called, or if the model was built / loaded
            without use_classifier=True.
        """
        if self.model is None:
            raise RuntimeError("Call fit() before anomaly_probability().")
        if not self.use_classifier or self.model.classifier_head is None:
            raise RuntimeError(
                "anomaly_probability() requires use_classifier=True. "
                "Reconstruct TransformerModel with use_classifier=True and re-train."
            )

        self.model.eval()
        X = torch.tensor(windows, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X), batch_size=self.batch_size, shuffle=False)

        probs = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                latent  = self.model.encode(batch_x)
                logits = self.model.classify(latent).squeeze(1)  # (batch,)
                prob = torch.sigmoid(logits)
                probs.append(prob.cpu().numpy())

        return np.concatenate(probs, axis=0)  # (N,)

    def save(self, path: str) -> None:
        """Save model weights to path (includes classifier head if present)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> "TransformerModel":
        """
        Load model weights from path.

        Backward compatibility: if the checkpoint was saved without a classifier
        head (old unsupervised model), missing classifier keys are initialised
        from scratch and a warning is emitted.  score() still works; calling
        anomaly_probability() after loading an old checkpoint will raise
        RuntimeError unless use_classifier=True and the key is present.
        """
        _set_seeds()
        self.model = self._build_model()
        state_dict = torch.load(path, map_location=self.device)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(
                f"[Transformer][{self.channel_name}] load(): {len(missing)} missing keys "
                f"(classifier head not in checkpoint — score() works, "
                f"anomaly_probability() requires re-training with use_classifier=True). "
                f"Missing: {missing}"
            )
        if unexpected:
            logger.warning(
                f"[Transformer][{self.channel_name}] load(): {len(unexpected)} unexpected keys "
                f"ignored: {unexpected}"
            )
        self.model.eval()
        return self
