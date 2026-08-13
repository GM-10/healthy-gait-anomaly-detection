"""
semg_pipeline/models/lstm_model.py

LSTM Autoencoder for per-channel sEMG anomaly detection.

Architecture (per channel):
    Encoder : LSTM(input_size=1, hidden_size=64, num_layers=1, batch_first=True)
              → take last hidden state h_n  shape (batch, 64)
    Bridge  : repeat h_n across window_size time steps
    Decoder : LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
              TimeDistributed(Linear(64, 1))  → reconstruction

    [Semi-supervised extension]
    Classifier head (optional):
              Linear(hidden_size, 1) + Sigmoid  applied to h_n
              Trained jointly: total_loss = recon_loss + lambda_cls * bce_loss

Input  shape : (batch, window_size, 1)   — univariate per channel
Output shape : (batch, window_size, 1)
Loss         : MSE  [+ optional lambda_cls × BCE]
Anomaly score: MSE between input and reconstruction (per window)
               OR sigmoid classifier probability via anomaly_probability()

One model instance is trained independently per sEMG channel (9 total).
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42


def _set_seeds() -> None:
    import random
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Model architecture
# ─────────────────────────────────────────────────────────────────────────────

class LSTMAutoencoder(nn.Module):
    """
    Sequence-to-sequence LSTM autoencoder with optional classifier head.

    The encoder compresses the input sequence into a fixed-size context vector
    (h_n).  The decoder reconstructs the full sequence from h_n.  When
    use_classifier=True a lightweight Linear(hidden_size, 1)+Sigmoid head is
    also attached to h_n for semi-supervised joint training.
    """

    def __init__(
        self,
        window_size: int = 1920,
        hidden_size: int = 64,
        use_classifier: bool = False,
    ):
        super().__init__()
        self.window_size     = window_size
        self.hidden_size     = hidden_size
        self.use_classifier  = use_classifier

        # Encoder: read (window_size, 1) → hidden state (64,)
        self.encoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # Decoder: read (window_size, 64) → (window_size, 1)
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.output_layer = nn.Linear(hidden_size, 1)

        # Optional classifier head: h_n (batch, hidden_size) → prob (batch, 1)
        if use_classifier:
            self.classifier_head = nn.Sequential(
                nn.Linear(hidden_size, 1),
                # Sigmoid is applied in anomaly_probability() for BCEWithLogitsLoss
            )
        else:
            self.classifier_head = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Core encode helpers (shared between forward and anomaly_probability)
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode x and return the context vector h_n.

        Parameters
        ----------
        x : torch.Tensor  shape (batch, window_size, 1)

        Returns
        -------
        context : torch.Tensor  shape (batch, hidden_size)
        """
        _, (h_n, _) = self.encoder(x)   # h_n: (1, batch, hidden_size)
        return h_n.squeeze(0)           # (batch, hidden_size)

    def classify(self, context: torch.Tensor) -> torch.Tensor:
        """
        Run the classifier head on a pre-computed context vector.

        Parameters
        ----------
        context : torch.Tensor  shape (batch, hidden_size)

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
                "Reconstruct LSTMModel with use_classifier=True."
            )
        return self.classifier_head(context)  # (batch, 1)

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
        context = self.encode(x)                                           # (batch, 64)
        expanded = context.unsqueeze(1).repeat(1, self.window_size, 1)    # (batch, T, 64)
        dec_out, _ = self.decoder(expanded)                                # (batch, T, 64)
        recon = self.output_layer(dec_out)                                 # (batch, T, 1)
        return recon


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper with fit / score interface
# ─────────────────────────────────────────────────────────────────────────────

class LSTMModel:
    """
    High-level wrapper around LSTMAutoencoder.

    One instance should be created per sEMG channel:
        models = [LSTMModel(ch_name) for ch_name in SEMG_CHANNELS]
        models[0].fit(train_windows[:, :, 0:1], val_windows[:, :, 0:1])
        errors = models[0].score(test_windows[:, :, 0:1])

    Semi-supervised mode (optional):
        m = LSTMModel(ch_name, use_classifier=True, lambda_cls=0.5)
        m.fit(train_windows, val_windows)   # augmentation happens internally
        probs = m.anomaly_probability(test_windows)   # [0, 1] per window

    All new parameters default to the original unsupervised behaviour so
    existing call sites (notebook, run_pipeline) work without modification.
    """

    def __init__(
        self,
        channel_name: str,
        window_size: int = 1920,
        hidden_size: int = 64,
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
        self.hidden_size           = hidden_size
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
        self.model: Optional[LSTMAutoencoder] = None

    def fit(
        self,
        train_windows: np.ndarray,
        val_windows: Optional[np.ndarray] = None,
    ) -> "LSTMModel":
        """
        Train the LSTM autoencoder (and optional classifier head) on windows.

        Parameters
        ----------
        train_windows : np.ndarray
            Shape (N_train, window_size, 1)  — single-channel windows.
        val_windows : np.ndarray, optional
            Shape (N_val, window_size, 1)    — for early stopping (reconstruction
            loss only; classifier does not affect early stopping).

        Returns
        -------
        self

        Notes
        -----
        When use_classifier=True the training set is internally augmented:
        a fraction of clean windows are replaced by synthetic anomalies and
        labeled 1 (anomalous).  The combined loss is:

            total_loss = recon_loss + lambda_cls * bce_loss

        Early stopping monitors the reconstruction-only val loss so that the
        stopping criterion is not contaminated by the classifier branch.
        """
        _set_seeds()
        self.model = LSTMAutoencoder(
            self.window_size, self.hidden_size, self.use_classifier
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
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
                f"[LSTM][{self.channel_name}] Semi-supervised augmentation: "
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
                    batch_y = batch_y.to(self.device)    # (batch,)

                    optimizer.zero_grad()

                    # Reconstruction loss
                    context = self.model.encode(batch_x)         # (batch, 64)
                    expanded = context.unsqueeze(1).repeat(
                        1, self.window_size, 1
                    )                                             # (batch, T, 64)
                    dec_out, _ = self.model.decoder(expanded)
                    recon = self.model.output_layer(dec_out)      # (batch, T, 1)
                    recon_loss = mse_criterion(recon, batch_x)

                    # Classification loss
                    prob = self.model.classify(context).squeeze(1)  # (batch,)
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
                        f"[LSTM][{self.channel_name}] "
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

        # Restore best weights
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
            Shape (N,)  — MSE per window.
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
                # MSE per window
                mse = ((recon - batch_x) ** 2).mean(dim=[1, 2])
                errors.append(mse.cpu().numpy())

        return np.concatenate(errors, axis=0)  # (N,)

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
                "Reconstruct LSTMModel with use_classifier=True and re-train."
            )

        self.model.eval()
        X = torch.tensor(windows, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X), batch_size=self.batch_size, shuffle=False)

        probs = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                context = self.model.encode(batch_x)
                logits = self.model.classify(context).squeeze(1)  # (batch,)
                prob = torch.sigmoid(logits)
                probs.append(prob.cpu().numpy())

        return np.concatenate(probs, axis=0)  # (N,)

    def save(self, path: str) -> None:
        """Save model weights to path (includes classifier head if present)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> "LSTMModel":
        """
        Load model weights from path.

        Backward compatibility: if the checkpoint was saved without a classifier
        head (old unsupervised model), missing classifier keys are initialised
        from scratch and a warning is emitted.  score() still works; calling
        anomaly_probability() after loading an old checkpoint will raise
        RuntimeError unless use_classifier=True and the key is present.
        """
        _set_seeds()
        self.model = LSTMAutoencoder(
            self.window_size, self.hidden_size, self.use_classifier
        ).to(self.device)
        state_dict = torch.load(path, map_location=self.device)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(
                f"[LSTM][{self.channel_name}] load(): {len(missing)} missing keys "
                f"(classifier head not in checkpoint — score() works, "
                f"anomaly_probability() requires re-training with use_classifier=True). "
                f"Missing: {missing}"
            )
        if unexpected:
            logger.warning(
                f"[LSTM][{self.channel_name}] load(): {len(unexpected)} unexpected keys "
                f"ignored: {unexpected}"
            )
        self.model.eval()
        return self
