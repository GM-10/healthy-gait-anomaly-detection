"""
kinetics_pipeline/models/sarima_model.py

SARIMA classical baseline for kinematics + kinetics anomaly detection.

One ARIMA model is fitted per channel using pmdarima.auto_arima().
Reconstruction is performed by one-step-ahead prediction over each test window.
The anomaly score for a window is the MSE between the predicted and actual signal.

Training is limited to a configurable subset of subjects (SARIMA_MAX_TRAIN_SUBJECTS = 5 by default)
and subsampled windows to keep execution times reasonable.
"""

import os
import pickle
import logging
import warnings
import numpy as np
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SARIMA_MAX_TRAIN_SUBJECTS: int = 5

# Suppress pmdarima/statsmodels convergence warnings to keep output readable
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class SARIMAModel:
    """
    Per-channel SARIMA anomaly detector.

    Usage
    -----
        model = SARIMAModel(channel_names=SIGNAL_COLUMNS)
        model.fit(train_windows)
        errors = model.score(test_windows)
    """

    def __init__(
        self,
        channel_names: List[str],
        max_windows_per_channel: int = 200,
        model_dir: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        channel_names : List[str]
            Names of the kinematic and kinetic channels.
        max_windows_per_channel : int, default 200
            Maximum number of training windows to use per channel.
        model_dir : str, optional
            Directory to save/load fitted ARIMA models.
        """
        self.channel_names = channel_names
        self.max_windows_per_channel = max_windows_per_channel
        self.model_dir = model_dir
        self._models: Dict[int, object] = {}   # channel_idx → fitted ARIMA

    def fit(self, windows: np.ndarray) -> "SARIMAModel":
        """
        Fit one ARIMA model per channel on a representative training set.

        Parameters
        ----------
        windows : np.ndarray
            Shape (N, window_size, num_channels). Training windows.

        Returns
        -------
        self
        """
        try:
            import pmdarima as pm
        except ImportError:
            raise ImportError(
                "pmdarima is required for SARIMAModel. "
                "Install it with: pip install pmdarima"
            )

        n_windows, window_size, n_channels = windows.shape
        n_use = min(n_windows, self.max_windows_per_channel)

        # Subsample windows deterministically
        rng = np.random.default_rng(42)
        indices = rng.choice(n_windows, size=n_use, replace=False)
        indices.sort()
        subset = windows[indices]   # (n_use, window_size, n_channels)

        for ch_idx in range(n_channels):
            ch_name = self.channel_names[ch_idx]

            # Check if we already have a saved model
            if self.model_dir is not None:
                saved_path = self._model_path(ch_idx)
                if os.path.exists(saved_path):
                    logger.info(f"[SARIMA] Loading cached model for {ch_name}")
                    self._models[ch_idx] = self._load_arima(saved_path)
                    continue

            logger.info(
                f"[SARIMA] Fitting channel {ch_idx + 1}/{n_channels}: "
                f"{ch_name} on {n_use} windows …"
            )

            # Concatenate all training windows for this channel into one long series
            long_series = subset[:, :, ch_idx].flatten()   # (n_use × window_size,)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                arima = pm.auto_arima(
                    long_series,
                    start_p=1, start_q=1,
                    max_p=3,   max_q=3,
                    seasonal=False,
                    d=None,             # auto-detect differencing
                    information_criterion="aic",
                    stepwise=True,
                    error_action="ignore",
                    suppress_warnings=True,
                )

            self._models[ch_idx] = arima

            if self.model_dir is not None:
                os.makedirs(self.model_dir, exist_ok=True)
                self._save_arima(arima, self._model_path(ch_idx))
                logger.info(f"[SARIMA] Saved model for {ch_name}")

        return self

    def score(self, windows: np.ndarray) -> np.ndarray:
        """
        Compute per-window, per-channel reconstruction MSE.

        Parameters
        ----------
        windows : np.ndarray
            Shape (N, window_size, num_channels).

        Returns
        -------
        errors : np.ndarray
            Shape (N, num_channels). Per-window MSE for each channel.
        """
        n_windows, window_size, n_channels = windows.shape
        errors = np.zeros((n_windows, n_channels), dtype=np.float64)

        n_context = window_size // 2
        n_pred    = window_size - n_context

        for ch_idx in range(n_channels):
            if ch_idx not in self._models:
                logger.warning(
                    f"[SARIMA] No model for channel {ch_idx} ({self.channel_names[ch_idx]}). "
                    "Skipping — errors will be 0."
                )
                continue

            arima = self._models[ch_idx]

            for w_idx in range(n_windows):
                context = windows[w_idx, :n_context, ch_idx]
                actual  = windows[w_idx, n_context:, ch_idx]

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        arima_updated = arima.update(context, maxiter=0)
                        forecast, _   = arima_updated.predict(n_periods=n_pred, return_conf_int=True)
                except Exception as e:
                    logger.debug(f"[SARIMA] Window {w_idx} ch {ch_idx} predict failed: {e}")
                    forecast = np.zeros(n_pred)

                errors[w_idx, ch_idx] = float(np.mean((forecast - actual) ** 2))

        return errors

    def _model_path(self, ch_idx: int) -> str:
        return os.path.join(self.model_dir, f"sarima_ch{ch_idx:02d}.pkl")

    @staticmethod
    def _save_arima(model, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def _load_arima(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

    def save(self, directory: str) -> None:
        """Persist all fitted ARIMA models to directory."""
        os.makedirs(directory, exist_ok=True)
        for ch_idx, model in self._models.items():
            self._save_arima(model, os.path.join(directory, f"sarima_ch{ch_idx:02d}.pkl"))

    def load(self, directory: str) -> "SARIMAModel":
        """Load all ARIMA models from a directory."""
        for ch_idx in range(len(self.channel_names)):
            path = os.path.join(directory, f"sarima_ch{ch_idx:02d}.pkl")
            if os.path.exists(path):
                self._models[ch_idx] = self._load_arima(path)
        return self
