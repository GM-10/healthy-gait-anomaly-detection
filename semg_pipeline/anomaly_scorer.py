"""
semg_pipeline/anomaly_scorer.py

Threshold computation and output CSV row builder for the sEMG pipeline.

Threshold rule (legacy, applied after training, using train set reconstruction errors):
    threshold = mean(train_errors) + 3 × std(train_errors)

The new evaluation_framework supports three threshold methods:
    mean_std    — the legacy rule above (default)
    percentile95 — 95th percentile of training errors
    percentile99 — 99th percentile of training errors

To enable multi-threshold evaluation, run the pipeline with --save_train_errors
to persist raw training errors as .npy files. evaluate.py reads these to compute
all threshold methods without re-running inference.

Output CSV schema (must match teammate's kinematics/kinetics pipeline exactly):
    subject_id, modality, channel_name, movement, window_id,
    window_start_time, window_end_time, reconstruction_error,
    is_synthetic_anomaly, anomaly_type, predicted_label, model_name,
    severity, threshold_method
"""

import json
import logging
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Threshold
# ─────────────────────────────────────────────────────────────────────────────

def compute_threshold(train_errors: np.ndarray, n_sigma: float = 3.0) -> float:
    """
    Compute the anomaly detection threshold from training reconstruction errors.

    threshold = mean(train_errors) + n_sigma × std(train_errors)

    Parameters
    ----------
    train_errors : np.ndarray
        1D array of reconstruction MSE values from the training set.
    n_sigma : float, default 3.0
        Number of standard deviations above the mean.

    Returns
    -------
    float
        Scalar threshold value.
    """
    if len(train_errors) == 0:
        raise ValueError("train_errors is empty — cannot compute threshold.")
    return float(np.mean(train_errors) + n_sigma * np.std(train_errors))


def label_windows(errors: np.ndarray, threshold: float) -> np.ndarray:
    """
    Binary-label each window based on its reconstruction error.

    Parameters
    ----------
    errors : np.ndarray
        1D array of reconstruction MSE values.
    threshold : float
        Decision boundary from compute_threshold().

    Returns
    -------
    np.ndarray
        Integer array of 0 (normal) or 1 (anomaly).
    """
    return (errors > threshold).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# Train error persistence — used by evaluate.py for multi-threshold evaluation
# ─────────────────────────────────────────────────────────────────────────────

def save_train_errors(
    errors: np.ndarray,
    channel_name: str,
    model_name: str,
    output_dir: str,
) -> str:
    """
    Save per-channel training reconstruction errors to a .npy file.

    These are loaded by evaluate.py to compute all three threshold methods
    (mean_std, percentile95, percentile99) without re-running model inference.

    Parameters
    ----------
    errors : np.ndarray
        1D array of training reconstruction errors (healthy windows only).
    channel_name : str
        Channel identifier (e.g. 'tensor_fascia_lata').
    model_name : str
        Model display name (e.g. 'LSTM', 'Transformer').
    output_dir : str
        Directory to save the .npy file (typically outputs/sEMG/).

    Returns
    -------
    str
        Absolute path to the saved file.
    """
    train_err_dir = os.path.join(output_dir, "train_errors")
    os.makedirs(train_err_dir, exist_ok=True)

    # Sanitize channel name for filename
    safe_ch = channel_name.replace(" ", "_").replace("/", "_")
    fname   = f"train_errors_{model_name}_{safe_ch}.npy"
    path    = os.path.join(train_err_dir, fname)
    np.save(path, errors.astype(np.float32))
    logger.info(f"  [TrainErrors] Saved {len(errors)} errors → {path}")
    return path


def load_train_errors(
    model_name: str,
    channel_names: List[str],
    output_dir: str,
) -> Dict[str, np.ndarray]:
    """
    Load persisted training reconstruction errors for all channels of a model.

    Parameters
    ----------
    model_name : str
        Model display name (e.g. 'LSTM').
    channel_names : list of str
        Channel names to load.
    output_dir : str
        Directory containing train_errors/ subfolder.

    Returns
    -------
    Dict[str, np.ndarray]
        {channel_name: 1D array of training errors}
        Missing channels are silently omitted.
    """
    train_err_dir = os.path.join(output_dir, "train_errors")
    result: Dict[str, np.ndarray] = {}

    for ch in channel_names:
        safe_ch = ch.replace(" ", "_").replace("/", "_")
        fname   = f"train_errors_{model_name}_{safe_ch}.npy"
        path    = os.path.join(train_err_dir, fname)
        if os.path.exists(path):
            result[ch] = np.load(path)
        else:
            logger.debug(f"  [TrainErrors] Not found: {path}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Output CSV row builder
# ─────────────────────────────────────────────────────────────────────────────

def build_output_rows(
    windows_meta: List[Dict],
    errors: np.ndarray,
    predicted_labels: np.ndarray,
    channel_name: str,
    subject_id: str,
    movement: str,
    model_name: str,
    is_synthetic_anomaly: int = 0,
    anomaly_type: str = "none",
    window_id_offset: int = 0,
    severity: float = 0.0,
) -> List[Dict]:
    """
    Build a list of output CSV row dicts for one channel, one model run.

    Each dict has the 13 output CSV columns (12 original + severity).

    Parameters
    ----------
    windows_meta : List[Dict]
        Metadata from create_semg_windows(), one dict per window.
    errors : np.ndarray
        Shape (N,) reconstruction MSE values.
    predicted_labels : np.ndarray
        Shape (N,) binary labels (0/1).
    channel_name : str
        Short channel name (e.g., 'tensor_fascia_lata').
    subject_id : str
        Subject ID string (e.g., 'Sub36').
    movement : str
        Movement code (e.g., 'WAK').
    model_name : str
        One of 'SARIMA', 'LSTM', 'Transformer'.
    is_synthetic_anomaly : int, default 0
        0 for clean windows, 1 for synthetically injected anomalies.
    anomaly_type : str, default 'none'
        One of 'none', 'amplitude_scale', 'time_warp', 'time_shift', 'combined'.
    window_id_offset : int, default 0
        Added to the window index to produce unique window_id values when
        combining clean + anomalous rows for the same trial.
    severity : float, default 0.0
        Numeric severity level (0.15=mild, 0.35=moderate, 0.60=severe, 0.0=clean).
        Used by evaluate.py for severity-stratified analysis.

    Returns
    -------
    List[Dict]
        One dict per window, ready to be passed to pd.DataFrame().
    """
    rows = []
    for i, (meta, err, pred) in enumerate(zip(windows_meta, errors, predicted_labels)):
        rows.append(
            {
                "subject_id":           subject_id,
                "modality":             "sEMG",
                "channel_name":         channel_name,
                "movement":             movement,
                "window_id":            i + window_id_offset,
                "window_start_time":    meta["start_time"],
                "window_end_time":      meta["end_time"],
                "reconstruction_error": float(err),
                "is_synthetic_anomaly": int(is_synthetic_anomaly),
                "anomaly_type":         anomaly_type,
                "predicted_label":      int(pred),
                "model_name":           model_name,
                "severity":             float(severity),
            }
        )
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: score + build in one call (for clean windows)
# ─────────────────────────────────────────────────────────────────────────────

def score_and_build_rows(
    model,
    windows: np.ndarray,
    windows_meta: List[Dict],
    channel_idx: int,
    channel_name: str,
    threshold: float,
    subject_id: str,
    movement: str,
    model_name: str,
    is_synthetic_anomaly: int = 0,
    anomaly_type: str = "none",
    window_id_offset: int = 0,
) -> Tuple[List[Dict], np.ndarray]:
    """
    Score a set of windows with the model and build output rows.

    Works for both LSTM/Transformer (which score per-channel slices)
    and SARIMA (which returns a 2D array).

    Parameters
    ----------
    model : LSTMModel | TransformerModel | SARIMAModel
        Fitted model instance.
    windows : np.ndarray
        Shape (N, window_size, 9) — full multi-channel windows.
    channel_idx : int
        Index of the channel to score.
    ... (see build_output_rows for other params)

    Returns
    -------
    (rows, errors) : Tuple[List[Dict], np.ndarray]
    """
    from .models.sarima_model import SARIMAModel

    if isinstance(model, SARIMAModel):
        # SARIMA.score returns (N, 9); extract this channel
        all_errors = model.score(windows)         # (N, 9)
        errors     = all_errors[:, channel_idx]   # (N,)
    else:
        # LSTM/Transformer: pass single-channel slice (N, T, 1)
        ch_windows = windows[:, :, channel_idx : channel_idx + 1]
        errors     = model.score(ch_windows)      # (N,)

    predicted = label_windows(errors, threshold)
    rows = build_output_rows(
        windows_meta, errors, predicted,
        channel_name, subject_id, movement, model_name,
        is_synthetic_anomaly, anomaly_type, window_id_offset,
    )
    return rows, errors
