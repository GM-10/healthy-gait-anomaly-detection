"""
semg_pipeline/anomaly_scorer.py

Threshold computation and output CSV row builder for the sEMG pipeline.

Threshold rule (applied after training, using train set reconstruction errors):
    threshold = mean(train_errors) + 3 × std(train_errors)

Output CSV schema (must match teammate's kinematics/kinetics pipeline exactly):
    subject_id, modality, channel_name, movement, window_id,
    window_start_time, window_end_time, reconstruction_error,
    is_synthetic_anomaly, anomaly_type, predicted_label, model_name
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


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
) -> List[Dict]:
    """
    Build a list of output CSV row dicts for one channel, one model run.

    Each dict has exactly the 12 output CSV columns, matching the
    kinematics/kinetics teammate's schema.

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
