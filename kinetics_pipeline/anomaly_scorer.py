"""
kinetics_pipeline/anomaly_scorer.py

Threshold computation and output CSV row builder for the Kinematics + Kinetics pipeline.

Threshold rule:
    threshold = mean(train_errors) + 3 × std(train_errors)

Output CSV schema (matches sEMG pipeline exactly):
    subject_id, modality, channel_name, movement, window_id,
    window_start_time, window_end_time, reconstruction_error,
    is_synthetic_anomaly, anomaly_type, predicted_label, model_name
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def compute_threshold(train_errors: np.ndarray, n_sigma: float = 3.0) -> float:
    """
    Compute the anomaly detection threshold from training reconstruction errors.
    """
    if len(train_errors) == 0:
        raise ValueError("train_errors is empty — cannot compute threshold.")
    return float(np.mean(train_errors) + n_sigma * np.std(train_errors))


def label_windows(errors: np.ndarray, threshold: float) -> np.ndarray:
    """
    Binary-label each window based on its reconstruction error.
    """
    return (errors > threshold).astype(int)


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
    """
    rows = []
    for i, (meta, err, pred) in enumerate(zip(windows_meta, errors, predicted_labels)):
        rows.append(
            {
                "subject_id":           subject_id,
                "modality":             "Kinematics+Kinetics",
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
    """
    from kinetics_pipeline.models.sarima_model import SARIMAModel

    if isinstance(model, SARIMAModel):
        # SARIMA.score returns (N, 16); extract this channel
        all_errors = model.score(windows)         # (N, 16)
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
