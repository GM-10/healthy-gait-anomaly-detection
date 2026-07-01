"""
semg_pipeline/normalizer.py

Per-subject, per-channel Min-Max scaling to [-1, 1].

Design principles (matching kinematics/kinetics conditioner.py):
  - Fit scalers on TRAINING subjects only (Sub01–Sub30)
  - Apply the same fitted params to val and test subjects — NO leakage
  - Scaler params are stored as a plain dict: {channel_name: (min_val, max_val)}
  - Parameters can be serialized to / loaded from JSON for reproducibility

The scaler is intentionally simple (no sklearn dependency) to match
the teammate's conditioner.py pattern exactly.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from .loader import SEMG_CHANNELS

# Type alias for scaler parameters
ScalerParams = Dict[str, Tuple[float, float]]   # {channel: (min, max)}


# ─────────────────────────────────────────────────────────────────────────────
# Fitting
# ─────────────────────────────────────────────────────────────────────────────

def fit_scaler(
    dfs: List[pd.DataFrame],
    channels: Optional[List[str]] = None,
) -> ScalerParams:
    """
    Compute per-channel Min-Max parameters from a list of training trial DataFrames.

    Parameters
    ----------
    dfs : List[pd.DataFrame]
        Trial DataFrames containing sEMG channel columns.
        Should include ALL training trials concatenated (no leakage).
    channels : List[str], optional
        Channel names to fit. Defaults to all SEMG_CHANNELS.

    Returns
    -------
    ScalerParams
        Dict mapping channel_name → (global_min, global_max) across all trials.
    """
    if channels is None:
        channels = SEMG_CHANNELS

    if not dfs:
        raise ValueError("fit_scaler: received empty list of DataFrames.")

    # Pool all training data to get global min/max per channel
    combined = pd.concat(dfs, ignore_index=True)

    params: ScalerParams = {}
    for ch in channels:
        if ch not in combined.columns:
            continue
        col = combined[ch].dropna()
        if col.empty:
            params[ch] = (0.0, 1.0)   # fallback: identity scaling
        else:
            params[ch] = (float(col.min()), float(col.max()))

    return params


# ─────────────────────────────────────────────────────────────────────────────
# Applying
# ─────────────────────────────────────────────────────────────────────────────

def apply_scaler(
    df: pd.DataFrame,
    scaling_params: ScalerParams,
    feature_range: Tuple[float, float] = (-1.0, 1.0),
    channels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Apply pre-fitted Min-Max scaling to a trial DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Trial DataFrame with sEMG channels to scale.
    scaling_params : ScalerParams
        Fitted parameters dict from fit_scaler().
    feature_range : Tuple[float, float], default (-1.0, 1.0)
        Target range after scaling.
    channels : List[str], optional
        Channels to scale. Defaults to all SEMG_CHANNELS.

    Returns
    -------
    pd.DataFrame
        Copy of df with sEMG channel columns scaled to feature_range.
    """
    if channels is None:
        channels = SEMG_CHANNELS

    lower, upper = feature_range
    scaled_df = df.copy()

    for ch in channels:
        if ch not in scaled_df.columns or ch not in scaling_params:
            continue

        min_val, max_val = scaling_params[ch]
        diff = max_val - min_val

        if np.isclose(diff, 0.0):
            # Constant channel — map to midpoint of target range
            scaled_df[ch] = 0.0
        else:
            scaled_df[ch] = lower + (scaled_df[ch] - min_val) / diff * (upper - lower)

    return scaled_df


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_scaler(params: ScalerParams, path: str) -> None:
    """
    Serialize scaler parameters to a JSON file.

    Parameters
    ----------
    params : ScalerParams
        Fitted parameters dict.
    path : str
        Destination file path (e.g. 'outputs/sEMG/scaler_params.json').
    """
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Convert tuples to lists for JSON serialization
    serializable = {ch: list(v) for ch, v in params.items()}
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def load_scaler(path: str) -> ScalerParams:
    """
    Load scaler parameters from a JSON file.

    Parameters
    ----------
    path : str
        Path to the JSON file saved by save_scaler().

    Returns
    -------
    ScalerParams
    """
    with open(path, "r") as f:
        raw = json.load(f)
    # Convert lists back to tuples
    return {ch: tuple(v) for ch, v in raw.items()}
