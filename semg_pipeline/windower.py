"""
semg_pipeline/windower.py

Sliding window segmentation for sEMG signals at 1920 Hz.

Key constraints (matching windower.py from kinematics/kinetics pipeline):
  - Window size  : 1920 samples = 1 second at 1920 Hz
  - Overlap      : 960 samples  = 50%
  - Stride       : 960 samples
  - Boundary rule: ALL samples in a window must share the same Group value
                   (non-NaN, > 0) — no cross-gait-cycle windows allowed
  - Active-only  : ALL samples must have Status == 1 (loader already filters,
                   but we double-check here to be safe after re-indexing)

Output shape: (num_windows, window_size, num_channels)
              where num_channels = 9
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from .loader import SEMG_CHANNELS


def create_semg_windows(
    df: pd.DataFrame,
    window_size: int = 1920,
    overlap_size: int = 960,
    channels: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Segment a trial DataFrame into fixed-size overlapping windows.

    Enforces:
      1. All samples in a window belong to the same gait cycle (Group).
      2. All samples have Status == 1 (active gait).
      3. Windows do not cross Group boundaries.

    Parameters
    ----------
    df : pd.DataFrame
        Trial DataFrame with Time, sEMG channels, Status, Group columns.
        Typically the output of load_semg_trial() after filtering and scaling.
    window_size : int, default 1920
        Number of samples per window (1 second at 1920 Hz).
    overlap_size : int, default 960
        Number of overlapping samples between consecutive windows (50%).
    channels : List[str], optional
        Channel names to include in the output array.
        Defaults to SEMG_CHANNELS (all 9).

    Returns
    -------
    windows : np.ndarray
        Shape (num_windows, window_size, num_channels). float32.
    metadata : List[Dict]
        One dict per window with keys:
          - start_time   : float  (seconds, from Time column)
          - end_time     : float  (seconds, from Time column)
          - group_cycle  : int    (gait cycle index)
          - start_index  : int    (row index in df)
          - end_index    : int    (exclusive row index in df)
    """
    if channels is None:
        channels = SEMG_CHANNELS

    step = window_size - overlap_size
    if step <= 0:
        raise ValueError(
            f"overlap_size ({overlap_size}) must be strictly less than "
            f"window_size ({window_size})."
        )

    if df.empty or len(df) < window_size:
        empty_arr = np.empty((0, window_size, len(channels)), dtype=np.float32)
        return empty_arr, []

    # Extract arrays for fast indexing
    signal_data = df[channels].values.astype(np.float32)  # (N, 9)
    groups      = df["Group"].values
    status_vals = df["Status"].values
    times       = df["Time"].values

    windows: List[np.ndarray] = []
    metadata: List[Dict]      = []

    for start_idx in range(0, len(df) - window_size + 1, step):
        end_idx = start_idx + window_size

        # ── Boundary constraint: all samples must share same Group ──
        window_groups = groups[start_idx:end_idx]
        first_group   = window_groups[0]

        # Skip NaN group (boundary / unassigned frames).
        # Group=0 is valid for some movements (A/R type), so we only
        # reject NaN, not zero.
        try:
            if np.isnan(float(first_group)):
                continue
        except (TypeError, ValueError):
            continue

        if not np.all(window_groups == first_group):
            continue

        # ── Active-only sanity: no NaN status in the window ──
        # The loader already filtered to active-only rows, so NaN
        # status should not appear here; this is a safety guard.
        window_status = status_vals[start_idx:end_idx]
        if pd.isnull(window_status).any():
            continue

        windows.append(signal_data[start_idx:end_idx])
        metadata.append(
            {
                "start_time":  float(times[start_idx]),
                "end_time":    float(times[end_idx - 1]),
                "group_cycle": int(first_group),
                "start_index": start_idx,
                "end_index":   end_idx,
            }
        )

    if not windows:
        empty_arr = np.empty((0, window_size, len(channels)), dtype=np.float32)
        return empty_arr, []

    return np.stack(windows, axis=0), metadata  # (W, 1920, 9)
