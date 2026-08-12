"""
semg_pipeline/augmenter.py

Semi-supervised training-set augmentation for the sEMG anomaly detection
pipeline.

PURPOSE
-------
To improve anomaly recall, this module exposes a configurable fraction of
*training* windows as synthetic anomalies (label=1), while the remainder stay
clean (label=0).  The resulting (windows, labels) pair is passed to
LSTMModel.fit() / TransformerModel.fit() so the model can learn a joint
reconstruction + classification objective instead of relying purely on
reconstruction error at test time.

DESIGN DECISIONS
----------------
* Augmentation operates on **single-channel** slices (N, T, 1), which is how
  LSTM/Transformer models receive data (one model instance per channel).
* A fraction of windows are **replaced** (not appended) by their anomalous
  counterparts.  This keeps the training set size N constant and avoids
  class-imbalance inflation.
* Multiple anomaly injection functions may be supplied; the one applied to each
  selected window is chosen uniformly at random.
* The anomaly severity used here is independent of the test-time severity sweep.
  A single moderate severity is a sensible default for training.

USAGE
-----
    from semg_pipeline.augmenter import augment_with_anomalies

    # windows : (N, T, 1) single-channel training slice
    aug_windows, labels = augment_with_anomalies(
        windows,
        fraction=0.20,
        anomaly_types=["amplitude_scale", "time_warp", "time_shift"],
        severity="moderate",
    )
    # aug_windows : (N, T, 1)  — 20 % replaced by anomalous versions
    # labels      : (N,)  int  — 0 = normal, 1 = anomalous
"""

import logging
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Ensure repo root is on sys.path for the shared synthetic_anomalies module
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.synthetic_anomalies import (
    inject_amplitude_scale,
    inject_combined,
    inject_time_shift,
    inject_time_warp,
    DEFAULT_SEVERITIES,
)

# Map string name → injection function (must match ANOMALY_TYPES in synthetic_anomalies.py)
_INJECT_FN_MAP = {
    "amplitude_scale": inject_amplitude_scale,
    "time_warp":       inject_time_warp,
    "time_shift":      inject_time_shift,
    "combined":        inject_combined,
}

_DEFAULT_ANOMALY_TYPES: List[str] = [
    "amplitude_scale",
    "time_warp",
    "time_shift",
    "combined",
]


def augment_with_anomalies(
    windows: np.ndarray,
    fraction: float = 0.20,
    anomaly_types: Optional[List[str]] = None,
    severity: str = "moderate",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Replace a fraction of training windows with synthetic anomalies and
    return binary labels.

    Parameters
    ----------
    windows : np.ndarray
        Shape (N, T, 1) — single-channel training windows (already normalised).
        The array is **not** modified in place; a copy is returned.
    fraction : float, default 0.20
        Fraction of windows to replace with synthetic anomalies.
        Must be in (0, 1).  e.g. 0.20 → 20 % of N windows become anomalous.
    anomaly_types : list of str, optional
        Subset of {'amplitude_scale', 'time_warp', 'time_shift', 'combined'}.
        Defaults to all four.  For each selected window the injection function
        is chosen uniformly at random from this list.
    severity : str, default 'moderate'
        Key into DEFAULT_SEVERITIES ('mild'=0.15, 'moderate'=0.35, 'severe'=0.60).
        All injections during training use the same severity level.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.  Defaults to
        np.random.default_rng(42).

    Returns
    -------
    aug_windows : np.ndarray
        Shape (N, T, 1) — copy of `windows` with anomalous replacements.
    labels : np.ndarray
        Shape (N,) dtype int — 0 = normal window, 1 = synthetic anomaly.

    Raises
    ------
    ValueError
        If `fraction` is not in (0, 1), `severity` is unknown, or
        an unrecognised anomaly type is requested.
    """
    if not (0.0 < fraction < 1.0):
        raise ValueError(
            f"augmentation fraction must be in (0, 1), got {fraction}"
        )

    if severity not in DEFAULT_SEVERITIES:
        raise ValueError(
            f"severity must be one of {list(DEFAULT_SEVERITIES)}, got '{severity}'"
        )

    if anomaly_types is None:
        anomaly_types = _DEFAULT_ANOMALY_TYPES

    unknown = set(anomaly_types) - set(_INJECT_FN_MAP)
    if unknown:
        raise ValueError(
            f"Unknown anomaly_types: {unknown}. "
            f"Valid options: {list(_INJECT_FN_MAP)}"
        )

    if rng is None:
        rng = np.random.default_rng(42)

    n_windows = len(windows)
    if n_windows == 0:
        return windows.copy(), np.zeros(0, dtype=int)

    n_anomalous = max(1, int(round(fraction * n_windows)))
    # Clamp so we never try to replace more windows than we have
    n_anomalous = min(n_anomalous, n_windows)

    # Choose which window indices become anomalous (without replacement)
    anomalous_indices = rng.choice(n_windows, size=n_anomalous, replace=False)

    sev_value = DEFAULT_SEVERITIES[severity]
    inject_fns = [_INJECT_FN_MAP[atype] for atype in anomaly_types]

    aug_windows = windows.copy()
    labels = np.zeros(n_windows, dtype=int)

    for w_idx in anomalous_indices:
        # Randomly choose one injection function for this window
        inject_fn = inject_fns[int(rng.integers(len(inject_fns)))]
        # Extract the 1D signal (squeeze the channel dim)
        sig_1d = aug_windows[w_idx, :, 0]

        # inject_combined expects no 'severity' kwarg (uses its own defaults)
        if inject_fn is inject_combined:
            anom_sig, _ = inject_fn(sig_1d)
        else:
            anom_sig, _ = inject_fn(sig_1d, severity=sev_value)

        aug_windows[w_idx, :, 0] = anom_sig.astype(windows.dtype)
        labels[w_idx] = 1

    n_clean = n_windows - n_anomalous
    logger.debug(
        f"[Augmenter] {n_windows} windows total: "
        f"{n_clean} clean ({n_clean/n_windows:.1%}), "
        f"{n_anomalous} anomalous ({n_anomalous/n_windows:.1%})  "
        f"severity={severity} ({sev_value})"
    )

    return aug_windows, labels
