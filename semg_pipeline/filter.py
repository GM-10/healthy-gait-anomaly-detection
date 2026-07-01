"""
semg_pipeline/filter.py

sEMG signal conditioning pipeline applied at the native 1920 Hz rate.
DO NOT downsample before or after this chain — sEMG must remain at 1920 Hz
to preserve frequency content up to 400 Hz (Nyquist at 1920 Hz = 960 Hz).

Filter chain (applied per channel in this exact order):
    1. Notch filter        — 50 Hz powerline noise removal
    2. Bandpass filter     — 15–400 Hz, Butterworth order 4, zero-phase
    3. Full-wave rectify   — abs(signal), converts bipolar EMG to envelope
    4. Linear envelope     — low-pass 6 Hz, Butterworth order 4, zero-phase

All filters use scipy.signal.filtfilt for zero-phase (non-causal) operation,
which doubles the effective order and introduces no phase distortion.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
from typing import Optional

from .loader import SEMG_CHANNELS

# ─────────────────────────────────────────────────────────────────────────────
# Filter design helpers
# ─────────────────────────────────────────────────────────────────────────────

# Minimum signal length that filtfilt can handle safely.
# filtfilt needs at least 3 × max(len(a), len(b)) samples.
_MIN_FILTFILT_SAMPLES = 27


def _design_notch(freq: float = 50.0, quality_factor: float = 30.0, fs: float = 1920.0):
    """Design a 2nd-order IIR notch filter (used with filtfilt → effective 4th order)."""
    b, a = iirnotch(freq, quality_factor, fs)
    return b, a


def _design_bandpass(low: float = 15.0, high: float = 400.0, order: int = 4, fs: float = 1920.0):
    """Design a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low_norm  = low  / nyq
    high_norm = high / nyq
    # Clamp to (0, 1) exclusive for numerical safety
    low_norm  = np.clip(low_norm,  1e-6, 1 - 1e-6)
    high_norm = np.clip(high_norm, 1e-6, 1 - 1e-6)
    b, a = butter(order, [low_norm, high_norm], btype="band")
    return b, a


def _design_lowpass(cutoff: float = 6.0, order: int = 4, fs: float = 1920.0):
    """Design a Butterworth low-pass filter (used for linear envelope)."""
    nyq = 0.5 * fs
    norm = np.clip(cutoff / nyq, 1e-6, 1 - 1e-6)
    b, a = butter(order, norm, btype="low")
    return b, a


# ─────────────────────────────────────────────────────────────────────────────
# Per-channel filter application
# ─────────────────────────────────────────────────────────────────────────────

def _apply_filter_chain_1d(
    signal: np.ndarray,
    b_notch: np.ndarray,
    a_notch: np.ndarray,
    b_bp: np.ndarray,
    a_bp: np.ndarray,
    b_env: np.ndarray,
    a_env: np.ndarray,
) -> np.ndarray:
    """
    Apply the four-stage filter chain to a single 1D signal array.

    Stages
    ------
    1. Notch  (50 Hz)
    2. Bandpass (15–400 Hz)
    3. Full-wave rectification
    4. Low-pass envelope (6 Hz)

    Short signals (< _MIN_FILTFILT_SAMPLES) are returned unchanged
    with only rectification applied.
    """
    signal = np.asarray(signal, dtype=float)
    n = len(signal)

    if n < _MIN_FILTFILT_SAMPLES:
        # Can't run filtfilt — at minimum rectify and return
        return np.abs(signal)

    # Step 1: Notch
    sig = filtfilt(b_notch, a_notch, signal)

    # Step 2: Bandpass
    sig = filtfilt(b_bp, a_bp, sig)

    # Step 3: Full-wave rectification
    sig = np.abs(sig)

    # Step 4: Linear envelope (low-pass)
    sig = filtfilt(b_env, a_env, sig)

    return sig


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def apply_semg_filter_chain(
    df: pd.DataFrame,
    fs: float = 1920.0,
    notch_freq: float = 50.0,
    notch_q: float = 30.0,
    bp_low: float = 15.0,
    bp_high: float = 400.0,
    bp_order: int = 4,
    env_cutoff: float = 6.0,
    env_order: int = 4,
    channels: Optional[list] = None,
) -> pd.DataFrame:
    """
    Apply the full sEMG filter chain to a trial DataFrame.

    The chain is applied independently to each sEMG channel column.
    Non-sEMG columns (Time, Status, Group) are preserved unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        Trial DataFrame containing sEMG channel columns (canonical short names).
    fs : float, default 1920.0
        Native sampling rate of the sEMG signals in Hz.
    notch_freq : float, default 50.0
        Powerline frequency to notch-filter in Hz.
    notch_q : float, default 30.0
        Quality factor of the notch filter. Higher = narrower notch.
    bp_low : float, default 15.0
        Lower cutoff frequency for bandpass filter in Hz.
    bp_high : float, default 400.0
        Upper cutoff frequency for bandpass filter in Hz.
    bp_order : int, default 4
        Order of the Butterworth bandpass filter.
    env_cutoff : float, default 6.0
        Cutoff frequency of the linear envelope low-pass filter in Hz.
    env_order : int, default 4
        Order of the envelope Butterworth low-pass filter.
    channels : list, optional
        Subset of channel names to filter. Defaults to all SEMG_CHANNELS.

    Returns
    -------
    pd.DataFrame
        Copy of input DataFrame with sEMG columns replaced by filtered signals.
    """
    if channels is None:
        channels = SEMG_CHANNELS

    # Design filters once — reuse across all channels
    b_notch, a_notch = _design_notch(notch_freq, notch_q, fs)
    b_bp,    a_bp    = _design_bandpass(bp_low, bp_high, bp_order, fs)
    b_env,   a_env   = _design_lowpass(env_cutoff, env_order, fs)

    filtered_df = df.copy()

    for ch in channels:
        if ch not in filtered_df.columns:
            continue
        filtered_df[ch] = _apply_filter_chain_1d(
            filtered_df[ch].values,
            b_notch, a_notch,
            b_bp,    a_bp,
            b_env,   a_env,
        )

    return filtered_df
