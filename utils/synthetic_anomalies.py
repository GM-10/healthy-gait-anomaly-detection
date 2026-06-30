"""
synthetic_anomalies.py

Shared synthetic anomaly injection module for the SIAT-LLMD gait anomaly
detection project (IEEE GSCon 2027 submission).

PURPOSE
-------
Since SIAT-LLMD contains only healthy subjects, we have no real pathological
gait data to validate our anomaly detection system against. This module
injects controlled, mathematically defined distortions into healthy signals
to simulate plausible gait pathologies, giving us ground-truth labels to
evaluate detection performance (Recall, F1, etc.).

BOTH teammates (sEMG pipeline + kinematics/kinetics pipeline) must import
THIS SAME FILE rather than writing their own versions, so that:
  1. The mathematical definition of each anomaly type is identical across
     modalities.
  2. Severity parameters mean the same thing everywhere.
  3. Results are directly comparable and mergeable for late fusion.

THREE ANOMALY TYPES
--------------------
1. Amplitude scaling   -> simulates reduced range of motion (e.g. muscle
                           weakness, joint stiffness post-injury)
2. Time warping         -> simulates asymmetric / uneven gait timing
                           (e.g. one leg moving slower than the other,
                           common in stroke patients)
3. Time shifting         -> simulates delayed muscle activation
                           (e.g. neuromuscular signal delay)

Each function takes a 1D numpy array (a single channel's signal, already
windowed) and a severity parameter, and returns:
    (modified_signal, anomaly_label)
where anomaly_label is always 1 (since by construction the output IS an
injected anomaly). Use anomaly_type strings (defined in ANOMALY_TYPES below)
when writing rows to your output CSV.

USAGE EXAMPLE
-------------
    from synthetic_anomalies import inject_amplitude_scale, ANOMALY_TYPES

    window = signal_array[start:end]          # your clean windowed signal
    anomalous_window, label = inject_amplitude_scale(window, severity=0.5)
    # label == 1
    # anomaly_type == "amplitude_scale"
"""

import numpy as np
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Shared constants — use these exact strings in your output CSV's
# `anomaly_type` column so both pipelines stay consistent.
# ---------------------------------------------------------------------------
ANOMALY_TYPES = {
    "none": "none",
    "amplitude_scale": "amplitude_scale",
    "time_warp": "time_warp",
    "time_shift": "time_shift",
}

# Default severity levels. Feel free to sweep across these for experiments,
# but report results using these as your standard set so SARIMA/LSTM/
# Transformer and sEMG/kinematics/kinetics comparisons stay apples-to-apples.
DEFAULT_SEVERITIES = {
    "mild": 0.15,
    "moderate": 0.35,
    "severe": 0.60,
}


def inject_amplitude_scale(signal: np.ndarray, severity: float = 0.35) -> tuple:
    """
    Simulates reduced range of motion / muscle weakness by scaling the
    signal's amplitude down around its own mean.

    Parameters
    ----------
    signal : np.ndarray
        1D array, a single windowed channel of clean (healthy) signal.
    severity : float
        Fraction of amplitude to REMOVE, in range (0, 1).
        e.g. severity=0.35 means the signal retains 65% of its original
        deviation from the mean (i.e. range of motion is reduced by 35%).

    Returns
    -------
    (modified_signal, label) : (np.ndarray, int)
        label is always 1 (anomalous by construction).
    """
    if not (0 < severity < 1):
        raise ValueError("severity must be strictly between 0 and 1")

    signal = np.asarray(signal, dtype=float)
    mean_val = np.mean(signal)
    scale_factor = 1.0 - severity

    modified_signal = mean_val + (signal - mean_val) * scale_factor
    label = 1

    return modified_signal, label


def inject_time_warp(signal: np.ndarray, severity: float = 0.35) -> tuple:
    """
    Simulates asymmetric / uneven gait timing by non-uniformly stretching
    and compressing the time axis. The first half of the window is
    stretched (slowed down) and the second half compressed (sped up),
    or vice versa, then the result is resampled back to the original
    window length so output shape == input shape.

    Parameters
    ----------
    signal : np.ndarray
        1D array, a single windowed channel of clean (healthy) signal.
    severity : float
        Fraction controlling how unevenly time is warped, in range (0, 1).
        e.g. severity=0.35 means the first half effectively takes ~35%
        longer (relative time-axis distortion) than the second half.

    Returns
    -------
    (modified_signal, label) : (np.ndarray, int)
        label is always 1 (anomalous by construction).
    """
    if not (0 < severity < 1):
        raise ValueError("severity must be strictly between 0 and 1")

    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    if n < 4:
        raise ValueError("signal too short to time-warp meaningfully (need >= 4 samples)")

    original_time = np.linspace(0, 1, n)

    # Build a monotonic warped time axis: slow down the first half,
    # speed up the second half, by `severity`. This keeps endpoints
    # fixed at 0 and 1 so the window boundaries are preserved.
    midpoint = 0.5
    warped_time = np.where(
        original_time <= midpoint,
        original_time * (1 + severity),
        midpoint * (1 + severity) + (original_time - midpoint) * (1 - severity),
    )
    # Renormalize so warped_time still spans exactly [0, 1]
    warped_time = warped_time / warped_time[-1]

    # Interpolate the original signal onto the warped time axis, then
    # resample back onto the original (uniform) time axis so the output
    # array length matches the input — this is what actually encodes the
    # "uneven timing" distortion.
    interpolator = interp1d(original_time, signal, kind="linear", fill_value="extrapolate")
    warped_signal = interpolator(warped_time)

    label = 1
    return warped_signal, label


def inject_time_shift(signal: np.ndarray, severity: float = 0.35) -> tuple:
    """
    Simulates delayed muscle/neural activation by shifting the entire
    signal forward in time by a number of samples proportional to
    `severity`. Positions vacated at the start are filled by repeating
    the first value (edge-hold), so output shape == input shape and no
    information from outside the window is introduced.

    Parameters
    ----------
    signal : np.ndarray
        1D array, a single windowed channel of clean (healthy) signal.
    severity : float
        Fraction of the window length to shift by, in range (0, 1).
        e.g. severity=0.35 on a 100-sample window shifts the signal
        forward by 35 samples.

    Returns
    -------
    (modified_signal, label) : (np.ndarray, int)
        label is always 1 (anomalous by construction).
    """
    if not (0 < severity < 1):
        raise ValueError("severity must be strictly between 0 and 1")

    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    shift_samples = int(round(severity * n))
    shift_samples = max(1, min(shift_samples, n - 1))  # keep it sane

    shifted_signal = np.empty_like(signal)
    shifted_signal[:shift_samples] = signal[0]              # edge-hold padding
    shifted_signal[shift_samples:] = signal[: n - shift_samples]

    label = 1
    return shifted_signal, label


def inject_combined(signal: np.ndarray, severities: dict = None) -> tuple:
    """
    Optional: applies all three anomaly types in sequence (scale -> warp ->
    shift) to simulate a more complex, compound pathology. Useful as a
    stress-test condition once the three individual anomaly types are
    validated separately.

    Parameters
    ----------
    signal : np.ndarray
        1D array, a single windowed channel of clean (healthy) signal.
    severities : dict, optional
        Keys: "amplitude_scale", "time_warp", "time_shift".
        Defaults to DEFAULT_SEVERITIES["moderate"] for each if not given.

    Returns
    -------
    (modified_signal, label) : (np.ndarray, int)
    """
    if severities is None:
        severities = {
            "amplitude_scale": DEFAULT_SEVERITIES["moderate"],
            "time_warp": DEFAULT_SEVERITIES["moderate"],
            "time_shift": DEFAULT_SEVERITIES["moderate"],
        }

    sig, _ = inject_amplitude_scale(signal, severities["amplitude_scale"])
    sig, _ = inject_time_warp(sig, severities["time_warp"])
    sig, label = inject_time_shift(sig, severities["time_shift"])

    return sig, label


# ---------------------------------------------------------------------------
# Quick self-test when run directly: python synthetic_anomalies.py
# Generates a synthetic clean sine wave and shows all three anomaly types
# applied to it, so both teammates can sanity-check their environment
# before plugging this into the real pipelines.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    n_samples = 100
    t = np.linspace(0, 2 * np.pi, n_samples)
    clean_signal = np.sin(t) * 30 + 50  # toy "knee angle"-like signal

    scaled, _ = inject_amplitude_scale(clean_signal, severity=0.35)
    warped, _ = inject_time_warp(clean_signal, severity=0.35)
    shifted, _ = inject_time_shift(clean_signal, severity=0.35)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes[0, 0].plot(clean_signal)
    axes[0, 0].set_title("Original (healthy)")

    axes[0, 1].plot(clean_signal, label="original", alpha=0.4)
    axes[0, 1].plot(scaled, label="amplitude_scale")
    axes[0, 1].set_title("Amplitude Scaled (severity=0.35)")
    axes[0, 1].legend()

    axes[1, 0].plot(clean_signal, label="original", alpha=0.4)
    axes[1, 0].plot(warped, label="time_warp")
    axes[1, 0].set_title("Time Warped (severity=0.35)")
    axes[1, 0].legend()

    axes[1, 1].plot(clean_signal, label="original", alpha=0.4)
    axes[1, 1].plot(shifted, label="time_shift")
    axes[1, 1].set_title("Time Shifted (severity=0.35)")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig("synthetic_anomalies_preview.png", dpi=120)
    print("Self-test complete. Preview saved to synthetic_anomalies_preview.png")
