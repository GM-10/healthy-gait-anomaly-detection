"""
evaluation_framework/thresholds.py

Pluggable thresholding strategies for anomaly detection.

Implements three strategies, all computed exclusively from training-set
reconstruction errors (healthy windows only):

    MeanStdThreshold   — threshold = μ + n·σ  (default n=3.0, existing method)
    PercentileThreshold — threshold = np.percentile(train_errors, p)

Factory function:
    get_threshold(method_name) → ThresholdBase instance

Comparison utility:
    compare_thresholds(threshold_results, test_scores) → pd.DataFrame
        Side-by-side comparison of threshold values, relative differences,
        and per-window label agreement rates across all three methods.

Usage:
    from evaluation_framework.thresholds import get_threshold, compare_thresholds

    thresh = get_threshold("percentile95")
    result = thresh.fit(train_errors)
    labels = thresh.predict(test_errors)

    comparison_df = compare_thresholds(
        threshold_results={\"mean_std\": r1, \"percentile95\": r2, \"percentile99\": r3},
        test_scores=test_errors,
    )
"""

from __future__ import annotations

import json
import os
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass — stores everything needed for reproducibility
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ThresholdResult:
    """
    Fully-documented threshold result for one (model, channel, method).

    Stores the threshold value, method metadata, and training error
    distribution statistics for full reproducibility.
    """
    method:      str           # e.g. "mean_std", "percentile95"
    value:       float         # the scalar threshold
    params:      Dict[str, Any] = field(default_factory=dict)  # method hyperparams

    # Training error distribution statistics
    train_mean:   float = 0.0
    train_std:    float = 0.0
    train_min:    float = 0.0
    train_max:    float = 0.0
    train_p50:    float = 0.0
    train_p75:    float = 0.0
    train_p90:    float = 0.0
    train_p95:    float = 0.0
    train_p99:    float = 0.0
    train_n:      int   = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "ThresholdResult":
        with open(path, "r") as f:
            d = json.load(f)
        return cls(**d)


def _compute_distribution_stats(errors: np.ndarray) -> Dict[str, float]:
    """Compute summary statistics of the training error distribution."""
    errors = np.asarray(errors, dtype=float)
    return {
        "train_mean": float(np.mean(errors)),
        "train_std":  float(np.std(errors)),
        "train_min":  float(np.min(errors)),
        "train_max":  float(np.max(errors)),
        "train_p50":  float(np.percentile(errors, 50)),
        "train_p75":  float(np.percentile(errors, 75)),
        "train_p90":  float(np.percentile(errors, 90)),
        "train_p95":  float(np.percentile(errors, 95)),
        "train_p99":  float(np.percentile(errors, 99)),
        "train_n":    int(len(errors)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class ThresholdBase(ABC):
    """Abstract base class for all thresholding strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier string, e.g. 'mean_std'."""

    @property
    @abstractmethod
    def params(self) -> Dict[str, Any]:
        """Hyperparameters for this strategy."""

    @abstractmethod
    def _compute(self, errors: np.ndarray) -> float:
        """Compute the scalar threshold from training errors."""

    def fit(self, train_errors: np.ndarray) -> ThresholdResult:
        """
        Fit the threshold to training reconstruction errors.

        Parameters
        ----------
        train_errors : np.ndarray
            1D array of reconstruction MSE values from **healthy training**
            windows only.

        Returns
        -------
        ThresholdResult
            Fully-documented threshold result.
        """
        errors = np.asarray(train_errors, dtype=float).ravel()
        if len(errors) == 0:
            raise ValueError(f"[{self.name}] train_errors is empty.")
        if len(errors) < 10:
            logger.warning(
                f"[{self.name}] Only {len(errors)} training errors — "
                "threshold may be unreliable."
            )

        value = self._compute(errors)
        stats = _compute_distribution_stats(errors)

        result = ThresholdResult(
            method=self.name,
            value=value,
            params=self.params,
            **stats,
        )
        logger.info(
            f"  [{self.name}] threshold={value:.6f}  "
            f"(μ={stats['train_mean']:.6f}, σ={stats['train_std']:.6f}, "
            f"p95={stats['train_p95']:.6f}, p99={stats['train_p99']:.6f}, "
            f"n={stats['train_n']})"
        )
        return result

    def predict(self, errors: np.ndarray, threshold: float) -> np.ndarray:
        """
        Binary-label errors using the given threshold.

        Parameters
        ----------
        errors : np.ndarray
            1D array of reconstruction MSE values.
        threshold : float
            Scalar decision boundary.

        Returns
        -------
        np.ndarray
            Integer array of 0 (normal) or 1 (anomaly).
        """
        return (np.asarray(errors, dtype=float) > threshold).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# Concrete strategies
# ─────────────────────────────────────────────────────────────────────────────

class MeanStdThreshold(ThresholdBase):
    """
    Mean + n·sigma threshold (existing method, default n=3.0).

    threshold = μ(train_errors) + n × σ(train_errors)
    """

    def __init__(self, n_sigma: float = 3.0):
        """
        Parameters
        ----------
        n_sigma : float
            Number of standard deviations above the mean. Default 3.0.
        """
        self._n_sigma = n_sigma

    @property
    def name(self) -> str:
        return "mean_std"

    @property
    def params(self) -> Dict[str, Any]:
        return {"n_sigma": self._n_sigma}

    def _compute(self, errors: np.ndarray) -> float:
        return float(np.mean(errors) + self._n_sigma * np.std(errors))


class PercentileThreshold(ThresholdBase):
    """
    Percentile-based threshold.

    threshold = np.percentile(train_errors, p)

    Common values: p=95 (PercentileThreshold(95)), p=99 (PercentileThreshold(99))
    """

    def __init__(self, percentile: float):
        """
        Parameters
        ----------
        percentile : float
            Percentile value in [0, 100]. Typical values: 95, 99.
        """
        if not (0 < percentile <= 100):
            raise ValueError(f"percentile must be in (0, 100], got {percentile}")
        self._percentile = float(percentile)

    @property
    def name(self) -> str:
        return f"percentile{int(self._percentile)}"

    @property
    def params(self) -> Dict[str, Any]:
        return {"percentile": self._percentile}

    def _compute(self, errors: np.ndarray) -> float:
        return float(np.percentile(errors, self._percentile))


# ─────────────────────────────────────────────────────────────────────────────
# Registry / factory
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLD_METHODS: Dict[str, ThresholdBase] = {
    "mean_std":     MeanStdThreshold(n_sigma=3.0),
    "percentile95": PercentileThreshold(95),
    "percentile99": PercentileThreshold(99),
}


def get_threshold(method: str) -> ThresholdBase:
    """
    Return the ThresholdBase instance for the given method name.

    Parameters
    ----------
    method : str
        One of "mean_std", "percentile95", "percentile99".

    Returns
    -------
    ThresholdBase
    """
    if method not in THRESHOLD_METHODS:
        raise ValueError(
            f"Unknown threshold method '{method}'. "
            f"Valid: {list(THRESHOLD_METHODS.keys())}"
        )
    return THRESHOLD_METHODS[method]


def fit_all_thresholds(
    train_errors: np.ndarray,
    methods: Optional[List[str]] = None,
) -> Dict[str, ThresholdResult]:
    """
    Fit multiple threshold methods to the same training error array.

    Parameters
    ----------
    train_errors : np.ndarray
        1D training reconstruction errors from healthy windows.
    methods : List[str], optional
        Subset of THRESHOLD_METHODS to fit. Default: all three.

    Returns
    -------
    Dict[str, ThresholdResult]
        Keys: method names. Values: ThresholdResult instances.
    """
    if methods is None:
        methods = list(THRESHOLD_METHODS.keys())

    results: Dict[str, ThresholdResult] = {}
    for method in methods:
        thresh = get_threshold(method)
        results[method] = thresh.fit(train_errors)
    return results


def save_all_thresholds(
    threshold_results: Dict[str, Dict[str, Dict[str, ThresholdResult]]],
    output_dir: str,
) -> None:
    """
    Serialize all threshold results to JSON.

    Parameters
    ----------
    threshold_results : dict
        Structure: {model_key: {channel_name: {method: ThresholdResult}}}
    output_dir : str
        Root directory; saves to output_dir/thresholds/
    """
    thresh_dir = os.path.join(output_dir, "thresholds")
    os.makedirs(thresh_dir, exist_ok=True)

    all_data: Dict[str, Any] = {}
    for model_key, ch_dict in threshold_results.items():
        all_data[model_key] = {}
        for ch_name, method_dict in ch_dict.items():
            all_data[model_key][ch_name] = {
                method: result.to_dict()
                for method, result in method_dict.items()
            }

    out_path = os.path.join(thresh_dir, "threshold_results.json")
    with open(out_path, "w") as f:
        json.dump(all_data, f, indent=2)
    logger.info(f"[Thresholds] Saved all threshold results → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Threshold comparison utility
# ─────────────────────────────────────────────────────────────────────────────

def compare_thresholds(
    threshold_results_map: Dict[str, Dict[str, Dict[str, "ThresholdResult"]]],
    score_df: "pd.DataFrame",
) -> "pd.DataFrame":
    """
    Produce a per-channel comparison table across all fitted threshold methods.

    For every (model_name, channel_name) triple the table shows:
      - Threshold value for each method (mean_std, percentile95, percentile99)
      - Relative difference: (Pxx − mean_std) / mean_std × 100%
      - Label-agreement rate between every pair of methods:
          agreement = fraction of windows where both methods yield the same label
      - Number of windows that *flip* label (0→1 or 1→0) when switching methods

    Parameters
    ----------
    threshold_results_map : dict
        Structure: {model_key: {channel_name: {method: ThresholdResult}}}
        Produced by _compute_all_thresholds() in evaluate.py.
    score_df : pd.DataFrame
        Score CSV data; must contain 'model_name', 'channel_name',
        'reconstruction_error'.

    Returns
    -------
    pd.DataFrame
        One row per (model_name, channel_name, modality) with comparison columns.
    """
    records: List[Dict[str, Any]] = []
    method_pairs = [
        ("mean_std", "percentile95"),
        ("mean_std", "percentile99"),
        ("percentile95", "percentile99"),
    ]

    for model_name, ch_map in threshold_results_map.items():
        model_df = score_df[score_df["model_name"] == model_name] \
            if "model_name" in score_df.columns else score_df

        for channel_name, method_map in ch_map.items():
            ch_df = model_df[model_df["channel_name"] == channel_name] \
                if "channel_name" in model_df.columns else model_df

            errors = ch_df["reconstruction_error"].values.astype(float) \
                if not ch_df.empty else np.array([])
            modality = ch_df["modality"].iloc[0] \
                if (not ch_df.empty and "modality" in ch_df.columns) else "unknown"

            rec: Dict[str, Any] = {
                "model_name":   model_name,
                "modality":     modality,
                "channel_name": channel_name,
                "n_windows":    len(errors),
            }

            # Threshold values
            thresh_values: Dict[str, float] = {}
            for method, result in method_map.items():
                thresh_values[method] = result.value
                rec[f"thresh_{method}"] = round(result.value, 8)

            # Relative differences relative to mean_std
            baseline = thresh_values.get("mean_std", float("nan"))
            for method in ["percentile95", "percentile99"]:
                if method in thresh_values and not np.isnan(baseline) and baseline != 0:
                    rel_diff = (thresh_values[method] - baseline) / abs(baseline) * 100.0
                    rec[f"rel_diff_{method}_vs_mean_std_pct"] = round(rel_diff, 4)
                else:
                    rec[f"rel_diff_{method}_vs_mean_std_pct"] = float("nan")

            # Pairwise label-agreement rates (requires test scores)
            if len(errors) > 0:
                preds: Dict[str, np.ndarray] = {}
                for method, tv in thresh_values.items():
                    preds[method] = (errors > tv).astype(int)

                for m_a, m_b in method_pairs:
                    if m_a in preds and m_b in preds:
                        agree = float((preds[m_a] == preds[m_b]).mean())
                        n_flip_01 = int(((preds[m_a] == 0) & (preds[m_b] == 1)).sum())
                        n_flip_10 = int(((preds[m_a] == 1) & (preds[m_b] == 0)).sum())
                        rec[f"agreement_{m_a}_vs_{m_b}"] = round(agree, 6)
                        rec[f"flip_0to1_{m_a}_vs_{m_b}"] = n_flip_01
                        rec[f"flip_1to0_{m_a}_vs_{m_b}"] = n_flip_10
                    else:
                        rec[f"agreement_{m_a}_vs_{m_b}"] = float("nan")
                        rec[f"flip_0to1_{m_a}_vs_{m_b}"] = 0
                        rec[f"flip_1to0_{m_a}_vs_{m_b}"] = 0
            else:
                for m_a, m_b in method_pairs:
                    rec[f"agreement_{m_a}_vs_{m_b}"] = float("nan")
                    rec[f"flip_0to1_{m_a}_vs_{m_b}"] = 0
                    rec[f"flip_1to0_{m_a}_vs_{m_b}"] = 0

            records.append(rec)
            logger.debug(
                f"[ThresholdCompare] {model_name}/{channel_name}: "
                f"mean_std={thresh_values.get('mean_std', float('nan')):.6f}  "
                f"P95={thresh_values.get('percentile95', float('nan')):.6f}  "
                f"P99={thresh_values.get('percentile99', float('nan')):.6f}"
            )

    df = pd.DataFrame(records)
    logger.info(f"[ThresholdCompare] Comparison table: {len(df)} rows")
    return df
