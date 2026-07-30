"""
evaluation_framework/thresholds.py

Pluggable thresholding strategies for anomaly detection.

Implements three strategies, all computed exclusively from training-set
reconstruction errors (healthy windows only):

    MeanStdThreshold   — threshold = μ + n·σ  (default n=3.0, existing method)
    PercentileThreshold — threshold = np.percentile(train_errors, p)

Factory function:
    get_threshold(method_name) → ThresholdBase instance

Usage:
    from evaluation_framework.thresholds import get_threshold

    thresh = get_threshold("percentile95")
    result = thresh.fit(train_errors)
    labels = thresh.predict(test_errors)
"""

from __future__ import annotations

import json
import os
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

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
