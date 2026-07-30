"""
evaluation_framework/bootstrap.py

Bootstrap confidence intervals and statistical significance tests.

Implements:
    bootstrap_metric()   — 95% CI for any scalar metric via bootstrap resampling
    bootstrap_comparison() — CI and p-value for difference between two models
    mcnemar_test()        — McNemar's test for paired binary classifiers
    paired_bootstrap_test() — Paired bootstrap significance test

All tests follow the methodology described in:
    Efron & Tibshirani (1993) — "An Introduction to the Bootstrap"
    Dror et al. (2018) — "Deep Dominance — How to Properly Compare Deep Neural Models"

Usage:
    from evaluation_framework.bootstrap import bootstrap_metric, mcnemar_test

    result = bootstrap_metric(y_true, y_pred, metric_fn=f1_score, n_bootstrap=1000)
    print(f"F1 = {result.mean:.4f} [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")

    mc = mcnemar_test(y_true, y_pred_lstm, y_pred_transformer)
    print(f"McNemar p = {mc.p_value:.4f}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BootstrapResult:
    """Bootstrap CI result for a single metric on a single model."""
    metric_name:  str
    observed:     float   # metric value on the original (non-resampled) data
    mean:         float   # mean across bootstrap samples
    std:          float   # std across bootstrap samples
    ci_lower:     float   # lower CI bound (percentile method)
    ci_upper:     float   # upper CI bound (percentile method)
    ci_level:     float   # e.g. 0.95
    n_bootstrap:  int

    def __str__(self) -> str:
        return (
            f"{self.metric_name}: {self.observed:.4f}  "
            f"[{self.ci_lower:.4f}, {self.ci_upper:.4f}] "
            f"(n={self.n_bootstrap}, CI={self.ci_level:.0%})"
        )


@dataclass
class BootstrapComparison:
    """Bootstrap comparison result between two models A and B."""
    metric_name: str
    delta_observed: float    # metric_B - metric_A on original data
    delta_mean:     float    # mean(metric_B - metric_A) across bootstrap
    delta_std:      float
    ci_lower:       float
    ci_upper:       float
    p_value:        float    # two-sided p-value: P(|delta*| >= |delta|)
    ci_level:       float
    n_bootstrap:    int
    effect_size:    float    # Cohen's d: delta / pooled std

    @property
    def is_significant(self) -> bool:
        return self.p_value < (1 - self.ci_level)

    def __str__(self) -> str:
        sig = "✓ significant" if self.is_significant else "✗ not significant"
        return (
            f"{self.metric_name} Δ={self.delta_observed:+.4f}  "
            f"CI=[{self.ci_lower:+.4f}, {self.ci_upper:+.4f}]  "
            f"p={self.p_value:.4f}  d={self.effect_size:.3f}  {sig}"
        )


@dataclass
class McNemarResult:
    """McNemar's test result for paired binary classifiers."""
    statistic:          float    # chi-squared statistic (with continuity correction)
    p_value:            float
    b:                  int      # A correct, B wrong
    c:                  int      # A wrong, B correct
    n_discordant:       int      # b + c
    is_significant:     bool

    def __str__(self) -> str:
        sig = "✓ p < 0.05" if self.is_significant else "✗ p >= 0.05"
        return (
            f"McNemar: χ²={self.statistic:.4f}  p={self.p_value:.4f}  "
            f"b={self.b} c={self.c}  {sig}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers (self-contained, no sklearn dependency for basic metrics)
# ─────────────────────────────────────────────────────────────────────────────

def _recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())
    return TP / (TP + FN) if (TP + FN) > 0 else 0.0


def _precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    return TP / (TP + FP) if (TP + FP) > 0 else 0.0


def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    p = _precision(y_true, y_pred)
    r = _recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


# Built-in metric registry so callers can pass a string name
BUILTIN_METRICS: Dict[str, Callable] = {
    "recall":    _recall,
    "precision": _precision,
    "f1":        _f1,
    "accuracy":  _accuracy,
}


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap CI
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    metric_name: str = "metric",
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """
    Compute a bootstrap confidence interval for a scalar metric.

    Uses the percentile bootstrap method (Efron, 1979).

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels.
    y_pred : np.ndarray
        Predicted binary labels.
    metric_fn : callable
        fn(y_true, y_pred) → float. Can also pass a string key from
        BUILTIN_METRICS ("recall", "precision", "f1", "accuracy").
    metric_name : str
        Label for the result.
    n_bootstrap : int
        Number of bootstrap resamples. Minimum 1000 recommended.
    ci_level : float
        Confidence level (default 0.95 → 95% CI).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    BootstrapResult
    """
    if isinstance(metric_fn, str):
        if metric_fn not in BUILTIN_METRICS:
            raise ValueError(f"Unknown metric '{metric_fn}'. Valid: {list(BUILTIN_METRICS)}")
        metric_name = metric_fn
        metric_fn = BUILTIN_METRICS[metric_fn]

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n      = len(y_true)

    observed = metric_fn(y_true, y_pred)

    rng      = np.random.default_rng(seed)
    samples  = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        samples[i] = metric_fn(y_true[idx], y_pred[idx])

    alpha   = 1.0 - ci_level
    lo      = np.percentile(samples, 100 * alpha / 2)
    hi      = np.percentile(samples, 100 * (1 - alpha / 2))

    return BootstrapResult(
        metric_name=metric_name,
        observed=observed,
        mean=float(samples.mean()),
        std=float(samples.std()),
        ci_lower=float(lo),
        ci_upper=float(hi),
        ci_level=ci_level,
        n_bootstrap=n_bootstrap,
    )


def bootstrap_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, BootstrapResult]:
    """
    Compute bootstrap CIs for all four built-in metrics simultaneously.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Ground truth and predicted labels.
    n_bootstrap : int
        Number of bootstrap resamples.
    ci_level : float
        Confidence level (default 0.95).
    seed : int
        Random seed.

    Returns
    -------
    Dict[str, BootstrapResult]
        Keys: "recall", "precision", "f1", "accuracy".
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n      = len(y_true)
    alpha  = 1.0 - ci_level

    observed = {k: fn(y_true, y_pred) for k, fn in BUILTIN_METRICS.items()}

    rng    = np.random.default_rng(seed)
    # Bootstrap once for all metrics (share resamples for efficiency)
    boot_samples: Dict[str, list] = {k: [] for k in BUILTIN_METRICS}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt  = y_true[idx]
        yp  = y_pred[idx]
        for k, fn in BUILTIN_METRICS.items():
            boot_samples[k].append(fn(yt, yp))

    results: Dict[str, BootstrapResult] = {}
    for k in BUILTIN_METRICS:
        arr = np.array(boot_samples[k])
        lo  = float(np.percentile(arr, 100 * alpha / 2))
        hi  = float(np.percentile(arr, 100 * (1 - alpha / 2)))
        results[k] = BootstrapResult(
            metric_name=k,
            observed=observed[k],
            mean=float(arr.mean()),
            std=float(arr.std()),
            ci_lower=lo,
            ci_upper=hi,
            ci_level=ci_level,
            n_bootstrap=n_bootstrap,
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap comparison: model A vs model B
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_comparison(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    metric_name: str = "metric",
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> BootstrapComparison:
    """
    Bootstrap CI and two-sided p-value for the difference (B - A) in a metric.

    The p-value is computed as the fraction of bootstrap resamples where
    the null hypothesis (delta_boot ≥ |delta_observed|) holds, following
    the "shift" method of Efron & Tibshirani.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels (shared by both models).
    y_pred_a : np.ndarray
        Predictions from model A (e.g. LSTM).
    y_pred_b : np.ndarray
        Predictions from model B (e.g. Transformer).
    metric_fn : callable or str
        Metric function or BUILTIN_METRICS key.
    metric_name : str
        Label for the comparison.
    n_bootstrap : int
        Number of bootstrap resamples.
    ci_level : float
        Confidence level (default 0.95).
    seed : int
        Random seed.

    Returns
    -------
    BootstrapComparison
    """
    if isinstance(metric_fn, str):
        metric_name = metric_fn
        metric_fn   = BUILTIN_METRICS[metric_fn]

    y_true   = np.asarray(y_true,    dtype=int)
    y_pred_a = np.asarray(y_pred_a,  dtype=int)
    y_pred_b = np.asarray(y_pred_b,  dtype=int)
    n        = len(y_true)

    obs_a = metric_fn(y_true, y_pred_a)
    obs_b = metric_fn(y_true, y_pred_b)
    delta_observed = obs_b - obs_a

    rng    = np.random.default_rng(seed)
    deltas = np.empty(n_bootstrap, dtype=float)
    vals_a = np.empty(n_bootstrap, dtype=float)
    vals_b = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        idx       = rng.integers(0, n, size=n)
        val_a     = metric_fn(y_true[idx], y_pred_a[idx])
        val_b     = metric_fn(y_true[idx], y_pred_b[idx])
        deltas[i] = val_b - val_a
        vals_a[i] = val_a
        vals_b[i] = val_b

    alpha    = 1.0 - ci_level
    lo       = float(np.percentile(deltas, 100 * alpha / 2))
    hi       = float(np.percentile(deltas, 100 * (1 - alpha / 2)))

    # Two-sided p-value: fraction of |bootstrap_delta| >= |observed_delta|
    # using the shifted null distribution
    shifted_deltas = deltas - deltas.mean()
    p_value = float(np.mean(np.abs(shifted_deltas) >= np.abs(delta_observed)))

    # Effect size: Cohen's d
    pooled_std = np.sqrt((vals_a.var() + vals_b.var()) / 2)
    effect_size = float(abs(delta_observed) / pooled_std) if pooled_std > 0 else 0.0

    return BootstrapComparison(
        metric_name=metric_name,
        delta_observed=delta_observed,
        delta_mean=float(deltas.mean()),
        delta_std=float(deltas.std()),
        ci_lower=lo,
        ci_upper=hi,
        p_value=p_value,
        ci_level=ci_level,
        n_bootstrap=n_bootstrap,
        effect_size=effect_size,
    )


# ─────────────────────────────────────────────────────────────────────────────
# McNemar's test
# ─────────────────────────────────────────────────────────────────────────────

def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    significance_level: float = 0.05,
) -> McNemarResult:
    """
    McNemar's test for two paired binary classifiers.

    Tests whether A and B make significantly different errors on the same data.
    Uses the mid-p-value variant with continuity correction (Fagerland et al. 2013).

    H0: The two classifiers have the same error rate.
    H1: They differ.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels.
    y_pred_a : np.ndarray
        Binary predictions from model A.
    y_pred_b : np.ndarray
        Binary predictions from model B.
    significance_level : float
        p-value threshold for significance (default 0.05).

    Returns
    -------
    McNemarResult
    """
    from scipy.stats import chi2  # type: ignore

    y_true   = np.asarray(y_true,    dtype=int)
    y_pred_a = np.asarray(y_pred_a,  dtype=int)
    y_pred_b = np.asarray(y_pred_b,  dtype=int)

    # Discordant pairs
    # b: A correct, B wrong
    # c: A wrong, B correct
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    b = int(( correct_a & ~correct_b).sum())   # A right, B wrong
    c = int((~correct_a &  correct_b).sum())   # A wrong, B right
    n_discordant = b + c

    if n_discordant == 0:
        logger.warning("[McNemar] No discordant pairs — models make identical errors.")
        return McNemarResult(
            statistic=0.0, p_value=1.0, b=0, c=0,
            n_discordant=0, is_significant=False,
        )

    # Chi-squared with continuity correction (Yates' correction)
    statistic = (abs(b - c) - 1) ** 2 / n_discordant
    p_value   = float(1 - chi2.cdf(statistic, df=1))

    return McNemarResult(
        statistic=statistic,
        p_value=p_value,
        b=b,
        c=c,
        n_discordant=n_discordant,
        is_significant=(p_value < significance_level),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Paired bootstrap significance test (for continuous scores)
# ─────────────────────────────────────────────────────────────────────────────

def paired_bootstrap_test(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    threshold_a: float,
    threshold_b: float,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    metric_name: str = "metric",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> BootstrapComparison:
    """
    Paired bootstrap significance test using continuous anomaly scores.

    Scores are thresholded before computing the metric on each bootstrap
    resample. Useful when comparing two models on the same test set.

    Parameters
    ----------
    scores_a, scores_b : np.ndarray
        Continuous reconstruction errors (anomaly scores).
    threshold_a, threshold_b : float
        Thresholds to convert scores to binary predictions.
    """
    y_pred_a = (scores_a > threshold_a).astype(int)
    y_pred_b = (scores_b > threshold_b).astype(int)

    return bootstrap_comparison(
        y_true=y_true,
        y_pred_a=y_pred_a,
        y_pred_b=y_pred_b,
        metric_fn=metric_fn,
        metric_name=metric_name,
        n_bootstrap=n_bootstrap,
        ci_level=0.95,
        seed=seed,
    )
