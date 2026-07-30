"""
evaluation_framework/__init__.py

Modular research evaluation framework for the SIAT-LLMD gait anomaly detection project.

Provides:
    - Configurable thresholding (MeanStd, Percentile95, Percentile99)
    - Full metric suite (Precision, Recall, F1, Accuracy, ROC-AUC, PR-AUC, FPR, FNR)
    - Statistical validation (bootstrap CIs, McNemar's test, paired bootstrap)
    - Diagnostic plots (error distributions, severity curves, ROC/PR/DET)
    - Severity-stratified analysis and fusion ablation
    - Publication-ready CSV / Markdown / LaTeX table export

Usage:
    python evaluate.py --config configs/transformer.yaml
"""

from evaluation_framework.thresholds import (
    MeanStdThreshold,
    PercentileThreshold,
    ThresholdResult,
    get_threshold,
    THRESHOLD_METHODS,
)
from evaluation_framework.evaluation import (
    compute_metrics,
    evaluate_all_thresholds,
    MetricsResult,
)
from evaluation_framework.bootstrap import (
    bootstrap_metric,
    bootstrap_comparison,
    mcnemar_test,
)

__all__ = [
    "MeanStdThreshold",
    "PercentileThreshold",
    "ThresholdResult",
    "get_threshold",
    "THRESHOLD_METHODS",
    "compute_metrics",
    "evaluate_all_thresholds",
    "MetricsResult",
    "bootstrap_metric",
    "bootstrap_comparison",
    "mcnemar_test",
]
