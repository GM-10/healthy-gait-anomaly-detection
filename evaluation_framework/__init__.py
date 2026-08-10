"""
evaluation_framework/__init__.py

Modular research evaluation framework for the SIAT-LLMD gait anomaly detection project.

Provides:
    - Configurable thresholding (MeanStd, Percentile95, Percentile99)
    - Threshold comparison: compare_thresholds() for side-by-side P95/P99 vs μ+3σ analysis
    - Full metric suite (Precision, Recall, F1, Accuracy, ROC-AUC, PR-AUC, FPR, FNR)
    - Statistical validation (bootstrap CIs, McNemar's test, Kruskal-Wallis, paired bootstrap)
    - Per-stratum recall bootstrap CIs in severity analysis
    - Pairwise fusion significance tests (OR vs AND vs MAJORITY)
    - OR-fusion dominance decomposition (recall bound, precision penalty, complementarity)
    - Diagnostic plots (error distributions, severity curves with CI bars, ROC/PR/DET)
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
    fit_all_thresholds,
    compare_thresholds,
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
from evaluation_framework.severity_analysis import SeverityAnalyzer, SeverityReport
from evaluation_framework.fusion_analysis import FusionAnalyzer, FusionReport

__all__ = [
    # Thresholds
    "MeanStdThreshold",
    "PercentileThreshold",
    "ThresholdResult",
    "get_threshold",
    "fit_all_thresholds",
    "compare_thresholds",
    "THRESHOLD_METHODS",
    # Evaluation
    "compute_metrics",
    "evaluate_all_thresholds",
    "MetricsResult",
    # Bootstrap / stats
    "bootstrap_metric",
    "bootstrap_comparison",
    "mcnemar_test",
    # Severity
    "SeverityAnalyzer",
    "SeverityReport",
    # Fusion
    "FusionAnalyzer",
    "FusionReport",
]
