"""
evaluation_framework/statistics.py

Statistical analysis orchestration for the gait anomaly detection evaluation.

Produces:
    - Per-metric 95% bootstrap CIs for every (model, modality, threshold_method)
    - McNemar's test: LSTM vs Transformer for each modality
    - Paired bootstrap significance table for all metric × model pairs
    - Effect sizes (Cohen's d)
    - CSV and LaTeX export

Usage:
    from evaluation_framework.statistics import run_statistical_analysis, export_statistics

    stats = run_statistical_analysis(
        comparison_df=metrics_df,
        score_df=raw_score_df,
        n_bootstrap=1000,
        ci_level=0.95,
        seed=42,
    )
    export_statistics(stats, output_dir="outputs/evaluation/statistics")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from evaluation_framework.bootstrap import (
    bootstrap_all_metrics,
    bootstrap_comparison,
    mcnemar_test,
    BootstrapResult,
    BootstrapComparison,
    McNemarResult,
    BUILTIN_METRICS,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Specificity / Balanced Accuracy helper
# ─────────────────────────────────────────────────────────────────────────────

def compute_specificity_and_balanced_accuracy(
    tn: int,
    fp: int,
    fn: int,
    tp: int,
) -> dict:
    """
    Compute specificity and balanced accuracy from a confusion matrix.

    Returns ``None`` for any metric whose denominator is zero rather than
    raising ``ZeroDivisionError``.  This can happen legitimately when the
    test set contains only anomalies (TN+FP=0) or only normal windows
    (TP+FN=0).

    Parameters
    ----------
    tn, fp, fn, tp : int
        True negatives, false positives, false negatives, true positives.

    Returns
    -------
    dict with keys:
        ``specificity``       — TN / (TN + FP), or None if undefined
        ``balanced_accuracy`` — (recall + specificity) / 2, or None if either
                                component is undefined
    """
    # Specificity = TN / (TN + FP).  Undefined when there are no true negatives
    # and no false positives (i.e. no actual-negative windows in the split).
    specificity: Optional[float] = (
        tn / (tn + fp) if (tn + fp) > 0 else None
    )

    # Recall = TP / (TP + FN).  Undefined when there are no actual-positive
    # windows in the split.
    recall: Optional[float] = (
        tp / (tp + fn) if (tp + fn) > 0 else None
    )

    # Balanced accuracy requires both components to be defined.
    balanced_accuracy: Optional[float] = (
        (recall + specificity) / 2
        if (recall is not None and specificity is not None)
        else None
    )

    return {
        "specificity":       specificity,
        "balanced_accuracy": balanced_accuracy,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StatisticsReport:
    """Container for all statistical analysis results."""

    # Bootstrap CIs — indexed by (model, modality, threshold_method, metric)
    bootstrap_ci_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # McNemar test results — one row per (modality, threshold_method)
    mcnemar_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Pairwise comparison (model A vs B) — one row per (metric, modality, method)
    comparison_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Config
    n_bootstrap: int   = 1000
    ci_level:    float = 0.95
    seed:        int   = 42


# ─────────────────────────────────────────────────────────────────────────────
# Core analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_statistical_analysis(
    metrics_df: pd.DataFrame,
    score_df: pd.DataFrame,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> StatisticsReport:
    """
    Run the full statistical analysis pipeline.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Output of evaluate_all_thresholds(). Must contain:
        model_name, modality, channel_name, threshold_method, threshold_value.
    score_df : pd.DataFrame
        Raw score CSV data with reconstruction_error, is_synthetic_anomaly,
        model_name, channel_name, modality columns.
    n_bootstrap : int
        Number of bootstrap resamples (minimum 1000 for reliable CIs).
    ci_level : float
        Confidence level (default 0.95 → 95% CI).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    StatisticsReport
    """
    report = StatisticsReport(n_bootstrap=n_bootstrap, ci_level=ci_level, seed=seed)

    models         = sorted(score_df["model_name"].unique())
    modalities     = sorted(score_df["modality"].unique())

    # Try to determine threshold methods from metrics_df
    threshold_methods = (
        sorted(metrics_df["threshold_method"].unique())
        if "threshold_method" in metrics_df.columns
        else ["mean_std"]
    )

    logger.info(f"[Statistics] Models: {models}")
    logger.info(f"[Statistics] Modalities: {modalities}")
    logger.info(f"[Statistics] Threshold methods: {threshold_methods}")
    logger.info(f"[Statistics] n_bootstrap={n_bootstrap}, ci_level={ci_level}")

    # ── 1. Bootstrap CIs ─────────────────────────────────────────────────────
    ci_records: List[Dict[str, Any]] = []

    for method in threshold_methods:
        method_metrics = metrics_df[metrics_df["threshold_method"] == method] \
            if "threshold_method" in metrics_df.columns else metrics_df

        for model_name in models:
            for modality in modalities:
                subset = method_metrics[
                    (method_metrics["model_name"] == model_name) &
                    (method_metrics["modality"]   == modality)
                ]
                if subset.empty:
                    continue

                # Build y_true / y_pred for this (model, modality, threshold method)
                raw = score_df[
                    (score_df["model_name"] == model_name) &
                    (score_df["modality"]   == modality)
                ]
                if raw.empty:
                    continue

                # Get the threshold value for each channel and re-derive predictions
                thresh_vals = dict(zip(
                    subset["channel_name"],
                    subset["threshold_value"],
                ))
                y_true_all: List[int] = []
                y_pred_all: List[int] = []

                for ch_name, ch_thresh in thresh_vals.items():
                    ch = raw[raw["channel_name"] == ch_name]
                    if ch.empty:
                        continue
                    y_true_all.extend(ch["is_synthetic_anomaly"].astype(int).tolist())
                    preds = (ch["reconstruction_error"].values > ch_thresh).astype(int)
                    y_pred_all.extend(preds.tolist())

                if len(y_true_all) < 10:
                    continue

                y_true_arr = np.array(y_true_all, dtype=int)
                y_pred_arr = np.array(y_pred_all, dtype=int)

                logger.info(
                    f"  [CI] {model_name}/{modality}/{method}: "
                    f"n={len(y_true_arr)}"
                )

                ci_results = bootstrap_all_metrics(
                    y_true_arr, y_pred_arr,
                    n_bootstrap=n_bootstrap,
                    ci_level=ci_level,
                    seed=seed,
                )

                for metric_name, br in ci_results.items():
                    ci_records.append({
                        "model_name":       model_name,
                        "modality":         modality,
                        "threshold_method": method,
                        "metric":           metric_name,
                        "observed":         br.observed,
                        "ci_lower":         br.ci_lower,
                        "ci_upper":         br.ci_upper,
                        "ci_mean":          br.mean,
                        "ci_std":           br.std,
                        "ci_level":         ci_level,
                        "n_bootstrap":      n_bootstrap,
                    })

    report.bootstrap_ci_df = pd.DataFrame(ci_records)
    logger.info(f"[Statistics] Bootstrap CI: {len(ci_records)} rows computed")

    # ── 2. McNemar's test: LSTM vs Transformer per modality ─────────────────
    mcnemar_records: List[Dict[str, Any]] = []

    model_pairs = [(a, b) for i, a in enumerate(models) for b in models[i + 1:]]

    for method in threshold_methods:
        method_metrics = metrics_df[metrics_df["threshold_method"] == method] \
            if "threshold_method" in metrics_df.columns else metrics_df

        for modality in modalities:
            raw_mod = score_df[score_df["modality"] == modality]
            if raw_mod.empty:
                continue

            for model_a, model_b in model_pairs:
                thresh_a = method_metrics[
                    (method_metrics["model_name"] == model_a) &
                    (method_metrics["modality"]   == modality)
                ].set_index("channel_name")["threshold_value"].to_dict()

                thresh_b = method_metrics[
                    (method_metrics["model_name"] == model_b) &
                    (method_metrics["modality"]   == modality)
                ].set_index("channel_name")["threshold_value"].to_dict()

                if not thresh_a or not thresh_b:
                    continue

                # Build window-level paired predictions (union of channels)
                common_channels = set(thresh_a) & set(thresh_b)
                if not common_channels:
                    continue

                y_true_all: List[int] = []
                y_pred_a_all: List[int] = []
                y_pred_b_all: List[int] = []

                raw_a = raw_mod[raw_mod["model_name"] == model_a]
                raw_b = raw_mod[raw_mod["model_name"] == model_b]

                for ch in sorted(common_channels):
                    ch_a = raw_a[raw_a["channel_name"] == ch]
                    ch_b = raw_b[raw_b["channel_name"] == ch]

                    # Align on window_id × subject_id × movement
                    merge_keys = ["subject_id", "movement", "window_id"]
                    available_keys = [k for k in merge_keys if k in ch_a.columns and k in ch_b.columns]
                    if not available_keys:
                        continue

                    merged = pd.merge(
                        ch_a[available_keys + ["reconstruction_error", "is_synthetic_anomaly"]],
                        ch_b[available_keys + ["reconstruction_error"]],
                        on=available_keys,
                        suffixes=("_a", "_b"),
                    )
                    if merged.empty:
                        continue

                    y_true_all.extend(merged["is_synthetic_anomaly"].astype(int).tolist())
                    y_pred_a_all.extend(
                        (merged["reconstruction_error_a"] > thresh_a[ch]).astype(int).tolist()
                    )
                    y_pred_b_all.extend(
                        (merged["reconstruction_error_b"] > thresh_b[ch]).astype(int).tolist()
                    )

                if len(y_true_all) < 20:
                    continue

                y_true_arr  = np.array(y_true_all,   dtype=int)
                y_pred_a_arr = np.array(y_pred_a_all, dtype=int)
                y_pred_b_arr = np.array(y_pred_b_all, dtype=int)

                mc = mcnemar_test(y_true_arr, y_pred_a_arr, y_pred_b_arr)
                comp = bootstrap_comparison(
                    y_true_arr, y_pred_a_arr, y_pred_b_arr,
                    metric_fn="f1",
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                )

                mcnemar_records.append({
                    "model_a":          model_a,
                    "model_b":          model_b,
                    "modality":         modality,
                    "threshold_method": method,
                    "mcnemar_stat":     mc.statistic,
                    "mcnemar_p":        mc.p_value,
                    "mcnemar_b":        mc.b,
                    "mcnemar_c":        mc.c,
                    "n_discordant":     mc.n_discordant,
                    "mcnemar_sig":      mc.is_significant,
                    "f1_delta":         comp.delta_observed,
                    "f1_ci_lower":      comp.ci_lower,
                    "f1_ci_upper":      comp.ci_upper,
                    "f1_p_value":       comp.p_value,
                    "effect_size_d":    comp.effect_size,
                    "is_significant":   comp.is_significant,
                    "n_windows":        len(y_true_all),
                })
                logger.info(f"  [McNemar] {model_a} vs {model_b} / {modality}/{method}: {mc}")

    report.mcnemar_df = pd.DataFrame(mcnemar_records)
    logger.info(f"[Statistics] McNemar: {len(mcnemar_records)} comparisons")

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

def export_statistics(report: StatisticsReport, output_dir: str) -> Dict[str, str]:
    """
    Export the statistical report to CSV and LaTeX.

    Parameters
    ----------
    report : StatisticsReport
    output_dir : str
        Directory to write files into. Will be created if it doesn't exist.

    Returns
    -------
    Dict[str, str]
        Paths to exported files.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    # ── Bootstrap CI ─────────────────────────────────────────────────────────
    if not report.bootstrap_ci_df.empty:
        csv_path = os.path.join(output_dir, "bootstrap_ci.csv")
        report.bootstrap_ci_df.to_csv(csv_path, index=False)
        paths["bootstrap_ci_csv"] = csv_path

        tex_path = os.path.join(output_dir, "bootstrap_ci.tex")
        _export_ci_latex(report.bootstrap_ci_df, tex_path, report.ci_level)
        paths["bootstrap_ci_tex"] = tex_path
        logger.info(f"[Statistics] Bootstrap CI → {csv_path}, {tex_path}")

    # ── McNemar ──────────────────────────────────────────────────────────────
    if not report.mcnemar_df.empty:
        csv_path = os.path.join(output_dir, "mcnemar_tests.csv")
        report.mcnemar_df.to_csv(csv_path, index=False)
        paths["mcnemar_csv"] = csv_path

        tex_path = os.path.join(output_dir, "mcnemar_tests.tex")
        _export_mcnemar_latex(report.mcnemar_df, tex_path)
        paths["mcnemar_tex"] = tex_path
        logger.info(f"[Statistics] McNemar → {csv_path}, {tex_path}")

    return paths


def _export_ci_latex(df: pd.DataFrame, path: str, ci_level: float) -> None:
    """Write a LaTeX booktabs table for bootstrap CI results."""
    ci_pct = int(ci_level * 100)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Bootstrap Confidence Intervals (" + f"{ci_pct}" + r"\% CI, 1000 resamples)}",
        r"\label{tab:bootstrap_ci}",
        r"\begin{tabular}{llllrrrr}",
        r"\toprule",
        r"Model & Modality & Method & Metric & Observed & CI Lower & CI Upper & Std \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        sig_str = ""
        # Mark if CI excludes zero (for deltas) or if > 0 (for recall/f1)
        lines.append(
            f"{row['model_name']} & {row['modality']} & {row['threshold_method']} & "
            f"{row['metric']} & {row['observed']:.4f} & {row['ci_lower']:.4f} & "
            f"{row['ci_upper']:.4f} & {row['ci_std']:.4f} {sig_str} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _export_mcnemar_latex(df: pd.DataFrame, path: str) -> None:
    """Write a LaTeX booktabs table for McNemar test results."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{McNemar's Test and Paired Bootstrap Comparison}",
        r"\label{tab:mcnemar}",
        r"\begin{tabular}{lllrrrrrr}",
        r"\toprule",
        r"Model A & Model B & Modality & Method & $\chi^2$ & $p$ & b & c & Sig. \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        sig = r"$*$" if row.get("mcnemar_sig", False) else ""
        lines.append(
            f"{row['model_a']} & {row['model_b']} & {row['modality']} & "
            f"{row['threshold_method']} & {row['mcnemar_stat']:.3f} & "
            f"{row['mcnemar_p']:.4f} & {row['mcnemar_b']} & {row['mcnemar_c']} & {sig} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\multicolumn{9}{l}{\footnotesize $*$ indicates $p < 0.05$.}",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
