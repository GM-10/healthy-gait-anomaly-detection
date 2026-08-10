"""
evaluation_framework/tables.py

Publication-ready table export in CSV, Markdown, and LaTeX formats.

Generates:
    - Comparison table: metrics × models × threshold methods × modalities
    - Statistics table: bootstrap CIs and McNemar test results
    - Severity table: per-anomaly-type, per-severity metrics
    - Fusion table: modality contributions and fusion strategy comparison

All LaTeX tables use booktabs formatting (\\toprule, \\midrule, \\bottomrule)
and are ready for direct inclusion in IEEE/ACM/Springer documents.

Usage:
    from evaluation_framework.tables import export_all_tables

    paths = export_all_tables(
        output_dir="outputs/evaluation/tables",
        comparison_df=comparison_df,
        stats_report=stats_report,
        severity_report=severity_report,
        fusion_report=fusion_report,
    )
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Column display names for LaTeX / Markdown
# ─────────────────────────────────────────────────────────────────────────────

METRIC_DISPLAY = {
    "precision":        "Precision",
    "recall":           "Recall",
    "f1":               "F1",
    "accuracy":         "Accuracy",
    "fpr":              "FPR",
    "fnr":              "FNR",
    "roc_auc":          "ROC-AUC",
    "pr_auc":           "PR-AUC",
    "threshold_value":  "Threshold",
    "threshold_method": "Method",
    "model_name":       "Model",
    "modality":         "Modality",
    "channel_name":     "Channel",
    "TP":               "TP",
    "FP":               "FP",
    "TN":               "TN",
    "FN":               "FN",
    "n_windows":        "N",
}

METHOD_DISPLAY = {
    "mean_std":     r"$\mu+3\sigma$",
    "percentile95": r"P95",
    "percentile99": r"P99",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_float(v, decimals: int = 4) -> str:
    """Format a float for table display; return 'N/A' for NaN."""
    try:
        if np.isnan(v):
            return "N/A"
        return f"{v:.{decimals}f}"
    except Exception:
        return str(v)


def _df_to_markdown(df: pd.DataFrame, float_decimals: int = 4) -> str:
    """Convert a DataFrame to a Markdown table string."""
    def fmt_cell(v):
        if isinstance(v, float):
            return _fmt_float(v, float_decimals)
        return str(v)

    # Header
    cols   = list(df.columns)
    header = "| " + " | ".join(METRIC_DISPLAY.get(c, c) for c in cols) + " |"
    sep    = "| " + " | ".join(["---"] * len(cols)) + " |"

    rows = [header, sep]
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(fmt_cell(row[c]) for c in cols) + " |")

    return "\n".join(rows) + "\n"


def _df_to_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    float_decimals: int = 4,
    highlight_best: bool = True,
    best_cols: Optional[List[str]] = None,
    bold_best: bool = True,
) -> str:
    """
    Convert a DataFrame to a LaTeX booktabs table.

    Parameters
    ----------
    df : pd.DataFrame
    caption : str
        Table caption.
    label : str
        LaTeX label (e.g. 'tab:comparison').
    float_decimals : int
        Decimal places for float columns.
    highlight_best : bool
        Bold the best value in each numeric column (if best_cols is specified).
    best_cols : list of str, optional
        Columns to bold the best (max) value in.
    bold_best : bool
        Whether to use \\textbf{} for the best value.

    Returns
    -------
    str
        Complete LaTeX table source.
    """
    cols = list(df.columns)
    n_cols = len(cols)
    col_spec = "l" * 2 + "r" * (n_cols - 2)  # first 2 left, rest right

    # Find best values
    best_vals: Dict[str, float] = {}
    if highlight_best and best_cols:
        for c in best_cols:
            if c in df.columns:
                try:
                    best_vals[c] = df[c].dropna().max()
                except Exception:
                    pass

    def fmt_cell(col, v):
        if isinstance(v, float):
            s = _fmt_float(v, float_decimals)
            if bold_best and col in best_vals and not np.isnan(v):
                if abs(v - best_vals[col]) < 1e-9:
                    return r"\textbf{" + s + r"}"
            return s
        # Apply method display name in LaTeX
        if col == "threshold_method" and str(v) in METHOD_DISPLAY:
            return METHOD_DISPLAY[str(v)]
        return str(v).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")

    header_cells = [METRIC_DISPLAY.get(c, c).replace("_", r"\_") for c in cols]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{" + caption + r"}",
        r"\label{" + label + r"}",
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
        " & ".join(header_cells) + r" \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        cells = [fmt_cell(c, row[c]) for c in cols]
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Table generators
# ─────────────────────────────────────────────────────────────────────────────

def generate_comparison_table(comparison_df: pd.DataFrame) -> Dict[str, str]:
    """
    Generate the main metrics comparison table.

    Returns dict with keys 'csv', 'markdown', 'latex'.
    """
    # Reorder / select display columns
    display_cols = [
        "model_name", "modality", "channel_name", "threshold_method",
        "precision", "recall", "f1", "accuracy", "fpr", "fnr",
        "roc_auc", "pr_auc",
        "TP", "FP", "TN", "FN", "n_windows",
    ]
    df = comparison_df[[c for c in display_cols if c in comparison_df.columns]].copy()

    best_cols = ["precision", "recall", "f1", "accuracy", "roc_auc", "pr_auc"]

    return {
        "csv":      df.to_csv(index=False),
        "markdown": _df_to_markdown(df),
        "latex":    _df_to_latex(
            df,
            caption="Anomaly Detection Performance Comparison Across Models, Modalities, and Threshold Methods",
            label="tab:comparison",
            best_cols=best_cols,
        ),
    }


def generate_statistics_table(stats_report) -> Dict[str, str]:
    """
    Generate the bootstrap CI and McNemar test table.

    Parameters
    ----------
    stats_report : StatisticsReport (from statistics.py)
    """
    results: Dict[str, str] = {"csv": "", "markdown": "", "latex": ""}

    if not stats_report.bootstrap_ci_df.empty:
        df = stats_report.bootstrap_ci_df.copy()
        display_cols = [
            "model_name", "modality", "threshold_method", "metric",
            "observed", "ci_lower", "ci_upper", "ci_std",
        ]
        df = df[[c for c in display_cols if c in df.columns]]
        results["csv"]      = df.to_csv(index=False)
        results["markdown"] = _df_to_markdown(df)
        results["latex"]    = _df_to_latex(
            df,
            caption=r"Bootstrap Confidence Intervals (95\%, 1000 resamples)",
            label="tab:bootstrap_ci",
            best_cols=["observed"],
        )

    return results


def generate_severity_table(severity_report) -> Dict[str, str]:
    """Generate severity analysis summary table with bootstrap recall CIs."""
    if severity_report.aggregate_df.empty:
        return {"csv": "", "markdown": "", "latex": ""}

    df = severity_report.aggregate_df.copy()
    display_cols = [
        "model_name", "modality", "anomaly_type", "severity_numeric",
        "threshold_method",
        "mean_error", "recall", "recall_ci_lower", "recall_ci_upper",
        "f1", "threshold_margin",
        "TP", "FN", "n_windows",
    ]
    df = df[[c for c in display_cols if c in df.columns]]

    return {
        "csv":      df.to_csv(index=False),
        "markdown": _df_to_markdown(df),
        "latex":    _df_to_latex(
            df,
            caption="Severity-Stratified Anomaly Detection Analysis (with 95\\% Bootstrap Recall CI)",
            label="tab:severity",
            best_cols=["recall", "f1"],
        ),
    }


def generate_fusion_table(
    fusion_report=None,
    fusion_metrics_df: Optional[pd.DataFrame] = None,
) -> Dict[str, str]:
    """
    Generate the fusion strategy comparison table.

    Parameters
    ----------
    fusion_report : FusionReport, optional
    fusion_metrics_df : pd.DataFrame, optional
        Pre-built fusion metrics (from evaluate_fusion_strategies).
    """
    if fusion_metrics_df is not None and not fusion_metrics_df.empty:
        df = fusion_metrics_df.copy()
    elif fusion_report is not None and not fusion_report.incremental_recall_df.empty:
        df = fusion_report.incremental_recall_df.copy()
    else:
        return {"csv": "", "markdown": "", "latex": ""}

    display_cols = [c for c in [
        "model_name", "fusion_strategy", "threshold_method",
        "precision", "recall", "f1", "accuracy",
        "semg_recall", "kinkin_recall", "or_recall",
        "semg_incremental_gain", "kinkin_incremental_gain",
        "TP", "FP", "TN", "FN", "n_windows",
    ] if c in df.columns]
    df = df[display_cols]

    return {
        "csv":      df.to_csv(index=False),
        "markdown": _df_to_markdown(df),
        "latex":    _df_to_latex(
            df,
            caption="Late Fusion Strategy Comparison and Modality Contribution",
            label="tab:fusion",
            best_cols=["recall", "f1"],
        ),
    }


def generate_threshold_comparison_table(
    threshold_comparison_df: pd.DataFrame,
) -> Dict[str, str]:
    """
    Generate a per-channel threshold comparison table showing:
    - Threshold values for each method
    - Relative differences: (Pxx - mean_std) / mean_std %
    - Label-agreement rates between method pairs
    - Label flip counts when switching methods
    """
    if threshold_comparison_df is None or threshold_comparison_df.empty:
        return {"csv": "", "markdown": "", "latex": ""}

    df = threshold_comparison_df.copy()

    # Display column order (include whatever columns exist)
    preferred_order = [
        "model_name", "modality", "channel_name", "n_windows",
        "thresh_mean_std", "thresh_percentile95", "thresh_percentile99",
        "rel_diff_percentile95_vs_mean_std_pct",
        "rel_diff_percentile99_vs_mean_std_pct",
        "agreement_mean_std_vs_percentile95",
        "agreement_mean_std_vs_percentile99",
        "agreement_percentile95_vs_percentile99",
        "flip_0to1_mean_std_vs_percentile95",
        "flip_1to0_mean_std_vs_percentile95",
        "flip_0to1_mean_std_vs_percentile99",
        "flip_1to0_mean_std_vs_percentile99",
    ]
    df = df[[c for c in preferred_order if c in df.columns]]

    return {
        "csv":      df.to_csv(index=False),
        "markdown": _df_to_markdown(df),
        "latex":    _df_to_latex(
            df,
            caption=r"Threshold Comparison: $\mu+3\sigma$ vs P95 vs P99 "
                    r"--- Values, Relative Differences, and Label Agreement Rates",
            label="tab:threshold_comparison",
            best_cols=["agreement_mean_std_vs_percentile95",
                       "agreement_mean_std_vs_percentile99"],
        ),
    }


def generate_kruskal_table(kruskal_df: pd.DataFrame) -> Dict[str, str]:
    """
    Generate a Kruskal-Wallis H-test results table (severity significance).
    """
    if kruskal_df is None or kruskal_df.empty:
        return {"csv": "", "markdown": "", "latex": ""}

    df = kruskal_df.copy()
    display_cols = [
        "model_name", "modality", "channel_name", "anomaly_type",
        "kruskal_H", "kruskal_p", "kruskal_sig",
        "n_severity_levels", "posthoc_mannwhitney_bonferroni",
    ]
    df = df[[c for c in display_cols if c in df.columns]]

    return {
        "csv":      df.to_csv(index=False),
        "markdown": _df_to_markdown(df),
        "latex":    _df_to_latex(
            df,
            caption="Kruskal-Wallis Test: Reconstruction Error vs Severity Level "
                    "(with Bonferroni-corrected Mann-Whitney U post-hoc tests)",
            label="tab:kruskal_wallis",
            best_cols=[],
        ),
    }


def generate_fusion_significance_table(
    significance_df: pd.DataFrame,
) -> Dict[str, str]:
    """
    Generate the pairwise fusion strategy bootstrap + McNemar significance table.
    """
    if significance_df is None or significance_df.empty:
        return {"csv": "", "markdown": "", "latex": ""}

    df = significance_df.copy()
    display_cols = [
        "model_name", "strategy_A", "strategy_B",
        "recall_A", "recall_B", "delta_recall_obs",
        "delta_recall_ci_lo", "delta_recall_ci_hi",
        "p_recall_bonf", "sig_recall_bonf",
        "f1_A", "f1_B", "delta_f1_obs",
        "delta_f1_ci_lo", "delta_f1_ci_hi",
        "p_f1_bonf", "sig_f1_bonf",
        "mcnemar_chi2", "mcnemar_p_bonf", "mcnemar_sig_bonf",
        "effect_size_d", "n_windows",
    ]
    df = df[[c for c in display_cols if c in df.columns]]

    return {
        "csv":      df.to_csv(index=False),
        "markdown": _df_to_markdown(df),
        "latex":    _df_to_latex(
            df,
            caption="Pairwise Fusion Strategy Significance Tests "
                    "(Bootstrap 95\\% CI + McNemar, Bonferroni-corrected, $n=1000$ resamples)",
            label="tab:fusion_significance",
            best_cols=[],
        ),
    }


def generate_or_dominance_table(
    or_dominance_df: pd.DataFrame,
) -> Dict[str, str]:
    """
    Generate the OR-fusion mathematical dominance decomposition table.
    """
    if or_dominance_df is None or or_dominance_df.empty:
        return {"csv": "", "markdown": "", "latex": ""}

    df = or_dominance_df.copy()
    display_cols = [
        "model_name",
        "recall_semg", "recall_kinkin", "recall_or",
        "precision_semg", "precision_kinkin", "precision_or",
        "f1_semg", "f1_kinkin", "f1_or",
        "recall_lower_bound", "recall_bound_holds",
        "precision_upper_bound", "precision_penalty_holds",
        "complementarity_score",
        "n_only_semg", "n_only_kinkin", "n_both_fire", "n_neither",
        "extra_tp_from_or", "extra_fp_from_or",
        "or_f1_exceeds_semg", "or_f1_exceeds_kinkin",
        "or_f1_exceeds_both_modalities",
        "best_single_f1", "or_f1_gain_vs_best_single",
        "n_windows",
    ]
    df = df[[c for c in display_cols if c in df.columns]]

    return {
        "csv":      df.to_csv(index=False),
        "markdown": _df_to_markdown(df),
        "latex":    _df_to_latex(
            df,
            caption="OR-Fusion Recall Dominance Decomposition: "
                    "Recall Bound, Precision Penalty, and Modality Complementarity",
            label="tab:or_dominance",
            best_cols=["recall_or", "f1_or", "complementarity_score"],
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Master export
# ─────────────────────────────────────────────────────────────────────────────

def export_all_tables(
    output_dir: str,
    comparison_df: Optional[pd.DataFrame] = None,
    stats_report=None,
    severity_report=None,
    fusion_report=None,
    fusion_metrics_df: Optional[pd.DataFrame] = None,
    threshold_comparison_df: Optional[pd.DataFrame] = None,
) -> Dict[str, str]:
    """
    Generate and write all tables to disk.

    Parameters
    ----------
    output_dir : str
        Directory to write files into. Will be created if it doesn't exist.
    comparison_df : pd.DataFrame, optional
        Output of build_comparison_dataframe().
    stats_report : StatisticsReport, optional
        Output of run_statistical_analysis().
    severity_report : SeverityReport, optional
        Output of SeverityAnalyzer.analyze().
    fusion_report : FusionReport, optional
        Output of FusionAnalyzer.analyze().
    fusion_metrics_df : pd.DataFrame, optional
        Output of evaluate_fusion_strategies().
    threshold_comparison_df : pd.DataFrame, optional
        Output of compare_thresholds() from thresholds.py.

    Returns
    -------
    Dict[str, str]
        Paths to all written files.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    table_specs = []

    if comparison_df is not None and not comparison_df.empty:
        table_specs.append(("comparison", generate_comparison_table(comparison_df)))

    if stats_report is not None:
        table_specs.append(("statistics", generate_statistics_table(stats_report)))

    if severity_report is not None:
        table_specs.append(("severity", generate_severity_table(severity_report)))
        # Kruskal-Wallis severity significance table
        if hasattr(severity_report, "kruskal_df") and not severity_report.kruskal_df.empty:
            table_specs.append(("severity_kruskal", generate_kruskal_table(severity_report.kruskal_df)))

    if fusion_report is not None or fusion_metrics_df is not None:
        table_specs.append(("fusion", generate_fusion_table(fusion_report, fusion_metrics_df)))
        # Significance tests
        if fusion_report is not None and hasattr(fusion_report, "significance_df") and not fusion_report.significance_df.empty:
            table_specs.append(("fusion_significance", generate_fusion_significance_table(fusion_report.significance_df)))
        # OR dominance
        if fusion_report is not None and hasattr(fusion_report, "or_dominance_df") and not fusion_report.or_dominance_df.empty:
            table_specs.append(("fusion_or_dominance", generate_or_dominance_table(fusion_report.or_dominance_df)))

    if threshold_comparison_df is not None and not threshold_comparison_df.empty:
        table_specs.append(("threshold_comparison", generate_threshold_comparison_table(threshold_comparison_df)))

    for table_name, content in table_specs:
        for fmt, data in content.items():
            if not data:
                continue
            ext = fmt
            fname = f"{table_name}_table.{ext}"
            path  = os.path.join(output_dir, fname)
            with open(path, "w", encoding="utf-8") as f:
                f.write(data)
            paths[fname] = path
            logger.info(f"[Tables] {fname} → {path}")

    logger.info(f"[Tables] Exported {len(paths)} files to {output_dir}")
    return paths
