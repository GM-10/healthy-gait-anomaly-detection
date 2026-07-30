"""
evaluation_framework/plots.py

Publication-ready figure generation for the gait anomaly detection evaluation.

Generates:
    - Reconstruction error distributions (histogram + KDE + threshold lines)
    - Severity vs. reconstruction error / recall / F1 / threshold margin
    - ROC curves, Precision-Recall curves, DET curves
    - Threshold-performance sweep curves
    - Robustness plots (performance vs threshold method × movement)

All figures are saved as PNG, PDF, and SVG by default (configurable via formats).

Usage:
    from evaluation_framework.plots import (
        plot_error_distribution,
        plot_severity_curves,
        plot_roc_pr_curves,
        plot_all,
    )
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette — consistent across all figures
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "normal":        "#4C9BE8",   # steel blue
    "anomaly":       "#E05F5F",   # warm red
    "mean_std":      "#2E86AB",   # teal
    "percentile95":  "#F18F01",   # amber
    "percentile99":  "#C73E1D",   # deep red
    "LSTM":          "#6A4C93",   # purple
    "Transformer":   "#1982C4",   # blue
    "SARIMA":        "#8AC926",   # green
    "OR":            "#FF595E",
    "MAJORITY":      "#FFCA3A",
    "AND":           "#6A4C93",
}

THRESHOLD_LABELS = {
    "mean_std":     r"$\mu + 3\sigma$",
    "percentile95": "P95",
    "percentile99": "P99",
}


def _save_fig(fig, stem: str, output_dir: str, formats: List[str]) -> None:
    """Save a figure in multiple formats."""
    os.makedirs(output_dir, exist_ok=True)
    for fmt in formats:
        path = os.path.join(output_dir, f"{stem}.{fmt}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"  [Plot] Saved → {path}")


def _setup_matplotlib():
    """Configure matplotlib for publication quality."""
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "font.size":        11,
        "axes.titlesize":   13,
        "axes.labelsize":   12,
        "legend.fontsize":  10,
        "figure.dpi":       150,
        "axes.spines.top":  False,
        "axes.spines.right": False,
        "axes.grid":        True,
        "grid.alpha":       0.3,
    })
    return plt


# ─────────────────────────────────────────────────────────────────────────────
# Task 4: Reconstruction error distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_error_distribution(
    train_errors: np.ndarray,
    test_errors_normal: np.ndarray,
    test_errors_anomaly: np.ndarray,
    threshold_results: Dict[str, Any],   # {method: ThresholdResult}
    title: str,
    stem: str,
    output_dir: str,
    formats: List[str] = None,
) -> None:
    """
    Histogram + KDE of reconstruction errors with threshold lines.

    Plots three overlapping distributions:
        - Training errors (healthy baseline)
        - Test clean errors (normal windows)
        - Test anomalous errors (injected anomaly windows)

    Plus vertical lines for each threshold method.

    Parameters
    ----------
    train_errors : np.ndarray
        Training reconstruction errors from healthy windows.
    test_errors_normal : np.ndarray
        Test reconstruction errors for clean windows.
    test_errors_anomaly : np.ndarray
        Test reconstruction errors for anomalous windows.
    threshold_results : dict
        {method_name: ThresholdResult} from thresholds.fit_all_thresholds().
    title : str
        Figure title (e.g. "LSTM — sEMG — tensor_fascia_lata").
    stem : str
        Output filename stem (without extension).
    output_dir : str
        Directory for saving figures.
    formats : list of str, optional
        File formats (default: ["png", "pdf", "svg"]).
    """
    if formats is None:
        formats = ["png", "pdf", "svg"]
    plt = _setup_matplotlib()
    from scipy.stats import gaussian_kde  # type: ignore

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    def _kde_plot(ax, errors, label, color, alpha=0.4):
        if len(errors) < 4:
            return
        errors = errors[np.isfinite(errors)]
        ax.hist(errors, bins=50, density=True, alpha=alpha, color=color, label=label)
        try:
            kde = gaussian_kde(errors)
            xs  = np.linspace(errors.min(), errors.max(), 400)
            ax.plot(xs, kde(xs), color=color, linewidth=2)
        except Exception:
            pass

    # ── Left: linear scale ───────────────────────────────────────────────────
    ax = axes[0]
    _kde_plot(ax, train_errors,       "Train (healthy)", PALETTE["normal"],  alpha=0.35)
    _kde_plot(ax, test_errors_normal, "Test (normal)",   PALETTE["normal"],  alpha=0.55)
    _kde_plot(ax, test_errors_anomaly,"Test (anomaly)",  PALETTE["anomaly"], alpha=0.55)

    # Threshold lines
    for method, result in threshold_results.items():
        ax.axvline(
            x=result.value,
            color=PALETTE.get(method, "gray"),
            linestyle="--",
            linewidth=1.5,
            label=f"{THRESHOLD_LABELS.get(method, method)} = {result.value:.4f}",
        )

    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.set_ylabel("Density")
    ax.set_title("Error Distributions")
    ax.legend(fontsize=9)

    # ── Right: log scale (reveals tail behaviour) ────────────────────────────
    ax2 = axes[1]
    _kde_plot(ax2, train_errors,       "Train (healthy)", PALETTE["normal"],  alpha=0.35)
    _kde_plot(ax2, test_errors_normal, "Test (normal)",   PALETTE["normal"],  alpha=0.55)
    _kde_plot(ax2, test_errors_anomaly,"Test (anomaly)",  PALETTE["anomaly"], alpha=0.55)

    for method, result in threshold_results.items():
        ax2.axvline(
            x=result.value,
            color=PALETTE.get(method, "gray"),
            linestyle="--",
            linewidth=1.5,
            label=THRESHOLD_LABELS.get(method, method),
        )

    ax2.set_yscale("log")
    ax2.set_xlabel("Reconstruction Error (MSE)")
    ax2.set_ylabel("Density (log scale)")
    ax2.set_title("Error Distributions (log scale)")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    _save_fig(fig, stem, output_dir, formats)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Task 5: Severity analysis plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_severity_curves(
    severity_df: pd.DataFrame,
    model_name: str,
    modality: str,
    output_dir: str,
    formats: List[str] = None,
) -> None:
    """
    Plot severity vs. reconstruction error, recall, F1, and threshold margin.

    Parameters
    ----------
    severity_df : pd.DataFrame
        Output of SeverityAnalyzer.analyze(). Required columns:
        severity_numeric, anomaly_type, mean_error, recall, f1,
        threshold_margin, threshold_method.
    model_name : str
        e.g. "LSTM"
    modality : str
        e.g. "sEMG"
    output_dir : str
        Directory for figure output.
    formats : list of str
        Output formats.
    """
    if formats is None:
        formats = ["png", "pdf", "svg"]
    plt = _setup_matplotlib()

    if severity_df.empty:
        logger.warning(f"[Plots] severity_df is empty for {model_name}/{modality} — skipping.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Severity Analysis — {model_name} / {modality}", fontsize=14, fontweight="bold")

    metrics = [
        ("mean_error",       "Mean Reconstruction Error", axes[0, 0]),
        ("recall",           "Recall",                    axes[0, 1]),
        ("f1",               "F1 Score",                  axes[1, 0]),
        ("threshold_margin", "Threshold Margin",          axes[1, 1]),
    ]

    anomaly_types = severity_df["anomaly_type"].unique()
    type_colors   = plt.cm.tab10(np.linspace(0, 0.8, len(anomaly_types)))

    for metric_col, metric_label, ax in metrics:
        if metric_col not in severity_df.columns:
            ax.text(0.5, 0.5, f"{metric_col}\nnot available", ha="center", va="center")
            continue

        for atype, color in zip(anomaly_types, type_colors):
            sub = severity_df[severity_df["anomaly_type"] == atype].sort_values("severity_numeric")
            if sub.empty:
                continue

            # Plot per threshold method if available
            if "threshold_method" in sub.columns:
                for method in sub["threshold_method"].unique():
                    m_sub = sub[sub["threshold_method"] == method]
                    ax.plot(
                        m_sub["severity_numeric"], m_sub[metric_col],
                        marker="o", linewidth=1.5,
                        linestyle="--" if method != "mean_std" else "-",
                        color=PALETTE.get(method, color),
                        label=f"{atype} ({THRESHOLD_LABELS.get(method, method)})",
                        alpha=0.85,
                    )
            else:
                ax.plot(
                    sub["severity_numeric"], sub[metric_col],
                    marker="o", linewidth=2, color=color,
                    label=atype,
                )

        ax.axhline(y=0.0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Severity Level")
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.legend(fontsize=8)
        ax.set_xticks([0.15, 0.35, 0.60])
        ax.set_xticklabels(["Mild\n(0.15)", "Moderate\n(0.35)", "Severe\n(0.60)"])

    plt.tight_layout()
    stem = f"severity_curves_{model_name}_{modality}".replace(" ", "_").replace("+", "")
    _save_fig(fig, stem, output_dir, formats)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Task 7: ROC, PR, DET curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(
    score_df: pd.DataFrame,
    output_dir: str,
    formats: List[str] = None,
) -> None:
    """
    Plot ROC curves for all models × modalities.

    Parameters
    ----------
    score_df : pd.DataFrame
        Raw score CSV with reconstruction_error, is_synthetic_anomaly,
        model_name, modality.
    """
    if formats is None:
        formats = ["png", "pdf", "svg"]
    plt = _setup_matplotlib()

    try:
        from sklearn.metrics import roc_curve, auc  # type: ignore
    except ImportError:
        logger.warning("[Plots] sklearn not available — skipping ROC curves.")
        return

    models    = sorted(score_df["model_name"].unique())
    modalities = sorted(score_df["modality"].unique())

    fig, axes = plt.subplots(
        1, len(modalities), figsize=(6 * len(modalities), 5), squeeze=False
    )

    for col_idx, modality in enumerate(modalities):
        ax = axes[0, col_idx]
        ax.set_title(f"ROC — {modality}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")

        for model_name in models:
            subset = score_df[
                (score_df["model_name"] == model_name) &
                (score_df["modality"]   == modality)
            ]
            if subset.empty or len(subset["is_synthetic_anomaly"].unique()) < 2:
                continue

            y_true  = subset["is_synthetic_anomaly"].values.astype(int)
            scores  = subset["reconstruction_error"].values.astype(float)

            fpr, tpr, _ = roc_curve(y_true, scores)
            roc_auc = auc(fpr, tpr)
            ax.plot(
                fpr, tpr,
                color=PALETTE.get(model_name, "gray"),
                linewidth=2,
                label=f"{model_name} (AUC={roc_auc:.3f})",
            )

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")

    plt.tight_layout()
    _save_fig(fig, "roc_curves", output_dir, formats)
    plt.close(fig)


def plot_pr_curves(
    score_df: pd.DataFrame,
    output_dir: str,
    formats: List[str] = None,
) -> None:
    """Plot Precision-Recall curves for all models × modalities."""
    if formats is None:
        formats = ["png", "pdf", "svg"]
    plt = _setup_matplotlib()

    try:
        from sklearn.metrics import precision_recall_curve, average_precision_score
    except ImportError:
        logger.warning("[Plots] sklearn not available — skipping PR curves.")
        return

    models     = sorted(score_df["model_name"].unique())
    modalities = sorted(score_df["modality"].unique())

    fig, axes = plt.subplots(
        1, len(modalities), figsize=(6 * len(modalities), 5), squeeze=False
    )

    for col_idx, modality in enumerate(modalities):
        ax = axes[0, col_idx]
        ax.set_title(f"Precision-Recall — {modality}")

        for model_name in models:
            subset = score_df[
                (score_df["model_name"] == model_name) &
                (score_df["modality"]   == modality)
            ]
            if subset.empty or len(subset["is_synthetic_anomaly"].unique()) < 2:
                continue

            y_true  = subset["is_synthetic_anomaly"].values.astype(int)
            scores  = subset["reconstruction_error"].values.astype(float)

            precision, recall, _ = precision_recall_curve(y_true, scores)
            ap = average_precision_score(y_true, scores)
            ax.plot(
                recall, precision,
                color=PALETTE.get(model_name, "gray"),
                linewidth=2,
                label=f"{model_name} (AP={ap:.3f})",
            )

        # Baseline (random)
        pos_rate = score_df[score_df["modality"] == modality]["is_synthetic_anomaly"].mean()
        ax.axhline(y=pos_rate, color="gray", linestyle="--", linewidth=0.8, label="Random")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(loc="upper right")

    plt.tight_layout()
    _save_fig(fig, "pr_curves", output_dir, formats)
    plt.close(fig)


def plot_det_curves(
    score_df: pd.DataFrame,
    output_dir: str,
    formats: List[str] = None,
) -> None:
    """Plot Detection Error Tradeoff (DET) curves."""
    if formats is None:
        formats = ["png", "pdf", "svg"]
    plt = _setup_matplotlib()

    try:
        from sklearn.metrics import det_curve  # type: ignore
    except ImportError:
        logger.warning("[Plots] sklearn.metrics.det_curve not available (sklearn < 0.24) — skipping DET curves.")
        return

    models     = sorted(score_df["model_name"].unique())
    modalities = sorted(score_df["modality"].unique())

    fig, axes = plt.subplots(
        1, len(modalities), figsize=(6 * len(modalities), 5), squeeze=False
    )

    for col_idx, modality in enumerate(modalities):
        ax = axes[0, col_idx]
        ax.set_title(f"DET — {modality}")

        for model_name in models:
            subset = score_df[
                (score_df["model_name"] == model_name) &
                (score_df["modality"]   == modality)
            ]
            if subset.empty or len(subset["is_synthetic_anomaly"].unique()) < 2:
                continue

            y_true = subset["is_synthetic_anomaly"].values.astype(int)
            scores = subset["reconstruction_error"].values.astype(float)

            fpr, fnr, _ = det_curve(y_true, scores)
            ax.plot(
                fpr, fnr,
                color=PALETTE.get(model_name, "gray"),
                linewidth=2,
                label=model_name,
            )

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("False Negative Rate")
        ax.legend()

    plt.tight_layout()
    _save_fig(fig, "det_curves", output_dir, formats)
    plt.close(fig)


def plot_threshold_performance(
    metrics_df: pd.DataFrame,
    output_dir: str,
    formats: List[str] = None,
) -> None:
    """
    Plot performance metrics vs. threshold method as a grouped bar chart.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Output of evaluate_all_thresholds() or aggregate_by_modality().
    """
    if formats is None:
        formats = ["png", "pdf", "svg"]
    plt = _setup_matplotlib()

    if metrics_df.empty or "threshold_method" not in metrics_df.columns:
        logger.warning("[Plots] metrics_df empty or missing threshold_method — skipping.")
        return

    # Use ALL_CHANNELS rows if available
    if "ALL_CHANNELS" in metrics_df.get("channel_name", pd.Series()).values:
        plot_df = metrics_df[metrics_df["channel_name"] == "ALL_CHANNELS"].copy()
    else:
        plot_df = metrics_df.copy()

    models    = sorted(plot_df["model_name"].unique())
    methods   = sorted(plot_df["threshold_method"].unique())
    modalities = sorted(plot_df["modality"].unique()) if "modality" in plot_df.columns else ["all"]

    metric_cols = ["precision", "recall", "f1", "accuracy"]
    available   = [c for c in metric_cols if c in plot_df.columns]

    n_metrics = len(available)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5), squeeze=False)

    x = np.arange(len(methods))
    bar_width = 0.8 / max(len(models), 1)

    for m_idx, metric in enumerate(available):
        ax = axes[0, m_idx]
        ax.set_title(metric.capitalize())

        for model_idx, model_name in enumerate(models):
            vals = []
            for method in methods:
                row = plot_df[
                    (plot_df["model_name"]       == model_name) &
                    (plot_df["threshold_method"] == method)
                ]
                vals.append(row[metric].mean() if not row.empty else 0.0)

            offset = (model_idx - len(models) / 2 + 0.5) * bar_width
            ax.bar(
                x + offset, vals, bar_width,
                label=model_name,
                color=PALETTE.get(model_name, None),
                alpha=0.85,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [THRESHOLD_LABELS.get(m, m) for m in methods],
            rotation=15,
        )
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.legend(fontsize=9)

    fig.suptitle("Performance vs. Threshold Method", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, "threshold_performance", output_dir, formats)
    plt.close(fig)


def plot_robustness(
    metrics_df: pd.DataFrame,
    output_dir: str,
    formats: List[str] = None,
) -> None:
    """
    Plot F1/Recall vs. threshold method for each movement (robustness analysis).

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Must contain: model_name, threshold_method, movement, recall, f1.
    """
    if formats is None:
        formats = ["png", "pdf", "svg"]
    plt = _setup_matplotlib()

    if "movement" not in metrics_df.columns or metrics_df.empty:
        logger.warning("[Plots] 'movement' column not in metrics_df — skipping robustness plot.")
        return

    movements = sorted(metrics_df["movement"].unique())
    models    = sorted(metrics_df["model_name"].unique())
    methods   = sorted(metrics_df["threshold_method"].unique()) if "threshold_method" in metrics_df.columns else []

    fig, axes = plt.subplots(
        1, len(models), figsize=(7 * len(models), 5), squeeze=False
    )

    for model_idx, model_name in enumerate(models):
        ax = axes[0, model_idx]
        ax.set_title(f"Robustness — {model_name}")

        for method in methods:
            recalls = []
            for mov in movements:
                row = metrics_df[
                    (metrics_df["model_name"]       == model_name) &
                    (metrics_df["threshold_method"] == method) &
                    (metrics_df["movement"]          == mov)
                ]
                recalls.append(row["recall"].mean() if not row.empty else np.nan)

            ax.plot(
                movements, recalls,
                marker="o", linewidth=2,
                color=PALETTE.get(method, None),
                label=THRESHOLD_LABELS.get(method, method),
            )

        ax.set_xlabel("Movement")
        ax.set_ylabel("Recall")
        ax.set_xticklabels(movements, rotation=45, ha="right")
        ax.legend()

    plt.tight_layout()
    _save_fig(fig, "robustness", output_dir, formats)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def plot_all(
    score_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    severity_df: Optional[pd.DataFrame],
    train_errors_map: Dict[str, Dict[str, np.ndarray]],
    threshold_results_map: Dict[str, Dict[str, Dict[str, Any]]],
    output_dir: str,
    formats: List[str] = None,
) -> None:
    """
    Generate all plots in one call.

    Parameters
    ----------
    score_df : pd.DataFrame
        Raw score CSV data.
    metrics_df : pd.DataFrame
        Output of evaluate_all_thresholds().
    severity_df : pd.DataFrame or None
        Output of SeverityAnalyzer.analyze(), or None if not available.
    train_errors_map : dict
        {model: {channel: np.ndarray}}
    threshold_results_map : dict
        {model: {channel: {method: ThresholdResult}}}
    output_dir : str
        Root output directory. Figures go into output_dir/figures/.
    formats : list of str
        Figure file formats.
    """
    if formats is None:
        formats = ["png", "pdf", "svg"]

    fig_dir = os.path.join(output_dir, "figures")

    logger.info("[Plots] Generating ROC curves …")
    plot_roc_curves(score_df, fig_dir, formats)

    logger.info("[Plots] Generating PR curves …")
    plot_pr_curves(score_df, fig_dir, formats)

    logger.info("[Plots] Generating DET curves …")
    plot_det_curves(score_df, fig_dir, formats)

    logger.info("[Plots] Generating threshold performance bars …")
    plot_threshold_performance(metrics_df, fig_dir, formats)

    if "movement" in metrics_df.columns:
        logger.info("[Plots] Generating robustness plots …")
        plot_robustness(metrics_df, fig_dir, formats)

    # Error distributions per model × channel
    logger.info("[Plots] Generating error distribution plots …")
    for model_name, ch_errors_map in train_errors_map.items():
        for channel_name, train_errors in ch_errors_map.items():
            ch_df = score_df[
                (score_df["model_name"]  == model_name) &
                (score_df["channel_name"] == channel_name)
            ]
            if ch_df.empty:
                continue

            normal_err  = ch_df[ch_df["is_synthetic_anomaly"] == 0]["reconstruction_error"].values
            anomaly_err = ch_df[ch_df["is_synthetic_anomaly"] == 1]["reconstruction_error"].values

            th_results = threshold_results_map.get(model_name, {}).get(channel_name, {})
            stem = (
                f"error_dist_{model_name}_{channel_name}"
                .replace(" ", "_").replace("+", "")
            )
            plot_error_distribution(
                train_errors=train_errors,
                test_errors_normal=normal_err,
                test_errors_anomaly=anomaly_err,
                threshold_results=th_results,
                title=f"{model_name} — {channel_name}",
                stem=stem,
                output_dir=os.path.join(fig_dir, "error_distributions"),
                formats=formats,
            )

    # Severity curves
    if severity_df is not None and not severity_df.empty:
        logger.info("[Plots] Generating severity curves …")
        for model_name in severity_df["model_name"].unique() if "model_name" in severity_df.columns else []:
            for modality in severity_df["modality"].unique() if "modality" in severity_df.columns else []:
                sub = severity_df[
                    (severity_df["model_name"] == model_name) &
                    (severity_df["modality"]   == modality)
                ]
                if not sub.empty:
                    plot_severity_curves(sub, model_name, modality, fig_dir, formats)

    logger.info(f"[Plots] All figures saved to {fig_dir}")
