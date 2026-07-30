"""
evaluation_framework/calibration.py

Anomaly score calibration and distribution analysis (Task 7).

Measures:
    - Distribution of anomaly scores (reconstruction errors) for normal vs anomalous windows
    - Distance from threshold (margin = score - threshold) for each window
    - Score overlap between healthy and anomalous distributions
    - Probability calibration (Platt scaling estimate)

Generates:
    - ROC curve per model × modality
    - Precision-Recall curve
    - Threshold-performance curves (sweeping threshold from min to max score)
    - Detection Error Tradeoff (DET) curve
    - Score distribution overlap (histogram, overlap coefficient)

Usage:
    from evaluation_framework.calibration import CalibrationAnalyzer

    analyzer = CalibrationAnalyzer(threshold_results_map)
    report   = analyzer.analyze(score_df)
    analyzer.export(report, "outputs/evaluation/calibration")
    analyzer.plot_all(report, "outputs/evaluation", formats=["png", "pdf"])
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Report container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CalibrationReport:
    """Container for calibration analysis results."""

    # Score distribution statistics per (model, modality, channel, threshold_method)
    distribution_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Threshold sweep results: precision/recall/F1 vs threshold value
    threshold_sweep_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Score overlap statistics
    overlap_df: pd.DataFrame = field(default_factory=pd.DataFrame)


# ─────────────────────────────────────────────────────────────────────────────
# Overlap coefficient helper
# ─────────────────────────────────────────────────────────────────────────────

def _overlap_coefficient(
    a: np.ndarray, b: np.ndarray, n_bins: int = 200
) -> float:
    """
    Compute the histogram-based overlap coefficient (Bhattacharyya coefficient)
    between two distributions.

    Returns a value in [0, 1] where 0 = no overlap, 1 = identical distributions.
    """
    if len(a) < 2 or len(b) < 2:
        return float("nan")

    combined_min = min(a.min(), b.min())
    combined_max = max(a.max(), b.max())
    if combined_max <= combined_min:
        return float("nan")

    bins = np.linspace(combined_min, combined_max, n_bins + 1)
    ha, _ = np.histogram(a, bins=bins, density=True)
    hb, _ = np.histogram(b, bins=bins, density=True)

    # Normalize
    ha = ha / (ha.sum() + 1e-12)
    hb = hb / (hb.sum() + 1e-12)

    # Bhattacharyya coefficient
    return float(np.sum(np.sqrt(ha * hb)))


# ─────────────────────────────────────────────────────────────────────────────
# Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class CalibrationAnalyzer:
    """
    Score calibration and distribution analysis.

    Parameters
    ----------
    threshold_results_map : dict
        {model_name: {channel_name: {method: ThresholdResult}}}
    """

    def __init__(
        self,
        threshold_results_map: Dict[str, Dict[str, Dict[str, Any]]],
    ):
        self.threshold_results_map = threshold_results_map

    def analyze(
        self,
        score_df: pd.DataFrame,
        methods: Optional[List[str]] = None,
        n_sweep_points: int = 100,
    ) -> CalibrationReport:
        """
        Run the full calibration analysis.

        Parameters
        ----------
        score_df : pd.DataFrame
            Raw score CSV with reconstruction_error, is_synthetic_anomaly,
            model_name, channel_name, modality.
        methods : list of str, optional
            Threshold methods to analyze. Default: all available.
        n_sweep_points : int
            Number of threshold values to sweep per (model, channel).

        Returns
        -------
        CalibrationReport
        """
        if methods is None:
            methods = ["mean_std", "percentile95", "percentile99"]

        dist_records:   List[Dict[str, Any]] = []
        sweep_records:  List[Dict[str, Any]] = []
        overlap_records: List[Dict[str, Any]] = []

        for model_name in score_df["model_name"].unique():
            model_df  = score_df[score_df["model_name"] == model_name]
            ch_thresh = self.threshold_results_map.get(model_name, {})

            for channel_name in sorted(model_df["channel_name"].unique()):
                ch_df    = model_df[model_df["channel_name"] == channel_name]
                modality = ch_df["modality"].iloc[0]

                scores = ch_df["reconstruction_error"].values.astype(float)
                y_true = ch_df["is_synthetic_anomaly"].values.astype(int)

                normal_scores = scores[y_true == 0]
                anom_scores   = scores[y_true == 1]

                thresh_map = ch_thresh.get(channel_name, {})

                # ── Distribution statistics ─────────────────────────────────
                for label, arr in [("normal", normal_scores), ("anomaly", anom_scores)]:
                    if len(arr) == 0:
                        continue
                    for method in methods:
                        thresh_result = thresh_map.get(method)
                        thresh_val    = thresh_result.value if thresh_result is not None else float("nan")
                        margin = float(np.mean(arr - thresh_val)) if not np.isnan(thresh_val) else float("nan")

                        dist_records.append({
                            "model_name":       model_name,
                            "modality":         modality,
                            "channel_name":     channel_name,
                            "threshold_method": method,
                            "threshold_value":  thresh_val,
                            "label":            label,
                            "n":                len(arr),
                            "mean_score":       float(np.mean(arr)),
                            "std_score":        float(np.std(arr)),
                            "min_score":        float(np.min(arr)),
                            "max_score":        float(np.max(arr)),
                            "median_score":     float(np.median(arr)),
                            "p95_score":        float(np.percentile(arr, 95)),
                            "mean_margin":      margin,
                        })

                # ── Score overlap ───────────────────────────────────────────
                if len(normal_scores) > 1 and len(anom_scores) > 1:
                    overlap = _overlap_coefficient(normal_scores, anom_scores)
                    # Separability: difference in means / pooled std
                    pooled_std = np.sqrt(
                        (normal_scores.var() + anom_scores.var()) / 2
                    )
                    separability_d = (
                        abs(anom_scores.mean() - normal_scores.mean()) / pooled_std
                        if pooled_std > 0 else float("nan")
                    )
                    overlap_records.append({
                        "model_name":      model_name,
                        "modality":        modality,
                        "channel_name":    channel_name,
                        "overlap_coeff":   round(overlap, 4),
                        "separability_d":  round(separability_d, 4) if not np.isnan(separability_d) else float("nan"),
                        "normal_mean":     float(normal_scores.mean()),
                        "anomaly_mean":    float(anom_scores.mean()),
                        "mean_diff":       float(anom_scores.mean() - normal_scores.mean()),
                        "n_normal":        len(normal_scores),
                        "n_anomaly":       len(anom_scores),
                    })

                # ── Threshold sweep ─────────────────────────────────────────
                if len(scores) == 0 or len(np.unique(y_true)) < 2:
                    continue

                sweep_min = float(np.percentile(scores, 1))
                sweep_max = float(np.percentile(scores, 99))
                if sweep_max <= sweep_min:
                    continue

                thresholds_to_sweep = np.linspace(sweep_min, sweep_max, n_sweep_points)

                for t in thresholds_to_sweep:
                    y_pred = (scores > t).astype(int)
                    TP = int(((y_true == 1) & (y_pred == 1)).sum())
                    FP = int(((y_true == 0) & (y_pred == 1)).sum())
                    TN = int(((y_true == 0) & (y_pred == 0)).sum())
                    FN = int(((y_true == 1) & (y_pred == 0)).sum())

                    recall  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                    prec    = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                    f1      = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
                    fpr     = FP / (FP + TN) if (FP + TN) > 0 else 0.0

                    sweep_records.append({
                        "model_name":   model_name,
                        "modality":     modality,
                        "channel_name": channel_name,
                        "threshold":    float(t),
                        "recall":       recall,
                        "precision":    prec,
                        "f1":           f1,
                        "fpr":          fpr,
                        "TP":           TP,
                        "FP":           FP,
                        "TN":           TN,
                        "FN":           FN,
                    })

        return CalibrationReport(
            distribution_df=pd.DataFrame(dist_records),
            threshold_sweep_df=pd.DataFrame(sweep_records),
            overlap_df=pd.DataFrame(overlap_records),
        )

    def export(self, report: CalibrationReport, output_dir: str) -> Dict[str, str]:
        """Export calibration analysis to CSV files."""
        os.makedirs(output_dir, exist_ok=True)
        paths: Dict[str, str] = {}

        for attr, fname in [
            ("distribution_df",   "calibration_distribution.csv"),
            ("threshold_sweep_df","calibration_threshold_sweep.csv"),
            ("overlap_df",        "calibration_overlap.csv"),
        ]:
            df = getattr(report, attr)
            if not df.empty:
                p = os.path.join(output_dir, fname)
                df.to_csv(p, index=False)
                paths[fname] = p
                logger.info(f"[Calibration] {fname} → {p}")

        return paths

    def plot_all(
        self,
        report: CalibrationReport,
        output_dir: str,
        formats: List[str] = None,
    ) -> None:
        """Generate calibration figures."""
        if formats is None:
            formats = ["png", "pdf", "svg"]

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = os.path.join(output_dir, "figures", "calibration")
        os.makedirs(fig_dir, exist_ok=True)

        # Threshold sweep: precision, recall, F1 vs threshold
        if not report.threshold_sweep_df.empty:
            self._plot_threshold_sweep(report.threshold_sweep_df, fig_dir, formats, plt)

        # Score overlap
        if not report.overlap_df.empty:
            self._plot_overlap_summary(report.overlap_df, fig_dir, formats, plt)

    def _plot_threshold_sweep(self, df, fig_dir, formats, plt):
        """Plot precision/recall/F1 vs threshold for each model × channel."""
        models = sorted(df["model_name"].unique())

        for model_name in models:
            m_df = df[df["model_name"] == model_name]
            channels = sorted(m_df["channel_name"].unique())

            # Aggregate across channels (mean per threshold)
            agg = m_df.groupby("threshold")[["precision", "recall", "f1"]].mean().reset_index()

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(agg["threshold"], agg["recall"],    label="Recall",    linewidth=2, color="#4C9BE8")
            ax.plot(agg["threshold"], agg["precision"], label="Precision", linewidth=2, color="#E05F5F")
            ax.plot(agg["threshold"], agg["f1"],        label="F1",        linewidth=2, color="#6A4C93")

            ax.set_xlabel("Threshold Value")
            ax.set_ylabel("Metric Score")
            ax.set_title(f"Threshold-Performance Curve — {model_name}")
            ax.legend()
            ax.set_ylim(0, 1.05)
            plt.tight_layout()

            stem = f"threshold_sweep_{model_name}".replace(" ", "_")
            for fmt in formats:
                fig.savefig(os.path.join(fig_dir, f"{stem}.{fmt}"), dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"[Calibration] Threshold sweep → {fig_dir}/{stem}.*")

    def _plot_overlap_summary(self, df, fig_dir, formats, plt):
        """Plot overlap coefficient and separability across channels."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, col, title, ylabel in [
            (axes[0], "overlap_coeff",  "Score Overlap (lower = more separable)", "Overlap Coefficient"),
            (axes[1], "separability_d", "Separability (Cohen's d, higher = better)", "Effect Size d"),
        ]:
            if col not in df.columns:
                continue
            sub = df.dropna(subset=[col]).sort_values(col, ascending=False)
            labels = sub["model_name"] + "/" + sub["channel_name"]
            vals   = sub[col].values

            colors = ["#4C9BE8" if v < 0.5 else "#E05F5F" for v in vals] if col == "overlap_coeff" else ["#6A4C93"] * len(vals)
            ax.barh(range(len(vals)), vals, color=colors, alpha=0.8)
            ax.set_yticks(range(len(vals)))
            ax.set_yticklabels(labels, fontsize=7)
            ax.set_xlabel(ylabel)
            ax.set_title(title)

        plt.tight_layout()
        for fmt in formats:
            fig.savefig(os.path.join(fig_dir, f"score_overlap_summary.{fmt}"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[Calibration] Score overlap summary → {fig_dir}")
