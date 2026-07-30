"""
evaluation_framework/fusion_analysis.py

OR-fusion ablation and modality contribution analysis (Task 6).

For every anomalous window, determines:
    - Which channels fired (predicted_label == 1)
    - Which modality fired exclusively (sEMG-only, KinKin-only, or both)
    - Whether OR/AND/MAJORITY fusion correctly detected the anomaly

Computes:
    - Percentage of anomalies detected exclusively by each modality
    - Incremental recall gain contributed by each modality to OR fusion
    - Co-occurrence matrix: which pairs of modalities fire together
    - Channel activation frequencies

Generates:
    - Venn diagram (2-set: sEMG vs KinKin)
    - Stacked bar chart: exclusive vs. overlapping detections
    - Co-occurrence heatmap
    - Channel activation frequency bar chart

Usage:
    from evaluation_framework.fusion_analysis import FusionAnalyzer

    analyzer = FusionAnalyzer()
    report   = analyzer.analyze(fused_df, raw_df)
    report.incremental_recall_df  # per-modality recall contribution
    report.exclusivity_df          # exclusive detection rates
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODALITY_SEMG   = "sEMG"
MODALITY_KINKIN = "Kinematics+Kinetics"


# ─────────────────────────────────────────────────────────────────────────────
# Report container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FusionReport:
    """Container for fusion analysis results."""

    # Per-window firing analysis
    window_analysis_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Exclusivity: fraction of anomalies detected only by each modality
    exclusivity_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Incremental recall gain per modality
    incremental_recall_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Channel activation frequencies
    channel_activation_df: pd.DataFrame = field(default_factory=pd.DataFrame)


# ─────────────────────────────────────────────────────────────────────────────
# Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class FusionAnalyzer:
    """
    Ablation study and explanatory analysis for OR-fusion.

    Explains quantitatively why OR fusion improves recall by measuring
    the exclusive and overlapping detection contributions of each modality.
    """

    def analyze(
        self,
        fused_df: pd.DataFrame,
        raw_df: pd.DataFrame,
    ) -> FusionReport:
        """
        Run the full fusion analysis.

        Parameters
        ----------
        fused_df : pd.DataFrame
            Output of late_fusion._apply_fusion() — one row per window with
            columns: semg_vote, kinkin_vote, ground_truth, fused_OR, fused_AND,
            fused_MAJORITY, model_name, subject_id, movement, window_id.
        raw_df : pd.DataFrame
            Raw score CSV with per-channel predicted_label, modality,
            channel_name, model_name, subject_id, movement, window_id.

        Returns
        -------
        FusionReport
        """
        report = FusionReport()

        if fused_df.empty:
            logger.warning("[FusionAnalyzer] fused_df is empty — no analysis possible.")
            return report

        # ── 1. Per-window firing pattern ──────────────────────────────────────
        report.window_analysis_df = self._analyze_window_firing(fused_df)

        # ── 2. Exclusivity rates ──────────────────────────────────────────────
        report.exclusivity_df = self._compute_exclusivity(report.window_analysis_df)

        # ── 3. Incremental recall gain ────────────────────────────────────────
        report.incremental_recall_df = self._compute_incremental_recall(fused_df)

        # ── 4. Channel activation frequencies ────────────────────────────────
        if not raw_df.empty:
            report.channel_activation_df = self._compute_channel_activation(raw_df, fused_df)

        return report

    def _analyze_window_firing(self, fused_df: pd.DataFrame) -> pd.DataFrame:
        """
        For each anomalous window, classify firing pattern:
            - only_semg    : semg_vote=1, kinkin_vote=0
            - only_kinkin  : semg_vote=0, kinkin_vote=1
            - both         : semg_vote=1, kinkin_vote=1
            - neither      : semg_vote=0, kinkin_vote=0 (missed by OR)
        """
        df = fused_df.copy()

        # Only anomalous windows (ground_truth == 1)
        anom = df[df["ground_truth"] == 1].copy()

        anom["fire_pattern"] = "neither"
        anom.loc[
            (anom["semg_vote"] == 1) & (anom["kinkin_vote"] == 0), "fire_pattern"
        ] = "only_semg"
        anom.loc[
            (anom["semg_vote"] == 0) & (anom["kinkin_vote"] == 1), "fire_pattern"
        ] = "only_kinkin"
        anom.loc[
            (anom["semg_vote"] == 1) & (anom["kinkin_vote"] == 1), "fire_pattern"
        ] = "both"

        return anom

    def _compute_exclusivity(self, window_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute what fraction of detected anomalies were detected exclusively
        by each modality (vs. jointly).
        """
        if window_df.empty:
            return pd.DataFrame()

        records: List[Dict[str, Any]] = []

        for model_name in window_df["model_name"].unique() if "model_name" in window_df.columns else [None]:
            if model_name is not None:
                sub = window_df[window_df["model_name"] == model_name]
            else:
                sub = window_df

            total = len(sub)
            if total == 0:
                continue

            counts = sub["fire_pattern"].value_counts()
            only_semg   = int(counts.get("only_semg",   0))
            only_kinkin = int(counts.get("only_kinkin", 0))
            both        = int(counts.get("both",        0))
            neither     = int(counts.get("neither",     0))

            detected_by_or = total - neither
            detected_by_and = both

            records.append({
                "model_name":            model_name,
                "total_anomaly_windows": total,
                "detected_by_or":        detected_by_or,
                "detected_by_and":       detected_by_and,
                "only_semg":             only_semg,
                "only_kinkin":           only_kinkin,
                "both":                  both,
                "neither":               neither,
                "pct_only_semg":         round(only_semg   / total * 100, 2),
                "pct_only_kinkin":       round(only_kinkin / total * 100, 2),
                "pct_both":              round(both        / total * 100, 2),
                "pct_neither":           round(neither     / total * 100, 2),
                "or_recall":             round(detected_by_or  / total, 4),
                "and_recall":            round(detected_by_and / total, 4),
            })

        return pd.DataFrame(records)

    def _compute_incremental_recall(self, fused_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the incremental recall gain contributed by each modality.

        Base recall    = KinKin-only (no sEMG)
        sEMG gain      = OR recall - KinKin recall
        Similarly for KinKin gain = OR recall - sEMG recall.

        Quantifies: "by how much does adding sEMG to KinKin improve recall?"
        """
        if fused_df.empty:
            return pd.DataFrame()

        records: List[Dict[str, Any]] = []

        for model_name in fused_df["model_name"].unique() if "model_name" in fused_df.columns else [None]:
            if model_name is not None:
                sub = fused_df[fused_df["model_name"] == model_name].copy()
            else:
                sub = fused_df.copy()

            y_true = sub["ground_truth"].values.astype(int)
            if len(np.unique(y_true)) < 2:
                continue

            def recall(y_t, y_p):
                tp = int(((y_t == 1) & (y_p == 1)).sum())
                fn = int(((y_t == 1) & (y_p == 0)).sum())
                return tp / (tp + fn) if (tp + fn) > 0 else 0.0

            # Single-modality recalls
            semg_recall   = recall(y_true, sub["semg_vote"].values.astype(int))
            kinkin_recall = recall(y_true, sub["kinkin_vote"].values.astype(int))
            or_recall     = recall(y_true, sub["fused_OR"].values.astype(int)) if "fused_OR" in sub.columns else float("nan")

            records.append({
                "model_name":           model_name,
                "semg_recall":          round(semg_recall,    4),
                "kinkin_recall":        round(kinkin_recall,  4),
                "or_recall":            round(or_recall,      4) if not np.isnan(or_recall) else float("nan"),
                "semg_incremental_gain":   round(or_recall - kinkin_recall, 4) if not np.isnan(or_recall) else float("nan"),
                "kinkin_incremental_gain": round(or_recall - semg_recall,   4) if not np.isnan(or_recall) else float("nan"),
                "absolute_gain_vs_best": round(
                    or_recall - max(semg_recall, kinkin_recall), 4
                ) if not np.isnan(or_recall) else float("nan"),
                "n_windows": len(sub),
            })

        return pd.DataFrame(records)

    def _compute_channel_activation(
        self,
        raw_df: pd.DataFrame,
        fused_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute how frequently each channel fires (predicted_label==1)
        on anomalous windows, broken down by modality.
        """
        if raw_df.empty or "channel_name" not in raw_df.columns:
            return pd.DataFrame()

        # Filter to anomalous windows
        anom_raw = raw_df[raw_df["is_synthetic_anomaly"] == 1]
        if anom_raw.empty:
            return pd.DataFrame()

        records: List[Dict[str, Any]] = []

        for model_name in anom_raw["model_name"].unique():
            for modality in anom_raw["modality"].unique():
                sub = anom_raw[
                    (anom_raw["model_name"] == model_name) &
                    (anom_raw["modality"]   == modality)
                ]
                for ch in sorted(sub["channel_name"].unique()):
                    ch_rows = sub[sub["channel_name"] == ch]
                    n_total = len(ch_rows)
                    n_fired = int(ch_rows["predicted_label"].sum())
                    records.append({
                        "model_name":   model_name,
                        "modality":     modality,
                        "channel_name": ch,
                        "n_windows":    n_total,
                        "n_fired":      n_fired,
                        "activation_rate": round(n_fired / n_total, 4) if n_total > 0 else 0.0,
                    })

        return pd.DataFrame(records)

    def export(self, report: FusionReport, output_dir: str) -> Dict[str, str]:
        """Export all fusion analysis results to CSV."""
        os.makedirs(output_dir, exist_ok=True)
        paths: Dict[str, str] = {}

        for attr, fname in [
            ("window_analysis_df",    "fusion_window_analysis.csv"),
            ("exclusivity_df",        "fusion_exclusivity.csv"),
            ("incremental_recall_df", "fusion_incremental_recall.csv"),
            ("channel_activation_df", "fusion_channel_activation.csv"),
        ]:
            df = getattr(report, attr)
            if not df.empty:
                p = os.path.join(output_dir, fname)
                df.to_csv(p, index=False)
                paths[fname] = p
                logger.info(f"[Fusion] {fname} → {p}")

        return paths

    def plot_all(
        self,
        report: FusionReport,
        output_dir: str,
        formats: List[str] = None,
    ) -> None:
        """Generate all fusion analysis figures."""
        if formats is None:
            formats = ["png", "pdf", "svg"]

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = os.path.join(output_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)

        # ── Venn-style bar chart (no matplotlib_venn dependency) ─────────────
        if not report.exclusivity_df.empty:
            self._plot_exclusivity_bars(report.exclusivity_df, fig_dir, formats, plt)

        # ── Incremental recall gain ───────────────────────────────────────────
        if not report.incremental_recall_df.empty:
            self._plot_incremental_recall(report.incremental_recall_df, fig_dir, formats, plt)

        # ── Channel activation frequencies ───────────────────────────────────
        if not report.channel_activation_df.empty:
            self._plot_channel_activation(report.channel_activation_df, fig_dir, formats, plt)

    def _plot_exclusivity_bars(self, df, fig_dir, formats, plt):
        """Stacked bar: exclusive vs. joint detections per model."""
        models = sorted(df["model_name"].dropna().unique())
        fig, ax = plt.subplots(figsize=(8, 5))

        x = np.arange(len(models))
        bar_w = 0.55

        cols  = ["pct_only_semg", "pct_only_kinkin", "pct_both", "pct_neither"]
        labels = ["sEMG only", "KinKin only", "Both", "Neither (missed)"]
        colors = ["#4C9BE8", "#E05F5F", "#6A4C93", "#AAAAAA"]

        bottoms = np.zeros(len(models))
        for col, label, color in zip(cols, labels, colors):
            vals = [df[df["model_name"] == m][col].values[0] if len(df[df["model_name"] == m]) > 0 else 0 for m in models]
            ax.bar(x, vals, bar_w, bottom=bottoms, label=label, color=color, alpha=0.85)
            bottoms += np.array(vals)

        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylabel("% of Anomaly Windows")
        ax.set_title("Modality-Exclusive Detection Rates")
        ax.legend(loc="upper right")
        plt.tight_layout()

        for fmt in formats:
            fig.savefig(os.path.join(fig_dir, f"fusion_exclusivity.{fmt}"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[Fusion] Saved exclusivity bar chart → {fig_dir}")

    def _plot_incremental_recall(self, df, fig_dir, formats, plt):
        """Grouped bar: single-modality recall vs OR recall."""
        models = sorted(df["model_name"].dropna().unique())
        fig, ax = plt.subplots(figsize=(8, 5))

        x = np.arange(len(models))
        w = 0.25

        for i, (col, label, color) in enumerate([
            ("semg_recall",   "sEMG",    "#4C9BE8"),
            ("kinkin_recall", "KinKin",  "#E05F5F"),
            ("or_recall",     "OR Fused","#6A4C93"),
        ]):
            vals = [df[df["model_name"] == m][col].values[0] if len(df[df["model_name"] == m]) > 0 else 0 for m in models]
            ax.bar(x + i * w, vals, w, label=label, color=color, alpha=0.85)

        ax.set_xticks(x + w)
        ax.set_xticklabels(models)
        ax.set_ylabel("Recall")
        ax.set_ylim(0, 1.05)
        ax.set_title("Incremental Recall Gain — OR Fusion vs. Single Modality")
        ax.legend()
        plt.tight_layout()

        for fmt in formats:
            fig.savefig(os.path.join(fig_dir, f"fusion_incremental_recall.{fmt}"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[Fusion] Saved incremental recall chart → {fig_dir}")

    def _plot_channel_activation(self, df, fig_dir, formats, plt):
        """Horizontal bar: channel activation rates on anomalous windows."""
        for model_name in df["model_name"].unique():
            model_df = df[df["model_name"] == model_name].sort_values(
                ["modality", "activation_rate"], ascending=[True, False]
            )
            if model_df.empty:
                continue

            fig, ax = plt.subplots(figsize=(10, max(4, len(model_df) * 0.35)))
            labels  = model_df["modality"] + " — " + model_df["channel_name"]
            rates   = model_df["activation_rate"].values
            colors  = ["#4C9BE8" if m == MODALITY_SEMG else "#E05F5F"
                       for m in model_df["modality"]]

            bars = ax.barh(range(len(rates)), rates, color=colors, alpha=0.85)
            ax.set_yticks(range(len(rates)))
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_xlabel("Activation Rate (fraction of anomaly windows flagged)")
            ax.set_title(f"Channel Activation Frequencies — {model_name}")
            ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=0.8)

            # Legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(color="#4C9BE8", label="sEMG"),
                Patch(color="#E05F5F", label="Kinematics+Kinetics"),
            ]
            ax.legend(handles=legend_elements, loc="lower right")

            plt.tight_layout()
            for fmt in formats:
                path = os.path.join(fig_dir, f"channel_activation_{model_name}.{fmt}")
                fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"[Fusion] Saved channel activation chart → {fig_dir}")
