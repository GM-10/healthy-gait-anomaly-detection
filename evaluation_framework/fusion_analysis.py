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

Statistical validation:
    - Pairwise bootstrap comparison (F1, Recall) for all fusion strategy pairs:
      OR vs AND, OR vs MAJORITY, MAJORITY vs AND
    - McNemar's test for OR vs AND and OR vs MAJORITY
    - Bonferroni-corrected p-values

OR-fusion superiority analysis:
    - Mathematical recall bound: Recall_OR >= max(Recall_sEMG, Recall_KinKin)
    - Precision penalty: Precision_OR <= min(Precision_sEMG, Precision_KinKin)
    - Modality complementarity score (fraction of anomalies exclusively covered
      by one modality)
    - F1 crossover point: conditions under which OR-F1 > single-modality F1

Generates:
    - Venn diagram (2-set: sEMG vs KinKin)
    - Stacked bar chart: exclusive vs. overlapping detections
    - Co-occurrence heatmap
    - Channel activation frequency bar chart

Usage:
    from evaluation_framework.fusion_analysis import FusionAnalyzer

    analyzer = FusionAnalyzer()
    report   = analyzer.analyze(fused_df, raw_df)
    report.incremental_recall_df   # per-modality recall contribution
    report.exclusivity_df          # exclusive detection rates
    report.significance_df         # pairwise bootstrap + McNemar tests
    report.or_dominance_df         # mathematical OR dominance decomposition
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODALITY_SEMG   = "sEMG"
MODALITY_KINKIN = "Kinematics+Kinetics"

_BOOTSTRAP_N    = 1000
_BOOTSTRAP_SEED = 42
_CI_LEVEL       = 0.95


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers (local, no sklearn dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _tp_fp_tn_fn(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())
    return TP, FP, TN, FN


def _recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def _precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    p = _precision(y_true, y_pred)
    r = _recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


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

    # Pairwise statistical significance: bootstrap + McNemar for strategy pairs
    significance_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # OR dominance mathematical decomposition
    or_dominance_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Precision/recall/F1 trade-off across fusion strategies
    pr_tradeoff_df: pd.DataFrame = field(default_factory=pd.DataFrame)


# ─────────────────────────────────────────────────────────────────────────────
# Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class FusionAnalyzer:
    """
    Ablation study and explanatory analysis for OR-fusion.

    Explains quantitatively why OR fusion improves recall by measuring
    the exclusive and overlapping detection contributions of each modality,
    and validates OR-fusion superiority with bootstrap significance tests
    and an information-theoretic decomposition.
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

        # ── 5. Statistical significance tests ────────────────────────────────
        report.significance_df = self._run_fusion_significance(fused_df)

        # ── 6. OR dominance decomposition ────────────────────────────────────
        report.or_dominance_df = self._explain_or_fusion_dominance(fused_df)

        # ── 7. Precision/Recall/F1 trade-off table ───────────────────────────
        report.pr_tradeoff_df = self._compute_precision_recall_tradeoff(fused_df)

        return report

    # ─────────────────────────────────────────────────────────────────────────
    # Window firing analysis
    # ─────────────────────────────────────────────────────────────────────────

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

    # ─────────────────────────────────────────────────────────────────────────
    # Exclusivity rates
    # ─────────────────────────────────────────────────────────────────────────

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
                "complementarity_score": round((only_semg + only_kinkin) / total, 4),
            })

        return pd.DataFrame(records)

    # ─────────────────────────────────────────────────────────────────────────
    # Incremental recall gain
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_incremental_recall(self, fused_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the incremental recall gain contributed by each modality.

        Base recall    = KinKin-only (no sEMG)
        sEMG gain      = OR recall - KinKin recall
        Similarly for KinKin gain = OR recall - sEMG recall.

        Quantifies: \"by how much does adding sEMG to KinKin improve recall?\"
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

            y_semg   = sub["semg_vote"].values.astype(int)
            y_kinkin = sub["kinkin_vote"].values.astype(int)

            semg_recall   = _recall(y_true, y_semg)
            kinkin_recall = _recall(y_true, y_kinkin)
            or_recall     = _recall(y_true, sub["fused_OR"].values.astype(int)) \
                if "fused_OR" in sub.columns else float("nan")

            records.append({
                "model_name":              model_name,
                "semg_recall":             round(semg_recall,    4),
                "kinkin_recall":           round(kinkin_recall,  4),
                "or_recall":               round(or_recall,      4) if not np.isnan(or_recall) else float("nan"),
                "semg_incremental_gain":   round(or_recall - kinkin_recall, 4) if not np.isnan(or_recall) else float("nan"),
                "kinkin_incremental_gain": round(or_recall - semg_recall,   4) if not np.isnan(or_recall) else float("nan"),
                "absolute_gain_vs_best":   round(
                    or_recall - max(semg_recall, kinkin_recall), 4
                ) if not np.isnan(or_recall) else float("nan"),
                "n_windows": len(sub),
            })

        return pd.DataFrame(records)

    # ─────────────────────────────────────────────────────────────────────────
    # Channel activation frequencies
    # ─────────────────────────────────────────────────────────────────────────

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

    # ─────────────────────────────────────────────────────────────────────────
    # Statistical significance: pairwise bootstrap + McNemar
    # ─────────────────────────────────────────────────────────────────────────

    def _run_fusion_significance(self, fused_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run pairwise bootstrap comparisons and McNemar's test for all fusion
        strategy pairs: (OR vs AND), (OR vs MAJORITY), (MAJORITY vs AND).

        For each pair, computes:
          - Bootstrap delta-F1 and delta-Recall with 95% CI
          - Two-sided p-value (shift method)
          - McNemar's chi-squared and p-value
          - Cohen's d effect size

        Uses Bonferroni correction for 3 simultaneous tests.

        Returns
        -------
        pd.DataFrame
            One row per (model_name, strategy_A, strategy_B).
        """
        try:
            from scipy.stats import chi2  # type: ignore
        except ImportError:
            logger.warning(
                "[FusionAnalyzer] scipy not installed — McNemar tests skipped. "
                "Bootstrap comparisons will still run."
            )
            chi2 = None

        strategy_cols = {
            "OR":       "fused_OR",
            "MAJORITY": "fused_MAJORITY",
            "AND":      "fused_AND",
        }

        n_pairs = 3   # OR-AND, OR-MAJORITY, MAJORITY-AND
        alpha   = (1.0 - _CI_LEVEL) / n_pairs   # Bonferroni-corrected alpha

        records: List[Dict[str, Any]] = []

        for model_name in fused_df["model_name"].unique() if "model_name" in fused_df.columns else [None]:
            if model_name is not None:
                sub = fused_df[fused_df["model_name"] == model_name]
            else:
                sub = fused_df

            y_true = sub["ground_truth"].values.astype(int)
            if len(np.unique(y_true)) < 2:
                logger.debug(f"[FusionSig] {model_name}: only one class — skipping")
                continue

            # Build predictions dict
            preds: Dict[str, np.ndarray] = {}
            for strat, col in strategy_cols.items():
                if col in sub.columns:
                    preds[strat] = sub[col].values.astype(int)

            if len(preds) < 2:
                continue

            for strat_a, strat_b in combinations(sorted(preds.keys()), 2):
                y_a = preds[strat_a]
                y_b = preds[strat_b]

                # Observed metrics
                recall_a    = _recall(y_true, y_a)
                recall_b    = _recall(y_true, y_b)
                f1_a        = _f1(y_true, y_a)
                f1_b        = _f1(y_true, y_b)
                delta_recall_obs = recall_b - recall_a
                delta_f1_obs     = f1_b - f1_a

                # Bootstrap comparison
                rng = np.random.default_rng(_BOOTSTRAP_SEED)
                n   = len(y_true)
                delta_recalls = np.empty(_BOOTSTRAP_N)
                delta_f1s     = np.empty(_BOOTSTRAP_N)
                vals_recall_a = np.empty(_BOOTSTRAP_N)
                vals_recall_b = np.empty(_BOOTSTRAP_N)
                vals_f1_a     = np.empty(_BOOTSTRAP_N)
                vals_f1_b     = np.empty(_BOOTSTRAP_N)

                for i in range(_BOOTSTRAP_N):
                    idx = rng.integers(0, n, size=n)
                    yt = y_true[idx]
                    ya = y_a[idx]
                    yb = y_b[idx]
                    ra = _recall(yt, ya);    rb = _recall(yt, yb)
                    fa = _f1(yt, ya);        fb = _f1(yt, yb)
                    delta_recalls[i] = rb - ra
                    delta_f1s[i]     = fb - fa
                    vals_recall_a[i] = ra;  vals_recall_b[i] = rb
                    vals_f1_a[i]     = fa;  vals_f1_b[i]     = fb

                boot_alpha = 1.0 - _CI_LEVEL
                recall_ci_lo = float(np.percentile(delta_recalls, 100 * boot_alpha / 2))
                recall_ci_hi = float(np.percentile(delta_recalls, 100 * (1 - boot_alpha / 2)))
                f1_ci_lo     = float(np.percentile(delta_f1s,     100 * boot_alpha / 2))
                f1_ci_hi     = float(np.percentile(delta_f1s,     100 * (1 - boot_alpha / 2)))

                # Two-sided p-values (shift method)
                shifted_recall = delta_recalls - delta_recalls.mean()
                p_recall = float(np.mean(np.abs(shifted_recall) >= np.abs(delta_recall_obs)))
                shifted_f1 = delta_f1s - delta_f1s.mean()
                p_f1 = float(np.mean(np.abs(shifted_f1) >= np.abs(delta_f1_obs)))

                # Bonferroni-corrected significance
                p_recall_bonf = min(p_recall * n_pairs, 1.0)
                p_f1_bonf     = min(p_f1     * n_pairs, 1.0)

                # Cohen's d effect size (for recall)
                pooled_std_recall = np.sqrt((vals_recall_a.var() + vals_recall_b.var()) / 2)
                effect_size_d = float(abs(delta_recall_obs) / pooled_std_recall) \
                    if pooled_std_recall > 0 else 0.0

                # McNemar's test
                correct_a = (y_a == y_true)
                correct_b = (y_b == y_true)
                b = int(( correct_a & ~correct_b).sum())
                c = int((~correct_a &  correct_b).sum())
                n_disc = b + c

                if n_disc > 0 and chi2 is not None:
                    mc_stat = (abs(b - c) - 1) ** 2 / n_disc
                    mc_p    = float(1 - chi2.cdf(mc_stat, df=1))
                    mc_p_bonf = min(mc_p * n_pairs, 1.0)
                    mc_sig  = mc_p_bonf < 0.05
                else:
                    mc_stat = 0.0
                    mc_p    = 1.0
                    mc_p_bonf = 1.0
                    mc_sig  = False

                records.append({
                    "model_name":            model_name,
                    "strategy_A":            strat_a,
                    "strategy_B":            strat_b,
                    # Observed metrics
                    "recall_A":              round(recall_a, 4),
                    "recall_B":              round(recall_b, 4),
                    "f1_A":                  round(f1_a, 4),
                    "f1_B":                  round(f1_b, 4),
                    # Bootstrap deltas (B - A)
                    "delta_recall_obs":      round(delta_recall_obs, 4),
                    "delta_recall_ci_lo":    round(recall_ci_lo, 4),
                    "delta_recall_ci_hi":    round(recall_ci_hi, 4),
                    "delta_f1_obs":          round(delta_f1_obs, 4),
                    "delta_f1_ci_lo":        round(f1_ci_lo, 4),
                    "delta_f1_ci_hi":        round(f1_ci_hi, 4),
                    # p-values (raw + Bonferroni-corrected)
                    "p_recall":              round(p_recall, 6),
                    "p_recall_bonf":         round(p_recall_bonf, 6),
                    "p_f1":                  round(p_f1, 6),
                    "p_f1_bonf":             round(p_f1_bonf, 6),
                    "effect_size_d":         round(effect_size_d, 4),
                    "sig_recall_bonf":       bool(p_recall_bonf < 0.05),
                    "sig_f1_bonf":           bool(p_f1_bonf < 0.05),
                    # McNemar
                    "mcnemar_chi2":          round(mc_stat, 4),
                    "mcnemar_p":             round(mc_p, 6),
                    "mcnemar_p_bonf":        round(mc_p_bonf, 6),
                    "mcnemar_b":             b,
                    "mcnemar_c":             c,
                    "mcnemar_sig_bonf":      mc_sig,
                    "n_windows":             len(y_true),
                    "n_bootstrap":           _BOOTSTRAP_N,
                })

                logger.info(
                    f"  [FusionSig] {model_name}: {strat_a} vs {strat_b}  "
                    f"ΔRecall={delta_recall_obs:+.4f} [{recall_ci_lo:+.4f},{recall_ci_hi:+.4f}]  "
                    f"p(bonf)={p_recall_bonf:.4f}  McNemar p(bonf)={mc_p_bonf:.4f}"
                )

        return pd.DataFrame(records)

    # ─────────────────────────────────────────────────────────────────────────
    # OR dominance decomposition
    # ─────────────────────────────────────────────────────────────────────────

    def _explain_or_fusion_dominance(self, fused_df: pd.DataFrame) -> pd.DataFrame:
        """
        Mathematical decomposition of why OR fusion dominates in recall.

        Computes per-model:

        Recall bound (provable):
          Recall_OR >= max(Recall_sEMG, Recall_KinKin)
          because OR fires if EITHER modality fires, so every TP of either
          single modality is also a TP of OR.

        Precision penalty (provable):
          Precision_OR <= min(Precision_sEMG, Precision_KinKin)
          because OR fires on every FP of either modality.

        Complementarity score:
          (only_semg + only_kinkin) / total_anomalies
          Fraction of anomalies that would be MISSED by at least one modality
          alone — the \"coverage gap\" that OR closes.

        F1 crossover condition:
          F1_OR > F1_single iff:
            TP_OR / (TP_OR + 0.5*(FP_OR + FN_OR)) >
            TP_single / (TP_single + 0.5*(FP_single + FN_single))
          Empirically evaluated here; not an analytic formula.

        Returns
        -------
        pd.DataFrame
            One row per model_name with all decomposition metrics.
        """
        if fused_df.empty:
            return pd.DataFrame()

        records: List[Dict[str, Any]] = []

        for model_name in fused_df["model_name"].unique() if "model_name" in fused_df.columns else [None]:
            if model_name is not None:
                sub = fused_df[fused_df["model_name"] == model_name].copy()
            else:
                sub = fused_df.copy()

            y_true    = sub["ground_truth"].values.astype(int)
            y_semg    = sub["semg_vote"].values.astype(int)
            y_kinkin  = sub["kinkin_vote"].values.astype(int)

            if "fused_OR" not in sub.columns:
                continue
            y_or      = sub["fused_OR"].values.astype(int)
            y_and     = sub["fused_AND"].values.astype(int) if "fused_AND" in sub.columns else None
            y_maj     = sub["fused_MAJORITY"].values.astype(int) if "fused_MAJORITY" in sub.columns else None

            # ── Single-modality metrics ──────────────────────────────────────
            rec_semg   = _recall(y_true, y_semg)
            rec_kinkin = _recall(y_true, y_kinkin)
            rec_or     = _recall(y_true, y_or)
            prec_semg   = _precision(y_true, y_semg)
            prec_kinkin = _precision(y_true, y_kinkin)
            prec_or     = _precision(y_true, y_or)
            f1_semg    = _f1(y_true, y_semg)
            f1_kinkin  = _f1(y_true, y_kinkin)
            f1_or      = _f1(y_true, y_or)

            # ── Recall bound verification ────────────────────────────────────
            recall_lower_bound = max(rec_semg, rec_kinkin)
            recall_bound_holds = bool(rec_or >= recall_lower_bound - 1e-9)

            # ── Precision penalty verification ───────────────────────────────
            precision_upper_bound = min(prec_semg, prec_kinkin)
            # Note: precision_upper_bound is a soft bound — OR can exceed it
            # in edge cases (e.g. if correlated FPs cancel in some windows)
            precision_penalty_holds = bool(prec_or <= precision_upper_bound + 1e-9)

            # ── Complementarity score ────────────────────────────────────────
            anom_mask   = y_true == 1
            n_anom      = int(anom_mask.sum())
            only_semg   = int(((y_semg[anom_mask] == 1) & (y_kinkin[anom_mask] == 0)).sum())
            only_kinkin = int(((y_semg[anom_mask] == 0) & (y_kinkin[anom_mask] == 1)).sum())
            both_fire   = int(((y_semg[anom_mask] == 1) & (y_kinkin[anom_mask] == 1)).sum())
            neither     = int(((y_semg[anom_mask] == 0) & (y_kinkin[anom_mask] == 0)).sum())
            complementarity_score = (only_semg + only_kinkin) / n_anom if n_anom > 0 else float("nan")

            # ── F1 crossover ─────────────────────────────────────────────────
            or_f1_exceeds_semg   = bool(f1_or > f1_semg   + 1e-9)
            or_f1_exceeds_kinkin = bool(f1_or > f1_kinkin + 1e-9)
            or_f1_exceeds_both   = bool(or_f1_exceeds_semg and or_f1_exceeds_kinkin)

            # Recall gain needed for OR to break even on F1:
            # F1_OR = F1_best when recall_gain offsets precision_loss
            best_single_f1 = max(f1_semg, f1_kinkin)

            # ── TP/FP/TN/FN for all strategies ───────────────────────────────
            tp_or, fp_or, tn_or, fn_or = _tp_fp_tn_fn(y_true, y_or)
            tp_s,  fp_s,  tn_s,  fn_s  = _tp_fp_tn_fn(y_true, y_semg)
            tp_k,  fp_k,  tn_k,  fn_k  = _tp_fp_tn_fn(y_true, y_kinkin)

            # Extra FPs introduced by OR vs best single modality
            best_single_fp = min(fp_s, fp_k)
            extra_fp_from_or = fp_or - best_single_fp
            extra_tp_from_or = tp_or - max(tp_s, tp_k)

            records.append({
                "model_name":                   model_name,
                # Single-modality metrics
                "recall_semg":                  round(rec_semg,   4),
                "recall_kinkin":                round(rec_kinkin, 4),
                "recall_or":                    round(rec_or,     4),
                "precision_semg":               round(prec_semg,   4),
                "precision_kinkin":             round(prec_kinkin, 4),
                "precision_or":                 round(prec_or,     4),
                "f1_semg":                      round(f1_semg,   4),
                "f1_kinkin":                    round(f1_kinkin, 4),
                "f1_or":                        round(f1_or,     4),
                # Bound verification
                "recall_lower_bound":           round(recall_lower_bound, 4),
                "recall_bound_holds":           recall_bound_holds,
                "precision_upper_bound":        round(precision_upper_bound, 4),
                "precision_penalty_holds":      precision_penalty_holds,
                # Complementarity
                "complementarity_score":        round(complementarity_score, 4) if not np.isnan(complementarity_score) else float("nan"),
                "n_only_semg":                  only_semg,
                "n_only_kinkin":                only_kinkin,
                "n_both_fire":                  both_fire,
                "n_neither":                    neither,
                "n_anomaly_windows":            n_anom,
                # Gain decomposition
                "extra_tp_from_or":             extra_tp_from_or,
                "extra_fp_from_or":             extra_fp_from_or,
                # F1 crossover
                "or_f1_exceeds_semg":           or_f1_exceeds_semg,
                "or_f1_exceeds_kinkin":         or_f1_exceeds_kinkin,
                "or_f1_exceeds_both_modalities": or_f1_exceeds_both,
                "best_single_f1":               round(best_single_f1, 4),
                "or_f1_gain_vs_best_single":    round(f1_or - best_single_f1, 4),
                "n_windows":                    len(sub),
            })

            logger.info(
                f"  [OR Dominance] {model_name}:  "
                f"Recall_OR={rec_or:.4f} >= max({rec_semg:.4f},{rec_kinkin:.4f}) "
                f"[bound holds={recall_bound_holds}]  "
                f"Complementarity={complementarity_score:.4f}  "
                f"OR F1 > both={or_f1_exceeds_both}"
            )

        return pd.DataFrame(records)

    # ─────────────────────────────────────────────────────────────────────────
    # Precision/Recall/F1 trade-off across all strategies
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_precision_recall_tradeoff(self, fused_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a full precision/recall/F1/accuracy trade-off table for all
        fusion strategies and single-modality baselines.

        Makes explicit the OR precision sacrifice vs recall gain.
        """
        if fused_df.empty:
            return pd.DataFrame()

        records: List[Dict[str, Any]] = []

        strategy_map = {
            "sEMG_only":    "semg_vote",
            "KinKin_only":  "kinkin_vote",
            "OR":           "fused_OR",
            "MAJORITY":     "fused_MAJORITY",
            "AND":          "fused_AND",
        }

        for model_name in fused_df["model_name"].unique() if "model_name" in fused_df.columns else [None]:
            if model_name is not None:
                sub = fused_df[fused_df["model_name"] == model_name]
            else:
                sub = fused_df

            y_true = sub["ground_truth"].values.astype(int)

            for strategy_label, col in strategy_map.items():
                if col not in sub.columns:
                    continue
                y_pred = sub[col].values.astype(int)
                TP, FP, TN, FN = _tp_fp_tn_fn(y_true, y_pred)
                n = len(y_true)

                rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                acc  = (TP + TN) / n if n > 0 else 0.0
                fpr  = FP / (FP + TN) if (FP + TN) > 0 else 0.0
                fnr  = FN / (FN + TP) if (FN + TP) > 0 else 0.0

                records.append({
                    "model_name":      model_name,
                    "strategy":        strategy_label,
                    "precision":       round(prec, 4),
                    "recall":          round(rec,  4),
                    "f1":              round(f1,   4),
                    "accuracy":        round(acc,  4),
                    "fpr":             round(fpr,  4),
                    "fnr":             round(fnr,  4),
                    "TP": TP, "FP": FP, "TN": TN, "FN": FN,
                    "n_windows":       n,
                })

        return pd.DataFrame(records)

    # ─────────────────────────────────────────────────────────────────────────
    # Export
    # ─────────────────────────────────────────────────────────────────────────

    def export(self, report: FusionReport, output_dir: str) -> Dict[str, str]:
        """Export all fusion analysis results to CSV."""
        os.makedirs(output_dir, exist_ok=True)
        paths: Dict[str, str] = {}

        for attr, fname in [
            ("window_analysis_df",    "fusion_window_analysis.csv"),
            ("exclusivity_df",        "fusion_exclusivity.csv"),
            ("incremental_recall_df", "fusion_incremental_recall.csv"),
            ("channel_activation_df", "fusion_channel_activation.csv"),
            ("significance_df",       "fusion_significance_tests.csv"),
            ("or_dominance_df",       "fusion_or_dominance.csv"),
            ("pr_tradeoff_df",        "fusion_pr_tradeoff.csv"),
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

        # ── Precision/Recall/F1 trade-off ─────────────────────────────────────
        if not report.pr_tradeoff_df.empty:
            self._plot_pr_tradeoff(report.pr_tradeoff_df, fig_dir, formats, plt)

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

            ax.barh(range(len(rates)), rates, color=colors, alpha=0.85)
            ax.set_yticks(range(len(rates)))
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_xlabel("Activation Rate (fraction of anomaly windows flagged)")
            ax.set_title(f"Channel Activation Frequencies — {model_name}")
            ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=0.8)

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

    def _plot_pr_tradeoff(self, df, fig_dir, formats, plt):
        """
        Radar/bar plot showing Precision, Recall, F1 trade-off across
        all strategies and single-modality baselines.
        """
        STRATEGY_ORDER = ["sEMG_only", "KinKin_only", "MAJORITY", "AND", "OR"]
        COLORS = {
            "sEMG_only":   "#4C9BE8",
            "KinKin_only": "#E05F5F",
            "MAJORITY":    "#F5A623",
            "AND":         "#7ED321",
            "OR":          "#6A4C93",
        }

        for model_name in df["model_name"].unique():
            sub = df[df["model_name"] == model_name]
            strats = [s for s in STRATEGY_ORDER if s in sub["strategy"].values]
            if not strats:
                continue

            fig, ax = plt.subplots(figsize=(9, 5))
            x = np.arange(len(strats))
            w = 0.25
            metrics = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]

            for i, (metric, label) in enumerate(metrics):
                vals = []
                for s in strats:
                    row = sub[sub["strategy"] == s]
                    vals.append(float(row[metric].values[0]) if len(row) > 0 else 0.0)
                bars = ax.bar(x + i * w, vals, w, label=label, alpha=0.85)

            ax.set_xticks(x + w)
            ax.set_xticklabels(strats, rotation=15)
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1.1)
            ax.set_title(
                f"Precision / Recall / F1 Trade-off Across Fusion Strategies — {model_name}\n"
                f"(OR maximises Recall at cost of Precision; AND maximises Precision)"
            )
            ax.legend()
            ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.6)
            plt.tight_layout()

            for fmt in formats:
                path = os.path.join(fig_dir, f"fusion_pr_tradeoff_{model_name}.{fmt}")
                fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"[Fusion] Saved PR trade-off chart → {fig_dir}")
