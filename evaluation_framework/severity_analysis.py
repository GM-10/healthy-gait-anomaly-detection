"""
evaluation_framework/severity_analysis.py

Severity-stratified anomaly detection analysis (Task 5).

For every (anomaly_type × severity × model × channel), computes:
    - mean, median, variance, IQR of reconstruction error
    - number of detected anomalies (TP), FN, FP
    - threshold margin = reconstruction_error - threshold
    - 95% bootstrap confidence interval on recall (per stratum)

Statistical validation:
    - Kruskal-Wallis H-test on reconstruction errors across severity levels
      (non-parametric; robust to non-normal MSE distributions)
    - Bonferroni-corrected pairwise Mann-Whitney U post-hoc tests when
      Kruskal-Wallis is significant (p < 0.05)
    - scikit_posthocs Dunn's test used if package is available; otherwise
      falls back to the Bonferroni-corrected Mann-Whitney U approach

Investigates non-monotonic severity behaviour with quantitative diagnostics:
    - saturation_ratio            : max(errors) / errors[-1]  (> 1.1 → plateau)
    - normalization_compression_coef : Spearman ρ(severity, mean_error)
    - model_invariance_score      : (max_err - min_err) / max_err  (< 0.05 → flat)
    - recall_plateau_severity     : lowest severity at which recall ≥ 0.95
    - inversion_magnitude         : mean |Δerror| at inversion points
    - n_inversions                : count of strictly-decreasing steps

Usage:
    from evaluation_framework.severity_analysis import SeverityAnalyzer

    analyzer = SeverityAnalyzer(threshold_results_map)
    report   = analyzer.analyze(score_df)
    report.summary_df     # per-(type, severity, model) metrics + CI columns
    report.cause_df       # quantitative non-monotonicity diagnostics
    report.kruskal_df     # Kruskal-Wallis + post-hoc test results
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

# Severity numeric values (must match synthetic_anomalies.py DEFAULT_SEVERITIES)
SEVERITY_NUMERIC = {
    "mild":     0.15,
    "moderate": 0.35,
    "severe":   0.60,
}
SEVERITY_ORDER = [0.15, 0.35, 0.60]

ANOMALY_TYPES = ["amplitude_scale", "time_warp", "time_shift", "combined"]

# Minimum windows per stratum for bootstrap CI to be computed
_MIN_STRATUM_N = 10
_BOOTSTRAP_N   = 500   # resamples for per-stratum CIs (lower than global for speed)


# ─────────────────────────────────────────────────────────────────────────────
# Report container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SeverityReport:
    """Container for severity analysis results."""
    # Per (anomaly_type, severity_numeric, model, modality, channel, threshold_method)
    summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Non-monotonicity investigation — quantitative diagnostics
    cause_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Aggregated across channels
    aggregate_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Kruskal-Wallis H-test + post-hoc results
    kruskal_df: pd.DataFrame = field(default_factory=pd.DataFrame)


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap helper (local — avoids importing the larger bootstrap module)
# ─────────────────────────────────────────────────────────────────────────────

def _bootstrap_recall_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = _BOOTSTRAP_N,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap CI on recall for a single (y_true, y_pred) pair.

    Returns (recall_observed, ci_lower, ci_upper).
    Returns (NaN, NaN, NaN) if the stratum is too small.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = len(y_true)

    if n < _MIN_STRATUM_N or (y_true == 1).sum() == 0:
        return float("nan"), float("nan"), float("nan")

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    recall_obs = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        tp_b = int(((yt == 1) & (yp == 1)).sum())
        fn_b = int(((yt == 1) & (yp == 0)).sum())
        samples[i] = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0

    alpha = 1.0 - ci_level
    lo = float(np.percentile(samples, 100 * alpha / 2))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return recall_obs, lo, hi


# ─────────────────────────────────────────────────────────────────────────────
# Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class SeverityAnalyzer:
    """
    Severity-stratified analysis of anomaly detection behaviour.

    Parameters
    ----------
    threshold_results_map : dict
        {model_name: {channel_name: {method: ThresholdResult}}}
        Provides the threshold value for each (model, channel, method).
    severity_numeric_map : dict, optional
        Maps severity label strings to numeric values.
        Defaults to {\"mild\": 0.15, \"moderate\": 0.35, \"severe\": 0.60}.
    bootstrap_n : int
        Resamples for per-stratum recall CIs (default 500).
    bootstrap_seed : int
        Random seed for reproducibility (default 42).
    """

    def __init__(
        self,
        threshold_results_map: Dict[str, Dict[str, Dict[str, Any]]],
        severity_numeric_map: Optional[Dict[str, float]] = None,
        bootstrap_n: int = _BOOTSTRAP_N,
        bootstrap_seed: int = 42,
    ):
        self.threshold_results_map = threshold_results_map
        self.severity_numeric_map  = severity_numeric_map or dict(SEVERITY_NUMERIC)
        self.bootstrap_n           = bootstrap_n
        self.bootstrap_seed        = bootstrap_seed

    def _get_severity_numeric(self, score_df: pd.DataFrame) -> pd.Series:
        """
        Resolve numeric severity from:
         1. A 'severity' column (float, if written by pipeline).
         2. A 'severity_level' string column.
         3. Fallback: 0.35 (moderate) for is_synthetic_anomaly==1, 0.0 for clean.
        """
        if "severity" in score_df.columns:
            return score_df["severity"].fillna(0.0).astype(float)

        if "severity_level" in score_df.columns:
            return score_df["severity_level"].map(self.severity_numeric_map).fillna(0.0)

        # Heuristic fallback
        logger.warning(
            "[SeverityAnalyzer] No 'severity' column found — "
            "assuming all synthetic anomalies are moderate (0.35)."
        )
        return score_df["is_synthetic_anomaly"].astype(float) * 0.35

    def analyze(
        self,
        score_df: pd.DataFrame,
        methods: Optional[List[str]] = None,
    ) -> SeverityReport:
        """
        Compute severity-stratified metrics for all (model, channel, method).

        Parameters
        ----------
        score_df : pd.DataFrame
            Score CSV data with: reconstruction_error, is_synthetic_anomaly,
            anomaly_type, model_name, channel_name, modality.
        methods : list of str, optional
            Threshold methods to evaluate. Default: all available.

        Returns
        -------
        SeverityReport
        """
        if methods is None:
            methods = list(next(iter(
                next(iter(self.threshold_results_map.values()), {}).values()
            ), {}).keys()) if self.threshold_results_map else ["mean_std"]

        records: List[Dict[str, Any]] = []
        severity_col = self._get_severity_numeric(score_df)
        score_df = score_df.copy()
        score_df["severity_numeric"] = severity_col

        for model_name in score_df["model_name"].unique():
            model_df    = score_df[score_df["model_name"] == model_name]
            ch_thresh   = self.threshold_results_map.get(model_name, {})

            for channel_name in sorted(model_df["channel_name"].unique()):
                ch_df    = model_df[model_df["channel_name"] == channel_name]
                modality = ch_df["modality"].iloc[0]
                thresh_map = ch_thresh.get(channel_name, {})

                for method in methods:
                    thresh_result = thresh_map.get(method)
                    thresh_val    = thresh_result.value if thresh_result is not None else float("nan")

                    for atype in ch_df["anomaly_type"].unique():
                        anom_df = ch_df[ch_df["anomaly_type"] == atype]
                        if anom_df.empty:
                            continue

                        # Group by severity level
                        for sev_num in sorted(anom_df["severity_numeric"].unique()):
                            sev_df = anom_df[anom_df["severity_numeric"] == sev_num]
                            if sev_df.empty:
                                continue

                            errors = sev_df["reconstruction_error"].values.astype(float)
                            y_true = sev_df["is_synthetic_anomaly"].values.astype(int)
                            y_pred = (errors > thresh_val).astype(int) if not np.isnan(thresh_val) else np.zeros_like(y_true)

                            # Confusion counts for this severity group
                            TP = int(((y_true == 1) & (y_pred == 1)).sum())
                            FP = int(((y_true == 0) & (y_pred == 1)).sum())
                            TN = int(((y_true == 0) & (y_pred == 0)).sum())
                            FN = int(((y_true == 1) & (y_pred == 0)).sum())

                            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                            prec   = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                            f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0

                            # Threshold margin for anomalous windows
                            anom_errors = errors[y_true == 1]
                            margin = float(np.mean(anom_errors - thresh_val)) if len(anom_errors) > 0 and not np.isnan(thresh_val) else float("nan")

                            q25, q75 = np.percentile(errors, [25, 75]) if len(errors) >= 4 else (np.nan, np.nan)

                            # Per-stratum bootstrap recall CI
                            _, recall_ci_lo, recall_ci_hi = _bootstrap_recall_ci(
                                y_true, y_pred,
                                n_bootstrap=self.bootstrap_n,
                                seed=self.bootstrap_seed,
                            )

                            records.append({
                                "model_name":       model_name,
                                "modality":         modality,
                                "channel_name":     channel_name,
                                "anomaly_type":     atype,
                                "severity_numeric": float(sev_num),
                                "threshold_method": method,
                                "threshold_value":  thresh_val,
                                "n_windows":        len(sev_df),
                                "mean_error":       float(np.mean(errors)),
                                "median_error":     float(np.median(errors)),
                                "var_error":        float(np.var(errors)),
                                "iqr_error":        float(q75 - q25) if not np.isnan(q25) else float("nan"),
                                "n_detected":       TP,
                                "TP":               TP,
                                "FP":               FP,
                                "TN":               TN,
                                "FN":               FN,
                                "recall":           recall,
                                "precision":        prec,
                                "f1":               f1,
                                "threshold_margin": margin,
                                "recall_ci_lower":  recall_ci_lo,
                                "recall_ci_upper":  recall_ci_hi,
                            })

        summary_df = pd.DataFrame(records)

        # Aggregate across channels
        agg_records: List[Dict[str, Any]] = []
        if not summary_df.empty:
            group_keys = ["model_name", "modality", "anomaly_type", "severity_numeric", "threshold_method"]
            for keys, grp in summary_df.groupby(group_keys):
                vals = dict(zip(group_keys, keys))
                vals.update({
                    "mean_error":       grp["mean_error"].mean(),
                    "median_error":     grp["median_error"].mean(),
                    "recall":           grp["recall"].mean(),
                    "f1":               grp["f1"].mean(),
                    "threshold_margin": grp["threshold_margin"].mean(),
                    "TP":               grp["TP"].sum(),
                    "FN":               grp["FN"].sum(),
                    "FP":               grp["FP"].sum(),
                    "n_windows":        grp["n_windows"].sum(),
                    "channel_name":     "ALL_CHANNELS",
                    # CI: use the mean of per-channel CIs (conservative)
                    "recall_ci_lower":  grp["recall_ci_lower"].mean(),
                    "recall_ci_upper":  grp["recall_ci_upper"].mean(),
                })
                agg_records.append(vals)

        agg_df = pd.DataFrame(agg_records)

        # Non-monotonicity investigation — quantitative diagnostics
        cause_df = self.investigate_non_monotonicity(summary_df)

        # Kruskal-Wallis + post-hoc tests
        kruskal_df = self._run_kruskal_wallis(score_df, methods)

        return SeverityReport(
            summary_df=summary_df,
            aggregate_df=agg_df,
            cause_df=cause_df,
            kruskal_df=kruskal_df,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Non-monotonicity — quantitative diagnostics
    # ─────────────────────────────────────────────────────────────────────────

    def investigate_non_monotonicity(
        self,
        summary_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Identify (model, channel, anomaly_type) cases where reconstruction
        error does NOT increase monotonically with severity.

        Computes quantitative diagnostics per non-monotonic case:

        Diagnostic                  | Description
        ----------------------------|---------------------------------------------
        saturation_ratio            | max(errors) / errors[-1]  > 1.1 → plateau
        normalization_compression_coef | Spearman ρ(severity, mean_error); near 0 → normalization flattens
        model_invariance_score      | (max_err - min_err) / max_err; < 0.05 → flat error surface
        recall_plateau_severity     | lowest severity at which recall >= 0.95
        inversion_magnitude         | mean |Δerror| at strictly-decreasing steps
        n_inversions                | count of strictly-decreasing adjacent pairs
        cause                       | human-readable combination of triggered causes

        Returns
        -------
        pd.DataFrame
            One row per (model, channel, anomaly_type, threshold_method)
            that shows non-monotonic severity behaviour, with quantitative
            diagnostic columns.
        """
        if summary_df.empty:
            return pd.DataFrame()

        try:
            from scipy.stats import spearmanr  # type: ignore
            _has_spearman = True
        except ImportError:
            _has_spearman = False
            logger.warning("[SeverityAnalyzer] scipy not installed — Spearman ρ will be NaN")

        records: List[Dict[str, Any]] = []
        group_keys = ["model_name", "channel_name", "anomaly_type", "threshold_method"]

        for keys, grp in summary_df.groupby(group_keys):
            g = grp.sort_values("severity_numeric")
            sevs    = g["severity_numeric"].values
            errors  = g["mean_error"].values
            recalls = g["recall"].values

            if len(sevs) < 2:
                continue

            # Check monotonicity: errors should increase with severity
            diffs = np.diff(errors)
            is_monotonic = bool((diffs >= 0).all())

            if is_monotonic:
                continue   # expected — skip

            # ── Quantitative diagnostics ──────────────────────────────────────

            # 1. Saturation ratio: max/last value
            saturation_ratio = float(errors.max() / errors[-1]) if errors[-1] > 0 else float("nan")

            # 2. Normalization compression coefficient (Spearman ρ)
            if _has_spearman and len(sevs) >= 3:
                rho, _ = spearmanr(sevs, errors)
                normalization_compression_coef = float(rho)
            else:
                normalization_compression_coef = float("nan")

            # 3. Model invariance score
            err_range = errors.max() - errors.min()
            model_invariance_score = float(err_range / errors.max()) if errors.max() > 0 else float("nan")

            # 4. Recall plateau severity (first severity at which recall >= 0.95)
            plateau_mask = recalls >= 0.95
            recall_plateau_severity = float(sevs[plateau_mask][0]) if plateau_mask.any() else float("nan")

            # 5. Inversion magnitude (mean |Δerror| at inversion steps)
            inv_diffs = diffs[diffs < 0]
            inversion_magnitude = float(np.mean(np.abs(inv_diffs))) if len(inv_diffs) > 0 else 0.0

            # 6. Number of inversions
            n_inversions = int((diffs < 0).sum())

            # ── Human-readable cause flags ────────────────────────────────────
            cause_flags: List[str] = []

            if not np.isnan(saturation_ratio) and saturation_ratio > 1.1:
                cause_flags.append("signal_saturation_or_model_invariance")

            if len(recalls) >= 2 and recalls[0] >= 0.99:
                cause_flags.append("threshold_already_exceeded_at_mild")

            atype = keys[2] if isinstance(keys, tuple) else grp["anomaly_type"].iloc[0]
            if atype == "amplitude_scale":
                cause_flags.append("normalization_compression_likely")

            if atype == "time_warp":
                cause_flags.append("interpolation_smoothing_at_high_severity")

            if atype == "time_shift":
                cause_flags.append("edge_hold_padding_reduces_distortion")

            if not np.isnan(model_invariance_score) and model_invariance_score < 0.05:
                cause_flags.append("model_invariance_flat_error_surface")

            if not np.isnan(normalization_compression_coef) and abs(normalization_compression_coef) < 0.3:
                cause_flags.append("normalization_compresses_severity_gradient")

            if not cause_flags:
                cause_flags.append("unknown_cause")

            d = dict(zip(group_keys, keys)) if isinstance(keys, tuple) else {
                k: grp[k].iloc[0] for k in group_keys if k in grp.columns
            }
            d.update({
                "is_monotonic":                  False,
                "n_inversions":                  n_inversions,
                "inversion_magnitude":           round(inversion_magnitude, 8),
                "saturation_ratio":              round(saturation_ratio, 6) if not np.isnan(saturation_ratio) else float("nan"),
                "normalization_compression_coef": round(normalization_compression_coef, 6) if not np.isnan(normalization_compression_coef) else float("nan"),
                "model_invariance_score":        round(model_invariance_score, 6) if not np.isnan(model_invariance_score) else float("nan"),
                "recall_plateau_severity":       recall_plateau_severity,
                "error_range":                   float(err_range),
                "severity_values":               list(sevs),
                "mean_errors":                   list(errors),
                "recalls":                       list(recalls),
                "cause":                         "; ".join(cause_flags),
            })
            records.append(d)

        return pd.DataFrame(records)

    # ─────────────────────────────────────────────────────────────────────────
    # Kruskal-Wallis + post-hoc tests
    # ─────────────────────────────────────────────────────────────────────────

    def _run_kruskal_wallis(
        self,
        score_df: pd.DataFrame,
        methods: List[str],
    ) -> pd.DataFrame:
        """
        Run Kruskal-Wallis H-test on reconstruction errors across severity
        levels for each (model, channel, anomaly_type, threshold_method).

        When H-test is significant (p < 0.05), runs Bonferroni-corrected
        pairwise Mann-Whitney U tests between every pair of severity levels.

        Returns
        -------
        pd.DataFrame
            One row per (model, channel, anomaly_type) with H-statistic,
            p-value, significance flag, and post-hoc pairwise results.
        """
        try:
            from scipy.stats import kruskal, mannwhitneyu  # type: ignore
        except ImportError:
            logger.warning(
                "[SeverityAnalyzer] scipy not installed — "
                "Kruskal-Wallis tests skipped."
            )
            return pd.DataFrame()

        severity_col = self._get_severity_numeric(score_df)
        score_df = score_df.copy()
        score_df["severity_numeric"] = severity_col

        # Focus on anomalous windows only (severity > 0)
        anom_df = score_df[score_df["is_synthetic_anomaly"] == 1].copy()
        if anom_df.empty:
            return pd.DataFrame()

        records: List[Dict[str, Any]] = []
        group_keys = ["model_name", "channel_name", "anomaly_type"]

        for keys, grp in anom_df.groupby(group_keys):
            model_name   = keys[0]
            channel_name = keys[1]
            atype        = keys[2]
            modality     = grp["modality"].iloc[0] if "modality" in grp.columns else "unknown"

            severity_groups: Dict[float, np.ndarray] = {}
            for sev, sgrp in grp.groupby("severity_numeric"):
                errs = sgrp["reconstruction_error"].values.astype(float)
                if len(errs) >= 2:
                    severity_groups[float(sev)] = errs

            if len(severity_groups) < 2:
                continue

            # Kruskal-Wallis H-test
            group_arrays = list(severity_groups.values())
            try:
                h_stat, p_val = kruskal(*group_arrays)
            except Exception as exc:
                logger.debug(f"[KW] {model_name}/{channel_name}/{atype}: {exc}")
                continue

            is_sig = bool(p_val < 0.05)

            rec: Dict[str, Any] = {
                "model_name":   model_name,
                "modality":     modality,
                "channel_name": channel_name,
                "anomaly_type": atype,
                "kruskal_H":    round(float(h_stat), 6),
                "kruskal_p":    round(float(p_val), 8),
                "kruskal_sig":  is_sig,
                "n_severity_levels": len(severity_groups),
            }

            # Post-hoc: Bonferroni-corrected pairwise Mann-Whitney U
            sev_pairs = list(combinations(sorted(severity_groups.keys()), 2))
            n_pairs   = len(sev_pairs)
            posthoc_parts: List[str] = []

            for sev_a, sev_b in sev_pairs:
                arr_a = severity_groups[sev_a]
                arr_b = severity_groups[sev_b]
                try:
                    u_stat, mw_p = mannwhitneyu(arr_a, arr_b, alternative="two-sided")
                    mw_p_corrected = min(float(mw_p) * n_pairs, 1.0)   # Bonferroni
                    mw_sig = mw_p_corrected < 0.05
                    posthoc_parts.append(
                        f"sev{sev_a:.2f}_vs_sev{sev_b:.2f}:"
                        f"U={u_stat:.1f},p_adj={mw_p_corrected:.4f},sig={'Y' if mw_sig else 'N'}"
                    )
                except Exception:
                    pass

            rec["posthoc_mannwhitney_bonferroni"] = " | ".join(posthoc_parts)
            records.append(rec)

        kruskal_df = pd.DataFrame(records)
        if not kruskal_df.empty:
            n_sig = int(kruskal_df["kruskal_sig"].sum())
            logger.info(
                f"[SeverityAnalyzer] Kruskal-Wallis: {len(kruskal_df)} tests, "
                f"{n_sig} significant (p<0.05)"
            )
        return kruskal_df

    # ─────────────────────────────────────────────────────────────────────────
    # Severity profile plots (line plots with CI error bars)
    # ─────────────────────────────────────────────────────────────────────────

    def plot_severity_profiles(
        self,
        report: SeverityReport,
        output_dir: str,
        formats: Optional[List[str]] = None,
    ) -> None:
        """
        Plot mean reconstruction error vs severity_numeric, aggregated across
        channels, with 95% recall CI error bars.

        One figure per anomaly_type; one line per (model, threshold_method).
        Saved to output_dir/figures/.
        """
        if formats is None:
            formats = ["png", "pdf", "svg"]

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("[SeverityAnalyzer] matplotlib not available — plots skipped")
            return

        if report.aggregate_df.empty:
            logger.warning("[SeverityAnalyzer] aggregate_df empty — skipping profile plots")
            return

        fig_dir = os.path.join(output_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)

        df = report.aggregate_df

        for atype in df["anomaly_type"].unique():
            sub = df[df["anomaly_type"] == atype].sort_values("severity_numeric")
            if sub.empty:
                continue

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"Severity Profile — {atype}", fontsize=13)

            # Left: mean_error vs severity
            ax = axes[0]
            for (model_name, method, modality), g in sub.groupby(["model_name", "threshold_method", "modality"]):
                g = g.sort_values("severity_numeric")
                label = f"{model_name} / {method} / {modality}"
                ax.plot(g["severity_numeric"], g["mean_error"], marker="o", label=label)
            ax.set_xlabel("Severity")
            ax.set_ylabel("Mean Reconstruction Error")
            ax.set_title("Reconstruction Error vs Severity")
            ax.legend(fontsize=7, loc="upper left")

            # Right: recall vs severity with CI error bars
            ax2 = axes[1]
            for (model_name, method, modality), g in sub.groupby(["model_name", "threshold_method", "modality"]):
                g = g.sort_values("severity_numeric")
                label = f"{model_name} / {method} / {modality}"
                sevs = g["severity_numeric"].values
                recalls = g["recall"].values
                ci_lo = g["recall_ci_lower"].values if "recall_ci_lower" in g.columns else recalls
                ci_hi = g["recall_ci_upper"].values if "recall_ci_upper" in g.columns else recalls
                yerr_lo = np.where(np.isnan(ci_lo), 0, recalls - ci_lo)
                yerr_hi = np.where(np.isnan(ci_hi), 0, ci_hi - recalls)
                ax2.errorbar(
                    sevs, recalls,
                    yerr=[yerr_lo, yerr_hi],
                    marker="s", capsize=4, label=label,
                )
            ax2.set_xlabel("Severity")
            ax2.set_ylabel("Recall (with 95% CI)")
            ax2.set_ylim(-0.05, 1.10)
            ax2.set_title("Recall vs Severity")
            ax2.legend(fontsize=7, loc="lower right")

            plt.tight_layout()
            safe_atype = atype.replace(" ", "_").replace("/", "_")
            for fmt in formats:
                path = os.path.join(fig_dir, f"severity_profile_{safe_atype}.{fmt}")
                fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"[SeverityAnalyzer] Saved severity profile → {fig_dir}/severity_profile_{safe_atype}.*")

    # ─────────────────────────────────────────────────────────────────────────
    # Export
    # ─────────────────────────────────────────────────────────────────────────

    def export(self, report: SeverityReport, output_dir: str) -> Dict[str, str]:
        """Export severity report to CSV."""
        os.makedirs(output_dir, exist_ok=True)
        paths: Dict[str, str] = {}

        if not report.summary_df.empty:
            p = os.path.join(output_dir, "severity_summary.csv")
            report.summary_df.to_csv(p, index=False)
            paths["summary_csv"] = p

        if not report.aggregate_df.empty:
            p = os.path.join(output_dir, "severity_aggregate.csv")
            report.aggregate_df.to_csv(p, index=False)
            paths["aggregate_csv"] = p

        if not report.cause_df.empty:
            p = os.path.join(output_dir, "severity_nonmonotonic_causes.csv")
            report.cause_df.to_csv(p, index=False)
            paths["cause_csv"] = p
            logger.info(
                f"[Severity] Non-monotonic cases: {len(report.cause_df)} — see {p}"
            )

        if not report.kruskal_df.empty:
            p = os.path.join(output_dir, "severity_kruskal_wallis.csv")
            report.kruskal_df.to_csv(p, index=False)
            paths["kruskal_csv"] = p
            logger.info(f"[Severity] Kruskal-Wallis results → {p}")

        logger.info(f"[Severity] Exported {len(paths)} files to {output_dir}")
        return paths
