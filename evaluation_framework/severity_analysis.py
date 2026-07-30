"""
evaluation_framework/severity_analysis.py

Severity-stratified anomaly detection analysis (Task 5).

For every (anomaly_type × severity × model × channel), computes:
    - mean, median, variance, IQR of reconstruction error
    - number of detected anomalies (TP), FN, FP
    - threshold margin = reconstruction_error - threshold

Investigates non-monotonic severity behaviour:
    - Checks if larger perturbations increase reconstruction error as expected
    - Identifies possible causes: signal saturation, model invariance,
      window interpolation, threshold effects, normalization effects,
      reconstruction smoothing

Usage:
    from evaluation_framework.severity_analysis import SeverityAnalyzer

    analyzer = SeverityAnalyzer(threshold_results_map)
    report   = analyzer.analyze(score_df)
    report.summary_df  # per-(type, severity, model) metrics
    report.cause_df    # non-monotonicity diagnosis per (type, model)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
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


# ─────────────────────────────────────────────────────────────────────────────
# Report container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SeverityReport:
    """Container for severity analysis results."""
    # Per (anomaly_type, severity_numeric, model, modality, channel, threshold_method)
    summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Non-monotonicity investigation
    cause_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Aggregated across channels
    aggregate_df: pd.DataFrame = field(default_factory=pd.DataFrame)


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
        Defaults to {"mild": 0.15, "moderate": 0.35, "severe": 0.60}.
    """

    def __init__(
        self,
        threshold_results_map: Dict[str, Dict[str, Dict[str, Any]]],
        severity_numeric_map: Optional[Dict[str, float]] = None,
    ):
        self.threshold_results_map = threshold_results_map
        self.severity_numeric_map  = severity_numeric_map or dict(SEVERITY_NUMERIC)

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
                })
                agg_records.append(vals)

        agg_df = pd.DataFrame(agg_records)

        # Non-monotonicity investigation
        cause_df = self.investigate_non_monotonicity(summary_df)

        return SeverityReport(
            summary_df=summary_df,
            aggregate_df=agg_df,
            cause_df=cause_df,
        )

    def investigate_non_monotonicity(
        self,
        summary_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Identify (model, channel, anomaly_type) cases where reconstruction
        error does NOT increase monotonically with severity.

        Investigates possible causes:
            - Signal saturation (errors plateau or decrease at severe)
            - Threshold effects (error above threshold even at mild → recall flat)
            - Normalization effects (amplitude scaling compressed by MinMax scaler)
            - Window interpolation artifacts (time warp introduces artefacts at high severity)
            - Model invariance (LSTM/Transformer reconstructs anomalies equally well)
            - Reconstruction smoothing (model averages out distortions)

        Returns
        -------
        pd.DataFrame
            One row per (model, channel, anomaly_type, threshold_method)
            that shows non-monotonic severity behaviour, with a `cause` column.
        """
        if summary_df.empty:
            return pd.DataFrame()

        records: List[Dict[str, Any]] = []
        group_keys = ["model_name", "channel_name", "anomaly_type", "threshold_method"]

        for keys, grp in summary_df.groupby(group_keys):
            g = grp.sort_values("severity_numeric")
            sevs   = g["severity_numeric"].values
            errors = g["mean_error"].values
            recalls = g["recall"].values

            if len(sevs) < 2:
                continue

            # Check monotonicity: errors should increase with severity
            diffs = np.diff(errors)
            is_monotonic = bool((diffs >= 0).all())

            if is_monotonic:
                continue   # expected — skip

            # Non-monotonic: investigate cause
            cause_flags: List[str] = []

            # 1. Signal saturation: error plateaus at high severity
            if len(errors) >= 3 and errors[-1] <= errors[-2]:
                cause_flags.append("signal_saturation_or_model_invariance")

            # 2. Threshold effects: recall is already 1.0 at mild severity
            if len(recalls) >= 2 and recalls[0] >= 0.99:
                cause_flags.append("threshold_already_exceeded_at_mild")

            # 3. Amplitude scaling compressed by MinMax normalization
            atype = keys[2] if isinstance(keys, tuple) else grp["anomaly_type"].iloc[0]
            if atype == "amplitude_scale":
                cause_flags.append("normalization_compression_likely")

            # 4. Time warp: interpolation at high severity may smooth error
            if atype == "time_warp":
                cause_flags.append("interpolation_smoothing_at_high_severity")

            # 5. Time shift: edge-hold padding reduces actual distortion at high severity
            if atype == "time_shift":
                cause_flags.append("edge_hold_padding_reduces_distortion")

            # 6. Model invariance: error range is too small
            err_range = errors.max() - errors.min()
            if err_range < 1e-5:
                cause_flags.append("model_invariance_flat_error_surface")

            if not cause_flags:
                cause_flags.append("unknown_cause")

            d = dict(zip(group_keys, keys)) if isinstance(keys, tuple) else {
                k: grp[k].iloc[0] for k in group_keys if k in grp.columns
            }
            d.update({
                "is_monotonic":       False,
                "error_range":        float(err_range),
                "severity_values":    list(sevs),
                "mean_errors":        list(errors),
                "recalls":            list(recalls),
                "cause":              "; ".join(cause_flags),
                "n_reversals":        int((diffs < 0).sum()),
            })
            records.append(d)

        return pd.DataFrame(records)

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

        logger.info(f"[Severity] Exported {len(paths)} files to {output_dir}")
        return paths
