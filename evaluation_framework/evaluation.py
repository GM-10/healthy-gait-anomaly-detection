"""
evaluation_framework/evaluation.py

Full metric computation engine for the gait anomaly detection research framework.

Computes:
    Precision, Recall, F1, Accuracy
    ROC-AUC, PR-AUC (using continuous reconstruction_error as score)
    False Positive Rate (FPR), False Negative Rate (FNR)
    TP, FP, TN, FN

Supports:
    - Per-channel evaluation across all threshold methods
    - Per-modality aggregation (sEMG / Kinematics / Kinetics / Fusion)
    - Cross-model comparison tables

Usage:
    from evaluation_framework.evaluation import evaluate_all_thresholds

    results_df = evaluate_all_thresholds(
        score_df=score_df,
        train_errors=train_errors_dict,  # {channel_name: np.ndarray}
        methods=["mean_std", "percentile95", "percentile99"],
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Metric result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MetricsResult:
    """All metrics for one (model, modality, channel, threshold_method) configuration."""
    model_name:       str
    modality:         str
    channel_name:     str
    threshold_method: str
    threshold_value:  float

    # Confusion matrix
    TP: int = 0
    FP: int = 0
    TN: int = 0
    FN: int = 0
    n_windows: int = 0

    # Classification metrics
    precision: float = 0.0
    recall:    float = 0.0
    f1:        float = 0.0
    accuracy:  float = 0.0
    fpr:       float = 0.0   # False Positive Rate = FP / (FP + TN)
    fnr:       float = 0.0   # False Negative Rate = FN / (FN + TP)

    # Ranking metrics (require continuous scores)
    roc_auc: float = float("nan")
    pr_auc:  float = float("nan")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Core metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """Return (TP, FP, TN, FN)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())
    return TP, FP, TN, FN


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: Optional[np.ndarray] = None,
    model_name: str = "unknown",
    modality: str = "unknown",
    channel_name: str = "unknown",
    threshold_method: str = "unknown",
    threshold_value: float = float("nan"),
) -> MetricsResult:
    """
    Compute the full metric suite for one (model, channel, threshold) configuration.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels (0=normal, 1=anomaly).
    y_pred : np.ndarray
        Predicted binary labels derived from thresholding.
    scores : np.ndarray, optional
        Continuous anomaly scores (reconstruction errors). If provided,
        ROC-AUC and PR-AUC are computed.
    model_name, modality, channel_name, threshold_method : str
        Metadata fields for the result.
    threshold_value : float
        The scalar threshold used.

    Returns
    -------
    MetricsResult
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    TP, FP, TN, FN = _confusion(y_true, y_pred)
    n = len(y_true)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (TP + TN) / n if n > 0 else 0.0
    fpr       = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    fnr       = FN / (FN + TP) if (FN + TP) > 0 else 0.0

    roc_auc = float("nan")
    pr_auc  = float("nan")
    if scores is not None and len(np.unique(y_true)) == 2:
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            roc_auc = float(roc_auc_score(y_true, scores))
            pr_auc  = float(average_precision_score(y_true, scores))
        except Exception as exc:
            logger.debug(f"AUC computation failed: {exc}")

    return MetricsResult(
        model_name=model_name,
        modality=modality,
        channel_name=channel_name,
        threshold_method=threshold_method,
        threshold_value=threshold_value,
        TP=TP, FP=FP, TN=TN, FN=FN,
        n_windows=n,
        precision=round(precision, 6),
        recall=round(recall, 6),
        f1=round(f1, 6),
        accuracy=round(accuracy, 6),
        fpr=round(fpr, 6),
        fnr=round(fnr, 6),
        roc_auc=round(roc_auc, 6) if not np.isnan(roc_auc) else float("nan"),
        pr_auc=round(pr_auc, 6)  if not np.isnan(pr_auc)  else float("nan"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation: re-threshold from raw reconstruction errors
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all_thresholds(
    score_df: pd.DataFrame,
    train_errors_map: Dict[str, Dict[str, np.ndarray]],
    methods: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Evaluate every (model × channel × threshold_method) combination.

    This function re-thresholds from the continuous `reconstruction_error`
    column in `score_df` — the existing `predicted_label` column is NOT used.

    Parameters
    ----------
    score_df : pd.DataFrame
        Concatenated score CSVs from one or more pipelines.
        Required columns: model_name, channel_name, modality,
                          reconstruction_error, is_synthetic_anomaly.
    train_errors_map : dict
        {model_display_name: {channel_name: np.ndarray of training errors}}
        e.g. {"LSTM": {"tensor_fascia_lata": np.array([...])}}
    methods : list of str, optional
        Threshold methods to apply. Default: all three.

    Returns
    -------
    pd.DataFrame
        One row per (model, modality, channel, threshold_method) with all metrics.
    """
    from evaluation_framework.thresholds import fit_all_thresholds

    if methods is None:
        methods = ["mean_std", "percentile95", "percentile99"]

    # Check required columns
    required = {"model_name", "channel_name", "modality",
                "reconstruction_error", "is_synthetic_anomaly"}
    missing = required - set(score_df.columns)
    if missing:
        raise ValueError(f"score_df missing required columns: {missing}")

    records: List[Dict[str, Any]] = []

    for model_name in score_df["model_name"].unique():
        model_df = score_df[score_df["model_name"] == model_name]
        train_ch_errors = train_errors_map.get(model_name, {})

        for channel_name in sorted(model_df["channel_name"].unique()):
            channel_df = model_df[model_df["channel_name"] == channel_name]
            if channel_df.empty:
                continue

            modality = channel_df["modality"].iloc[0]
            y_true   = channel_df["is_synthetic_anomaly"].values.astype(int)
            scores   = channel_df["reconstruction_error"].values.astype(float)

            # Get training errors for this channel
            train_errors = train_ch_errors.get(channel_name)

            if train_errors is None or len(train_errors) == 0:
                logger.warning(
                    f"  No training errors for [{model_name}][{channel_name}] — "
                    "computing rough fallback threshold from test scores."
                )
                # Fallback: use clean windows from test set as proxy
                clean_mask  = y_true == 0
                train_errors = scores[clean_mask] if clean_mask.sum() > 0 else scores

            # Fit all requested threshold methods
            threshold_results = fit_all_thresholds(train_errors, methods)

            for method, thresh_result in threshold_results.items():
                y_pred = (scores > thresh_result.value).astype(int)
                m = compute_metrics(
                    y_true=y_true,
                    y_pred=y_pred,
                    scores=scores,
                    model_name=model_name,
                    modality=modality,
                    channel_name=channel_name,
                    threshold_method=method,
                    threshold_value=thresh_result.value,
                )
                records.append(m.to_dict())

    return pd.DataFrame(records)


def aggregate_by_modality(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Macro-average metrics across all channels within each
    (model_name, modality, threshold_method) group.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Output of evaluate_all_thresholds().

    Returns
    -------
    pd.DataFrame
        One row per (model_name, modality, threshold_method).
    """
    numeric_cols = ["precision", "recall", "f1", "accuracy", "fpr", "fnr",
                    "roc_auc", "pr_auc", "threshold_value"]
    sum_cols     = ["TP", "FP", "TN", "FN", "n_windows"]
    group_keys   = ["model_name", "modality", "threshold_method"]

    agg_dict: Dict[str, Any] = {}
    for col in numeric_cols:
        if col in metrics_df.columns:
            agg_dict[col] = "mean"
    for col in sum_cols:
        if col in metrics_df.columns:
            agg_dict[col] = "sum"

    agg = metrics_df.groupby(group_keys, sort=True).agg(agg_dict).reset_index()

    # Add channel_name marker
    agg["channel_name"] = "ALL_CHANNELS"

    # Recompute F1 from pooled TP/FP/FN for consistency
    if all(c in agg.columns for c in ["TP", "FP", "FN"]):
        p = agg["TP"] / (agg["TP"] + agg["FP"]).replace(0, np.nan)
        r = agg["TP"] / (agg["TP"] + agg["FN"]).replace(0, np.nan)
        agg["f1_pooled"] = (2 * p * r / (p + r)).fillna(0.0)

    return agg


def build_comparison_dataframe(
    metrics_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a single wide comparison DataFrame suitable for table export.

    Combines per-channel and aggregate metrics, ordered by
    model_name → threshold_method → channel_name.
    """
    # Tag aggregate rows
    aggregate_df = aggregate_df.copy()

    # Concatenate
    combined = pd.concat([metrics_df, aggregate_df], ignore_index=True)

    # Canonical column order
    col_order = [
        "model_name", "modality", "channel_name", "threshold_method",
        "threshold_value",
        "precision", "recall", "f1", "accuracy", "fpr", "fnr",
        "roc_auc", "pr_auc",
        "TP", "FP", "TN", "FN", "n_windows",
    ]
    combined = combined[[c for c in col_order if c in combined.columns]]
    combined = combined.sort_values(
        ["model_name", "threshold_method", "channel_name"]
    ).reset_index(drop=True)

    return combined


def evaluate_fusion_strategies(
    fused_df: pd.DataFrame,
    model_name: str,
    threshold_method: str = "mean_std",
    threshold_value: float = float("nan"),
) -> pd.DataFrame:
    """
    Compute metrics for each fusion strategy (OR, MAJORITY, AND).

    Parameters
    ----------
    fused_df : pd.DataFrame
        Output of fusion._apply_fusion() with columns:
        ground_truth, fused_OR, fused_MAJORITY, fused_AND.
    model_name : str
        Model display name (e.g. 'LSTM').
    threshold_method : str
        Threshold method used to derive predictions.

    Returns
    -------
    pd.DataFrame
        One row per fusion strategy with all metrics.
    """
    y_true = fused_df["ground_truth"].values.astype(int)
    records = []

    for strategy in ["OR", "MAJORITY", "AND"]:
        col = f"fused_{strategy}"
        if col not in fused_df.columns:
            continue
        y_pred = fused_df[col].values.astype(int)

        # Use max reconstruction error as anomaly score for AUC
        score_col = "max_reconstruction_error"
        scores = fused_df[score_col].values if score_col in fused_df.columns else None

        m = compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            scores=scores,
            model_name=model_name,
            modality="Fusion",
            channel_name=f"fusion_{strategy}",
            threshold_method=threshold_method,
            threshold_value=threshold_value,
        )
        d = m.to_dict()
        d["fusion_strategy"] = strategy
        records.append(d)

    return pd.DataFrame(records)
