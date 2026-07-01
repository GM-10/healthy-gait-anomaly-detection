"""
semg_pipeline/evaluator.py

Post-hoc evaluation of anomaly detection performance on the test set.

Metrics (per model, per channel):
    - Recall         (primary)
    - F1 score       (secondary)
    - RMSE           of reconstruction error
    - TP, FP, TN, FN (confusion matrix counts)

An aggregate summary across all channels is also computed.
Results are saved as CSV files.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Required output CSV columns — validate before computing metrics
_REQUIRED_COLS = {
    "subject_id", "modality", "channel_name", "movement", "window_id",
    "window_start_time", "window_end_time", "reconstruction_error",
    "is_synthetic_anomaly", "anomaly_type", "predicted_label", "model_name",
}


# ─────────────────────────────────────────────────────────────────────────────
# Core metric computation
# ─────────────────────────────────────────────────────────────────────────────

def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """Compute TP, FP, TN, FN from binary arrays."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"TP": TP, "FP": FP, "TN": TN, "FN": FN}


def _recall(TP: int, FN: int) -> float:
    denom = TP + FN
    return TP / denom if denom > 0 else 0.0


def _precision(TP: int, FP: int) -> float:
    denom = TP + FP
    return TP / denom if denom > 0 else 0.0


def _f1(TP: int, FP: int, FN: int) -> float:
    p = _precision(TP, FP)
    r = _recall(TP, FN)
    denom = p + r
    return 2 * p * r / denom if denom > 0 else 0.0


def _rmse(errors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(errors ** 2)))


# ─────────────────────────────────────────────────────────────────────────────
# Per-model evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(score_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-channel evaluation metrics for a single model's test scores.

    Parameters
    ----------
    score_df : pd.DataFrame
        Concatenated output CSV rows for ONE model across all test
        subjects and movements. Must contain all columns in _REQUIRED_COLS.

    Returns
    -------
    pd.DataFrame
        One row per channel, columns:
            model_name, channel_name,
            recall, f1_score, rmse,
            TP, FP, TN, FN, n_windows
    """
    missing_cols = _REQUIRED_COLS - set(score_df.columns)
    if missing_cols:
        raise ValueError(f"score_df missing required columns: {missing_cols}")

    model_name = score_df["model_name"].iloc[0]
    records    = []

    for channel_name, group in score_df.groupby("channel_name"):
        y_true  = group["is_synthetic_anomaly"].values.astype(int)
        y_pred  = group["predicted_label"].values.astype(int)
        errors  = group["reconstruction_error"].values.astype(float)

        counts = _confusion_counts(y_true, y_pred)
        TP, FP, TN, FN = counts["TP"], counts["FP"], counts["TN"], counts["FN"]

        records.append(
            {
                "model_name":   model_name,
                "channel_name": channel_name,
                "recall":       round(_recall(TP, FN), 6),
                "f1_score":     round(_f1(TP, FP, FN), 6),
                "rmse":         round(_rmse(errors),   6),
                "TP":           TP,
                "FP":           FP,
                "TN":           TN,
                "FN":           FN,
                "n_windows":    len(group),
            }
        )

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate summary across channels
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_aggregate(score_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute macro-average metrics across all channels for a single model.

    Parameters
    ----------
    score_df : pd.DataFrame
        Same as evaluate_model input.

    Returns
    -------
    pd.DataFrame
        One-row DataFrame with aggregated metrics.
    """
    per_channel = evaluate_model(score_df)
    model_name  = score_df["model_name"].iloc[0]

    # Macro-average of per-channel metrics
    agg = {
        "model_name":          model_name,
        "channel_name":        "ALL_CHANNELS",
        "recall":              round(per_channel["recall"].mean(),   6),
        "f1_score":            round(per_channel["f1_score"].mean(), 6),
        "rmse":                round(per_channel["rmse"].mean(),     6),
        "TP":                  int(per_channel["TP"].sum()),
        "FP":                  int(per_channel["FP"].sum()),
        "TN":                  int(per_channel["TN"].sum()),
        "FN":                  int(per_channel["FN"].sum()),
        "n_windows":           int(per_channel["n_windows"].sum()),
    }
    return pd.DataFrame([agg])


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def save_evaluation_report(
    results: Dict[str, pd.DataFrame],
    output_dir: str,
    filename: str = "evaluation_summary.csv",
) -> str:
    """
    Concatenate per-model evaluation DataFrames and save as a single CSV.

    Parameters
    ----------
    results : Dict[str, pd.DataFrame]
        Keys: model names ('SARIMA', 'LSTM', 'Transformer').
        Values: DataFrames from evaluate_model() or evaluate_aggregate().
    output_dir : str
        Directory to write the CSV.
    filename : str, default 'evaluation_summary.csv'

    Returns
    -------
    str
        Absolute path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    combined = pd.concat(list(results.values()), ignore_index=True)

    # Reorder columns for readability
    col_order = [
        "model_name", "channel_name",
        "recall", "f1_score", "rmse",
        "TP", "FP", "TN", "FN", "n_windows",
    ]
    combined = combined[[c for c in col_order if c in combined.columns]]

    out_path = os.path.join(output_dir, filename)
    combined.to_csv(out_path, index=False)
    logger.info(f"[Evaluator] Saved evaluation report → {out_path}")
    return out_path


def print_evaluation_summary(results: Dict[str, pd.DataFrame]) -> None:
    """Pretty-print per-model per-channel results to stdout."""
    for model_name, df in results.items():
        print(f"\n{'='*60}")
        print(f"  Model: {model_name}")
        print(f"{'='*60}")
        print(
            df[["channel_name", "recall", "f1_score", "rmse", "TP", "FP", "TN", "FN"]]
            .to_string(index=False)
        )
