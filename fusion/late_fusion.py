"""
fusion/late_fusion.py

Late fusion of sEMG and Kinematics/Kinetics anomaly detection scores
for the SIAT-LLMD dataset.

Combines per-window, per-channel predicted_labels from both modalities
using three fusion strategies, then evaluates Recall and F1 against
single-modality baselines.

Fusion strategies
-----------------
  OR      — flag if ANY channel across EITHER modality predicts anomalous.
             Maximises recall; minimises false negatives. (Clinical priority)
  MAJORITY — flag if > 50% of ALL channels (both modalities combined) vote 1.
  AND      — flag only if BOTH modalities independently flag (≥1 channel each).

Usage
-----
    python -m fusion.late_fusion --output_dir outputs
    python -m fusion.late_fusion --output_dir outputs --model lstm
    python -m fusion.late_fusion --output_dir outputs --model transformer
    python -m fusion.late_fusion --output_dir /kaggle/working/outputs
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fusion")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MODALITY_SEMG   = "sEMG"
MODALITY_KINKIN = "Kinematics+Kinetics"

FUSION_STRATEGIES = ["OR", "MAJORITY", "AND"]

MODEL_DISPLAY = {
    "lstm":        "LSTM",
    "transformer": "Transformer",
}

# Columns that uniquely identify a window across both pipelines
WINDOW_KEY = ["subject_id", "movement", "window_id", "model_name"]

# Required columns in every input CSV
REQUIRED_COLS = {
    "subject_id", "modality", "channel_name", "movement", "window_id",
    "reconstruction_error", "is_synthetic_anomaly", "predicted_label",
    "model_name",
}

# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _recall(tp: int, fn: int) -> float:
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def _precision(tp: int, fp: int) -> float:
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def _f1(tp: int, fp: int, fn: int) -> float:
    p = _precision(tp, fp)
    r = _recall(tp, fn)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """Return (TP, FP, TN, FN)."""
    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())
    return TP, FP, TN, FN


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_subject_csv(
    sub_dir: str,
    model_display: str,
) -> pd.DataFrame:
    """
    Load and concatenate all *_[model_display]_scores.csv files for one subject.
    Returns an empty DataFrame if no files found.
    """
    dfs = []
    if not os.path.isdir(sub_dir):
        return pd.DataFrame()

    for fname in sorted(os.listdir(sub_dir)):
        if fname.endswith(f"_{model_display}_scores.csv"):
            path = os.path.join(sub_dir, fname)
            try:
                df = pd.read_csv(path)
                missing = REQUIRED_COLS - set(df.columns)
                if missing:
                    logger.warning(f"  Skipping {fname}: missing columns {missing}")
                    continue
                dfs.append(df)
            except Exception as exc:
                logger.warning(f"  Could not read {fname}: {exc}")

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_all_scores(
    output_dir: str,
    model_display: str,
) -> pd.DataFrame:
    """
    Walk output_dir/Sub##/ directories and load all score CSVs for
    the given model. Returns a single concatenated DataFrame.
    """
    all_dfs = []

    sub_dirs = sorted([
        d for d in os.listdir(output_dir)
        if d.startswith("Sub") and os.path.isdir(os.path.join(output_dir, d))
    ])

    if not sub_dirs:
        logger.error(f"No Sub## directories found in {output_dir}")
        return pd.DataFrame()

    for sub in sub_dirs:
        logger.info(f"  Loading {sub} / {model_display} …")
        sub_dir = os.path.join(output_dir, sub)
        df = _load_subject_csv(sub_dir, model_display)
        if not df.empty:
            all_dfs.append(df)
        else:
            logger.warning(f"  {sub}: no {model_display} score CSVs found — skipping.")

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(
        f"Loaded {len(combined):,} rows for model={model_display} "
        f"across {len(sub_dirs)} subjects."
    )
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Per-modality vote aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_modality_votes(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (subject_id, movement, window_id, model_name), aggregate
    all per-channel predicted_labels within one modality into:
      - n_channels    : total channel count
      - n_votes_1     : channels that predicted anomalous
      - modality_vote : 1 if ANY channel flagged, else 0  (used for OR and AND)
      - ground_truth  : is_synthetic_anomaly (taken from first row of group)

    Returns one row per (subject_id, movement, window_id, model_name, modality).
    """
    def agg(group: pd.DataFrame) -> pd.Series:
        votes  = group["predicted_label"].values.astype(int)
        gt     = int(group["is_synthetic_anomaly"].iloc[0])
        return pd.Series({
            "n_channels":    len(votes),
            "n_votes_1":     int(votes.sum()),
            "modality_vote": int(votes.sum() > 0),
            "ground_truth":  gt,
        })

    key = WINDOW_KEY + ["modality"]
    return (
        df.groupby(key, sort=False)
          .apply(agg)
          .reset_index()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single-modality baseline recall
# ─────────────────────────────────────────────────────────────────────────────

def _single_modality_metrics(
    agg_df: pd.DataFrame,
    modality: str,
    model_display: str,
) -> Dict[str, float]:
    """
    Compute per-window recall for a single modality using the OR rule
    (modality_vote == 1 means at least one channel flagged).
    This gives the single-modality baseline used in the comparison table.
    """
    subset = agg_df[
        (agg_df["modality"]    == modality) &
        (agg_df["model_name"]  == model_display)
    ].copy()

    if subset.empty:
        return {"recall": float("nan"), "f1": float("nan"),
                "TP": 0, "FP": 0, "TN": 0, "FN": 0, "n": 0}

    y_true = subset["ground_truth"].values.astype(int)
    y_pred = subset["modality_vote"].values.astype(int)
    TP, FP, TN, FN = _confusion(y_true, y_pred)
    return {
        "recall": _recall(TP, FN),
        "f1":     _f1(TP, FP, FN),
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "n": len(subset),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fusion
# ─────────────────────────────────────────────────────────────────────────────

def _apply_fusion(
    agg_df: pd.DataFrame,
    model_display: str,
) -> pd.DataFrame:
    """
    For each (subject_id, movement, window_id, model_name) window present
    in BOTH modalities, apply all three fusion strategies.

    Returns a DataFrame with columns:
        subject_id, movement, window_id, model_name,
        semg_vote, kinkin_vote,
        n_channels_semg, n_votes_1_semg,
        n_channels_kinkin, n_votes_1_kinkin,
        total_channels, total_votes_1,
        fused_OR, fused_MAJORITY, fused_AND,
        ground_truth
    """
    model_df = agg_df[agg_df["model_name"] == model_display].copy()

    semg_df   = model_df[model_df["modality"] == MODALITY_SEMG].copy()
    kinkin_df = model_df[model_df["modality"] == MODALITY_KINKIN].copy()

    if semg_df.empty or kinkin_df.empty:
        logger.warning(
            f"  [{model_display}] One or both modalities are empty — "
            f"sEMG rows={len(semg_df)}, Kin/Kin rows={len(kinkin_df)}. "
            "Skipping fusion."
        )
        return pd.DataFrame()

    join_key = ["subject_id", "movement", "window_id"]

    merged = pd.merge(
        semg_df.rename(columns={
            "n_channels":    "n_channels_semg",
            "n_votes_1":     "n_votes_1_semg",
            "modality_vote": "semg_vote",
            "ground_truth":  "ground_truth",
        })[join_key + [
            "n_channels_semg", "n_votes_1_semg", "semg_vote", "ground_truth"
        ]],
        kinkin_df.rename(columns={
            "n_channels":    "n_channels_kinkin",
            "n_votes_1":     "n_votes_1_kinkin",
            "modality_vote": "kinkin_vote",
        })[join_key + [
            "n_channels_kinkin", "n_votes_1_kinkin", "kinkin_vote"
        ]],
        on=join_key,
        how="inner",   # only windows present in BOTH modalities
    )

    if merged.empty:
        logger.warning(
            f"  [{model_display}] No windows matched between modalities "
            "(check subject_id / window_id alignment)."
        )
        return pd.DataFrame()

    # Totals across both modalities
    merged["total_channels"] = merged["n_channels_semg"] + merged["n_channels_kinkin"]
    merged["total_votes_1"]  = merged["n_votes_1_semg"]  + merged["n_votes_1_kinkin"]

    # ── Strategy 1: OR ────────────────────────────────────────────────────
    merged["fused_OR"] = (
        (merged["semg_vote"] == 1) | (merged["kinkin_vote"] == 1)
    ).astype(int)

    # ── Strategy 2: MAJORITY ──────────────────────────────────────────────
    merged["fused_MAJORITY"] = (
        merged["total_votes_1"] > (merged["total_channels"] / 2)
    ).astype(int)

    # ── Strategy 3: AND ───────────────────────────────────────────────────
    merged["fused_AND"] = (
        (merged["semg_vote"] == 1) & (merged["kinkin_vote"] == 1)
    ).astype(int)

    merged["model_name"] = model_display
    logger.info(
        f"  [{model_display}] Fused {len(merged):,} matched windows "
        f"({merged['ground_truth'].sum():,} synthetic anomaly windows)."
    )
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation per fusion strategy
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_fusion(
    fusion_df: pd.DataFrame,
    model_display: str,
    semg_baseline: Dict[str, float],
    kinkin_baseline: Dict[str, float],
) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Compute Recall, F1, TP, FP, TN, FN for each fusion strategy.

    Returns
    -------
    summary_rows : List[Dict]  — one row per strategy for the summary table
    detail_df    : pd.DataFrame — row-level fusion decisions for saving
    """
    y_true = fusion_df["ground_truth"].values.astype(int)
    summary_rows = []

    for strategy in FUSION_STRATEGIES:
        col    = f"fused_{strategy}"
        y_pred = fusion_df[col].values.astype(int)
        TP, FP, TN, FN = _confusion(y_true, y_pred)

        summary_rows.append({
            "model_name":       model_display,
            "fusion_strategy":  strategy,
            "semg_recall":      round(semg_baseline.get("recall", float("nan")), 4),
            "kinkin_recall":    round(kinkin_baseline.get("recall", float("nan")), 4),
            "fused_recall":     round(_recall(TP, FN), 4),
            "fused_f1":         round(_f1(TP, FP, FN), 4),
            "fused_precision":  round(_precision(TP, FP), 4),
            "TP": TP, "FP": FP, "TN": TN, "FN": FN,
            "n_windows":        len(fusion_df),
        })

    # Build row-level detail DataFrame (one row per window per strategy)
    detail_rows = []
    for strategy in FUSION_STRATEGIES:
        col = f"fused_{strategy}"
        strategy_df = fusion_df[
            ["subject_id", "movement", "window_id", "model_name",
             "semg_vote", "kinkin_vote", "ground_truth"]
        ].copy()
        strategy_df["fusion_strategy"]  = strategy
        strategy_df["fused_prediction"] = fusion_df[col].values
        strategy_df["is_correct"]       = (
            fusion_df[col].values == fusion_df["ground_truth"].values
        ).astype(int)
        detail_rows.append(strategy_df)

    detail_df = pd.concat(detail_rows, ignore_index=True)
    return summary_rows, detail_df


# ─────────────────────────────────────────────────────────────────────────────
# Printing
# ─────────────────────────────────────────────────────────────────────────────

def _print_comparison_table(summary_rows: List[Dict]) -> None:
    """Print a formatted comparison table to stdout."""
    header = (
        f"{'Model':<14} {'Strategy':<10} "
        f"{'sEMG Recall':>12} {'Kin/Kin Recall':>15} "
        f"{'Fused Recall':>13} {'Fused F1':>10}"
    )
    sep = "─" * len(header)

    print(f"\n{sep}")
    print("  LATE FUSION — COMPARISON TABLE")
    print(sep)
    print(header)
    print(sep)

    for row in summary_rows:
        semg_r   = f"{row['semg_recall']:.4f}"   if not _is_nan(row['semg_recall'])   else "  N/A  "
        kinkin_r = f"{row['kinkin_recall']:.4f}"  if not _is_nan(row['kinkin_recall'])  else "  N/A  "
        fused_r  = f"{row['fused_recall']:.4f}"
        fused_f1 = f"{row['fused_f1']:.4f}"

        print(
            f"{row['model_name']:<14} {row['fusion_strategy']:<10} "
            f"{semg_r:>12} {kinkin_r:>15} "
            f"{fused_r:>13} {fused_f1:>10}"
        )

    print(sep)


def _is_nan(v) -> bool:
    try:
        return v != v  # NaN check without importing math
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Late fusion of sEMG and Kinematics/Kinetics anomaly scores"
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join("outputs"),
        help="Root outputs directory containing Sub## subdirectories (default: outputs/)",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=["lstm", "transformer"],
        choices=["lstm", "transformer"],
        help="Models to fuse (default: lstm transformer)",
    )
    return parser.parse_args()


def main() -> None:
    args      = parse_args()
    out_root  = args.output_dir
    models    = args.model
    fusion_dir = os.path.join(out_root, "fusion")
    os.makedirs(fusion_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  Late Fusion — sEMG ⊕ Kinematics/Kinetics")
    logger.info("=" * 60)
    logger.info(f"  output_dir : {out_root}")
    logger.info(f"  models     : {models}")

    all_summary_rows: List[Dict] = []

    for model_key in models:
        model_display = MODEL_DISPLAY[model_key]
        logger.info(f"\n{'─'*55}")
        logger.info(f"  Model: {model_display}")
        logger.info(f"{'─'*55}")

        # ── 1. Load all score CSVs ─────────────────────────────────────────
        logger.info(f"[Load] Reading {model_display} score CSVs …")
        raw_df = load_all_scores(out_root, model_display)

        if raw_df.empty:
            logger.warning(f"  No data found for {model_display}. Skipping.")
            continue

        # Validate modality values present
        present_modalities = raw_df["modality"].unique().tolist()
        logger.info(f"  Modalities present: {present_modalities}")

        if MODALITY_SEMG not in present_modalities:
            logger.warning(f"  '{MODALITY_SEMG}' not found — cannot fuse.")
            continue
        if MODALITY_KINKIN not in present_modalities:
            logger.warning(f"  '{MODALITY_KINKIN}' not found — cannot fuse.")
            continue

        # ── 2. Aggregate per-modality votes per window ─────────────────────
        logger.info("[Aggregate] Computing per-window modality votes …")
        agg_df = _aggregate_modality_votes(raw_df)

        # ── 3. Single-modality baselines ───────────────────────────────────
        semg_baseline   = _single_modality_metrics(agg_df, MODALITY_SEMG,   model_display)
        kinkin_baseline = _single_modality_metrics(agg_df, MODALITY_KINKIN, model_display)

        logger.info(
            f"  Baseline — sEMG: recall={semg_baseline['recall']:.4f}  "
            f"Kin/Kin: recall={kinkin_baseline['recall']:.4f}"
        )

        # ── 4. Apply fusion strategies ─────────────────────────────────────
        logger.info("[Fusion] Applying fusion strategies …")
        fusion_df = _apply_fusion(agg_df, model_display)

        if fusion_df.empty:
            continue

        # ── 5. Evaluate ────────────────────────────────────────────────────
        summary_rows, detail_df = _evaluate_fusion(
            fusion_df, model_display, semg_baseline, kinkin_baseline
        )
        all_summary_rows.extend(summary_rows)

        # ── 6. Save detail CSV ─────────────────────────────────────────────
        detail_col_order = [
            "subject_id", "movement", "window_id", "model_name",
            "fusion_strategy", "semg_vote", "kinkin_vote",
            "fused_prediction", "ground_truth", "is_correct",
        ]
        detail_df = detail_df[[c for c in detail_col_order if c in detail_df.columns]]
        detail_path = os.path.join(fusion_dir, f"fusion_results_{model_display}.csv")
        detail_df.to_csv(detail_path, index=False)
        logger.info(f"  Detail results saved → {detail_path}")

    # ── 7. Print comparison table ──────────────────────────────────────────
    if all_summary_rows:
        _print_comparison_table(all_summary_rows)

        # ── 8. Save summary CSV ────────────────────────────────────────────
        summary_df = pd.DataFrame(all_summary_rows)
        summary_col_order = [
            "model_name", "fusion_strategy",
            "semg_recall", "kinkin_recall",
            "fused_recall", "fused_f1", "fused_precision",
            "TP", "FP", "TN", "FN", "n_windows",
        ]
        summary_df = summary_df[
            [c for c in summary_col_order if c in summary_df.columns]
        ]
        summary_path = os.path.join(fusion_dir, "fusion_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"\nSummary saved → {summary_path}")
    else:
        logger.warning("No fusion results produced. Check that output CSVs exist.")


if __name__ == "__main__":
    main()
