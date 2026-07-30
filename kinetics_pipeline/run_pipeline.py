"""
kinetics_pipeline/run_pipeline.py

Master orchestration script for the Kinematics + Kinetics anomaly detection pipeline.

Runs the complete pipeline end-to-end:
    1. Load all training trials and fit Min-Max scaler using SIATGaitDataset
    2. Segment into windows (180 samples, 50% overlap)
    3. Train SARIMA / LSTM / Transformer models (one per channel)
    4. Compute per-channel thresholds from training reconstruction errors
    5. Process validation set (early stopping for LSTM & Transformer)
    6. Process test set: score clean + synthetically injected windows
    7. Save output CSVs per subject per movement per model
    8. Run evaluation: Recall, F1, RMSE, confusion matrix
    9. Save aggregate evaluation summary
"""

import argparse
import logging
import os
import sys
import time
import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure repo root is on sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Imports from preprocessing
from preprocessing.loader import SIGNAL_COLUMNS
from preprocessing.dataset import SIATGaitDataset

# Imports from kinetics_pipeline
from kinetics_pipeline.anomaly_scorer import (
    compute_threshold,
    label_windows,
    build_output_rows,
    score_and_build_rows,
    save_train_errors,
    load_train_errors,
)
from kinetics_pipeline.evaluator import (
    evaluate_model,
    evaluate_aggregate,
    save_evaluation_report,
    print_evaluation_summary,
)

# Shared synthetic anomaly injection
from utils.synthetic_anomalies import (
    inject_amplitude_scale,
    inject_time_warp,
    inject_time_shift,
    inject_combined,
    DEFAULT_SEVERITIES,
)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("kinetics_pipeline")

# Constants
WINDOW_SIZE  = 100   # ~0.83 seconds at 120 Hz
OVERLAP_SIZE = 50    # 50% overlap
TARGET_FS    = 120.0

MOVEMENTS = [
    "WAK", "UPS", "DNS", "HS", "KLCL", "KLFT", "LLB", "LLF",
    "LLS", "LUGB", "LUGF", "SITDN", "STC", "STDUP", "TO", "TPTO"
]

TRAIN_SUBS = [f"Sub{i:02d}" for i in range(1, 31)]
VAL_SUBS   = [f"Sub{i:02d}" for i in range(31, 36)]
TEST_SUBS  = [f"Sub{i:02d}" for i in range(36, 41)]

MODEL_REGISTRY = {
    "sarima":       "SARIMA",
    "lstm":         "LSTM",
    "transformer":  "Transformer",
}


def _build_anomaly_conditions(
    severity_levels: List[str],
) -> List[Tuple]:
    """
    Build the list of (inject_fn, kwargs, atype, level) tuples for the
    requested severity levels.

    Parameters
    ----------
    severity_levels : list of str
        Subset of DEFAULT_SEVERITIES keys to include, e.g. ['mild', 'moderate'].

    Returns
    -------
    List of tuples — one per (inject_fn, severity) combination, plus one
    combined-anomaly entry at moderate severity.
    """
    inject_fns = [
        (inject_amplitude_scale, "amplitude_scale"),
        (inject_time_warp,       "time_warp"),
        (inject_time_shift,      "time_shift"),
    ]
    conditions: List[Tuple] = []
    for fn, atype in inject_fns:
        for level in severity_levels:
            sev = DEFAULT_SEVERITIES[level]
            conditions.append((fn, {"severity": sev}, atype, level))
    # Combined anomaly always uses moderate regardless of selected levels
    conditions.append((inject_combined, {}, "combined", "moderate"))
    return conditions


# Scaler serialization helpers
def save_scaler(params: Dict[str, Tuple[float, float]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    serializable = {ch: list(v) for ch, v in params.items()}
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def load_scaler(path: str) -> Dict[str, Tuple[float, float]]:
    with open(path, "r") as f:
        raw = json.load(f)
    return {ch: tuple(v) for ch, v in raw.items()}


# Training helpers
def _train_sarima(
    train_windows: np.ndarray,
    model_dir: str,
    sarima_train_subs: int,
) -> "SARIMAModel":
    from kinetics_pipeline.models.sarima_model import SARIMAModel, SARIMA_MAX_TRAIN_SUBJECTS

    max_subs = sarima_train_subs or SARIMA_MAX_TRAIN_SUBJECTS
    logger.info(
        f"[SARIMA] Training on up to {max_subs} subjects' windows "
        f"({len(train_windows)} total train windows available)."
    )
    model = SARIMAModel(
        channel_names=SIGNAL_COLUMNS,
        max_windows_per_channel=200,
        model_dir=os.path.join(model_dir, "sarima"),
    )
    model.fit(train_windows)
    return model


def _train_lstm(
    train_windows: np.ndarray,
    val_windows: np.ndarray,
    model_dir: str,
) -> List:
    from kinetics_pipeline.models.lstm_model import LSTMModel

    models = []
    for ch_idx, ch_name in enumerate(SIGNAL_COLUMNS):
        logger.info(f"[LSTM] Training channel {ch_idx + 1}/16: {ch_name}")
        m = LSTMModel(channel_name=ch_name, window_size=WINDOW_SIZE)
        train_ch = train_windows[:, :, ch_idx : ch_idx + 1]
        val_ch   = val_windows[:, :, ch_idx : ch_idx + 1] if len(val_windows) > 0 else None
        m.fit(train_ch, val_ch)
        save_path = os.path.join(model_dir, "lstm", f"lstm_ch{ch_idx:02d}.pt")
        m.save(save_path)
        models.append(m)
    return models


def _train_transformer(
    train_windows: np.ndarray,
    val_windows: np.ndarray,
    model_dir: str,
) -> List:
    from kinetics_pipeline.models.transformer_model import TransformerModel

    models = []
    for ch_idx, ch_name in enumerate(SIGNAL_COLUMNS):
        logger.info(f"[Transformer] Training channel {ch_idx + 1}/16: {ch_name}")
        m = TransformerModel(channel_name=ch_name, window_size=WINDOW_SIZE)
        train_ch = train_windows[:, :, ch_idx : ch_idx + 1]
        val_ch   = val_windows[:, :, ch_idx : ch_idx + 1] if len(val_windows) > 0 else None
        m.fit(train_ch, val_ch)
        save_path = os.path.join(model_dir, "transformer", f"transformer_ch{ch_idx:02d}.pt")
        m.save(save_path)
        models.append(m)
    return models


# Model loading helpers
def _load_lstm_models(model_dir: str) -> List:
    from kinetics_pipeline.models.lstm_model import LSTMModel
    models = []
    for ch_idx, ch_name in enumerate(SIGNAL_COLUMNS):
        m = LSTMModel(channel_name=ch_name, window_size=WINDOW_SIZE)
        path = os.path.join(model_dir, "lstm", f"lstm_ch{ch_idx:02d}.pt")
        if os.path.exists(path):
            m.load(path)
        else:
            logger.warning(f"[LSTM] Weight file not found: {path}")
        models.append(m)
    return models


def _load_transformer_models(model_dir: str) -> List:
    from kinetics_pipeline.models.transformer_model import TransformerModel
    models = []
    for ch_idx, ch_name in enumerate(SIGNAL_COLUMNS):
        m = TransformerModel(channel_name=ch_name, window_size=WINDOW_SIZE)
        path = os.path.join(model_dir, "transformer", f"transformer_ch{ch_idx:02d}.pt")
        if os.path.exists(path):
            m.load(path)
        else:
            logger.warning(f"[Transformer] Weight file not found: {path}")
        models.append(m)
    return models


def _load_sarima_models(model_dir: str) -> "SARIMAModel":
    from kinetics_pipeline.models.sarima_model import SARIMAModel
    m = SARIMAModel(
        channel_names=SIGNAL_COLUMNS,
        model_dir=os.path.join(model_dir, "sarima"),
    )
    m.load(os.path.join(model_dir, "sarima"))
    return m


# Threshold computing helper
def compute_train_thresholds(
    active_models: Dict[str, object],
    train_windows: np.ndarray,
    output_dir: Optional[str] = None,
    save_errors: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-model per-channel anomaly thresholds from training windows.

    Optionally saves training errors as .npy files for use by evaluate.py
    when computing percentile thresholds (--save_train_errors flag).

    Returns
    -------
    thresholds[model_key][channel_name] = float threshold
    """
    thresholds: Dict[str, Dict[str, float]] = {}

    for model_key, model_or_list in active_models.items():
        model_display = MODEL_REGISTRY[model_key]
        logger.info(f"[Threshold] Computing thresholds for {model_display} …")
        ch_thresholds = {}

        for ch_idx, ch_name in enumerate(SIGNAL_COLUMNS):
            if model_key == "sarima":
                from kinetics_pipeline.models.sarima_model import SARIMAModel
                errors = model_or_list.score(train_windows)[:, ch_idx]
            else:
                ch_windows = train_windows[:, :, ch_idx : ch_idx + 1]
                errors     = model_or_list[ch_idx].score(ch_windows)

            ch_thresholds[ch_name] = compute_threshold(errors)
            logger.info(
                f"  {ch_name}: threshold = {ch_thresholds[ch_name]:.6f} "
                f"(mean={np.mean(errors):.6f}, std={np.std(errors):.6f})"
            )

            # Persist training errors for evaluate.py multi-threshold support
            if save_errors and output_dir is not None:
                save_train_errors(errors, ch_name, model_display, output_dir)

        thresholds[model_key] = ch_thresholds

    return thresholds


# Scoring test helper
def score_test_trial(
    base_dir: str,
    subject: str,
    movement: str,
    scaling_params: Dict,
    active_models: Dict[str, object],
    thresholds: Dict[str, Dict[str, float]],
    output_dir: str,
    anomaly_conditions: List[Tuple],
    severity_tag: str = "",
) -> None:
    """
    Score one test trial for all active models.

    Parameters
    ----------
    anomaly_conditions : list of (inject_fn, kwargs, atype, severity_label)
        Built by _build_anomaly_conditions() for the requested severity levels.
    severity_tag : str
        Inserted into the output filename before '_scores.csv' when non-empty.
        e.g. 'mild' -> Sub36_WAK_LSTM_mild_scores.csv
        Empty string -> Sub36_WAK_LSTM_scores.csv  (default / backward-compat)
    """
    dataset = SIATGaitDataset(
        base_dir=base_dir,
        subjects=[subject],
        movements=[movement],
        window_size=WINDOW_SIZE,
        overlap_size=OVERLAP_SIZE,
        target_fs=TARGET_FS,
        scaling_params=scaling_params,
        fit_scaling=False
    )
    windows = dataset.windows_data
    meta = dataset.windows_metadata

    if len(windows) == 0:
        logger.warning(f"  [test] {subject}/{movement}: no windows — skipping.")
        return

    n_windows = len(windows)
    sub_out_dir = os.path.join(output_dir, subject)
    os.makedirs(sub_out_dir, exist_ok=True)

    for model_key, model_or_list in active_models.items():
        model_display = MODEL_REGISTRY[model_key]
        all_rows: List[Dict] = []

        # ── A) Clean windows ─────────────────────────────────────────────────
        for ch_idx, ch_name in enumerate(SIGNAL_COLUMNS):
            th = thresholds[model_key][ch_name]

            if model_key == "sarima":
                errors = model_or_list.score(windows)[:, ch_idx]
            else:
                ch_win = windows[:, :, ch_idx : ch_idx + 1]
                errors = model_or_list[ch_idx].score(ch_win)

            preds = label_windows(errors, th)
            rows  = build_output_rows(
                meta, errors, preds, ch_name, subject, movement,
                model_display,
                is_synthetic_anomaly=0, anomaly_type="none",
                window_id_offset=0,
            )
            all_rows.extend(rows)

        # ── B) Anomalous windows ─────────────────────────────────────────────
        for cond_idx, (inject_fn, inject_kwargs, atype, severity_label) in enumerate(anomaly_conditions):
            anom_windows = np.empty_like(windows)
            for w_idx in range(n_windows):
                for ch_idx in range(len(SIGNAL_COLUMNS)):
                    sig = windows[w_idx, :, ch_idx]
                    anom_sig, _ = inject_fn(sig, **inject_kwargs)
                    anom_windows[w_idx, :, ch_idx] = anom_sig

            window_id_offset = n_windows * (cond_idx + 1)

            for ch_idx, ch_name in enumerate(SIGNAL_COLUMNS):
                th = thresholds[model_key][ch_name]

                if model_key == "sarima":
                    errors = model_or_list.score(anom_windows)[:, ch_idx]
                else:
                    ch_win = anom_windows[:, :, ch_idx : ch_idx + 1]
                    errors = model_or_list[ch_idx].score(ch_win)

                preds = label_windows(errors, th)
                rows  = build_output_rows(
                    meta, errors, preds, ch_name, subject, movement,
                    model_display,
                    is_synthetic_anomaly=1,
                    anomaly_type=atype,
                    window_id_offset=window_id_offset,
                )
                all_rows.extend(rows)

        # ── Save output CSV ───────────────────────────────────────────────────
        sev_part = f"_{severity_tag}" if severity_tag else ""
        out_path = os.path.join(
            sub_out_dir, f"{subject}_{movement}_{model_display}{sev_part}_scores.csv"
        )
        out_df = pd.DataFrame(all_rows)

        col_order = [
            "subject_id", "modality", "channel_name", "movement", "window_id",
            "window_start_time", "window_end_time", "reconstruction_error",
            "is_synthetic_anomaly", "anomaly_type", "predicted_label", "model_name",
        ]
        out_df = out_df[col_order]
        out_df.to_csv(out_path, index=False)
        logger.info(
            f"  [{model_display}] {subject}/{movement} → {out_path}  "
            f"({len(out_df)} rows)"
        )


def run_evaluation(output_dir: str, active_model_keys: List[str], test_subs: List[str]) -> None:
    logger.info("\n[Evaluation] Loading test score CSVs …")
    all_results: Dict[str, pd.DataFrame] = {}

    for model_key in active_model_keys:
        model_display = MODEL_REGISTRY[model_key]
        dfs = []

        for sub in test_subs:
            sub_dir = os.path.join(output_dir, sub)
            if not os.path.isdir(sub_dir):
                continue
            for f in os.listdir(sub_dir):
                if f.endswith(f"_{model_display}_scores.csv"):
                    dfs.append(pd.read_csv(os.path.join(sub_dir, f)))

        if not dfs:
            logger.warning(f"[Evaluation] No CSVs found for {model_display}.")
            continue

        score_df = pd.concat(dfs, ignore_index=True)

        per_channel = evaluate_model(score_df)
        aggregate   = evaluate_aggregate(score_df)

        combined = pd.concat([per_channel, aggregate], ignore_index=True)
        all_results[model_display] = combined

        print_evaluation_summary({model_display: per_channel})

    summary_path = save_evaluation_report(all_results, output_dir)
    logger.info(f"\n[Evaluation] Summary saved → {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kinematics + Kinetics anomaly detection pipeline (SIAT-LLMD)"
    )
    parser.add_argument(
        "--base_dir",
        default=os.path.join("SIAT_LLMD20230404", "SIAT_LLMD20230404"),
        help="Root path of the SIAT-LLMD dataset",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join("outputs", "kinetics"),
        help="Root output directory",
    )
    parser.add_argument(
        "--movements",
        nargs="+",
        default=MOVEMENTS,
        help="Movement codes to process",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lstm", "transformer", "sarima"],
        choices=list(MODEL_REGISTRY.keys()),
        help="Models to run",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training; load saved model weights instead.",
    )
    parser.add_argument(
        "--skip_scoring",
        action="store_true",
        help="Skip scoring; only run evaluation on existing output CSVs.",
    )
    parser.add_argument(
        "--sarima_max_subjects",
        type=int,
        default=5,
        help="Number of train subjects to use for SARIMA fitting",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Process reduced subject list for smoke-testing.",
    )
    parser.add_argument(
        "--severities",
        nargs="+",
        default=["moderate"],
        choices=["mild", "moderate", "severe", "all"],
        help=(
            "Severity levels for synthetic anomaly injection. "
            "Use 'all' to run mild (0.15), moderate (0.35), and severe (0.60). "
            "Default: moderate only (backward-compatible)."
        ),
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Custom list of subjects for train/val/test splits",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=100,
        help="Window size in frames",
    )
    parser.add_argument(
        "--overlap_size",
        type=int,
        default=50,
        help="Overlap size in frames",
    )
    parser.add_argument(
        "--save_train_errors",
        action="store_true",
        help=(
            "Save per-channel training reconstruction errors as .npy files. "
            "Required by evaluate.py to compute percentile thresholds without re-inference. "
            "Files saved to output_dir/train_errors/."
        ),
    )
    parser.add_argument(
        "--threshold_method",
        default="mean_std",
        choices=["mean_std", "percentile95", "percentile99"],
        help=(
            "Thresholding method for scoring test windows. "
            "'mean_std' = mu + 3*sigma (default, backward-compatible). "
            "Note: evaluate.py re-computes all three methods from saved train errors."
        ),
    )
    return parser.parse_args()


def main() -> None:
    global WINDOW_SIZE, OVERLAP_SIZE
    args = parse_args()
    t0   = time.time()

    if args.window_size is not None:
        WINDOW_SIZE = args.window_size
    if args.overlap_size is not None:
        OVERLAP_SIZE = args.overlap_size

    logger.info("=" * 65)
    logger.info("  Kinematics + Kinetics Anomaly Detection Pipeline")
    logger.info("=" * 65)
    logger.info(f"  base_dir   : {args.base_dir}")
    logger.info(f"  output_dir : {args.output_dir}")
    logger.info(f"  movements  : {args.movements}")
    logger.info(f"  models     : {args.models}")
    logger.info(f"  severities : {args.severities}")
    logger.info(f"  dry_run    : {args.dry_run}")
    logger.info(f"  window_size: {WINDOW_SIZE}")
    logger.info(f"  overlap    : {OVERLAP_SIZE}")

    model_dir = os.path.join(args.output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    if args.subjects is not None:
        train_subs = args.subjects
        val_subs   = args.subjects
        test_subs  = args.subjects
        logger.info(f"Using custom subject list: {args.subjects}")
    elif args.dry_run:
        train_subs = ["Sub01", "Sub02"]
        val_subs   = ["Sub31"]
        test_subs  = ["Sub36"]
        logger.info("[DRY RUN] Using reduced subject lists.")
    else:
        train_subs = TRAIN_SUBS
        val_subs   = VAL_SUBS
        test_subs  = TEST_SUBS

    scaler_path = os.path.join(args.output_dir, "scaler_params.json")

    # ── Resolve severity levels ────────────────────────────────────────────
    requested_sevs = args.severities
    if "all" in requested_sevs:
        severity_levels = ["mild", "moderate", "severe"]
    else:
        severity_levels = requested_sevs
    severity_tag = severity_levels[0] if severity_levels != ["moderate"] else ""
    anomaly_conditions = _build_anomaly_conditions(severity_levels)
    logger.info(
        f"[Severities] Running: {severity_levels}  "
        f"→ {len(anomaly_conditions)} anomaly conditions per window  "
        f"(filename tag: '{severity_tag or 'none (moderate default)'}')"
    )

    # ══════════════════════════════════════════════════════════════════════
    # TRAIN PHASE
    # ══════════════════════════════════════════════════════════════════════
    if not args.skip_scoring:
        # ── 1. Fit/Load scaler using SIATGaitDataset ──
        if os.path.exists(scaler_path) and args.skip_training:
            logger.info(f"[Scaler] Loading existing scaler from {scaler_path}")
            scaling_params = load_scaler(scaler_path)
        else:
            logger.info("[Scaler] Fitting scaler on train subjects …")
            train_dataset = SIATGaitDataset(
                base_dir=args.base_dir,
                subjects=train_subs,
                movements=args.movements,
                window_size=WINDOW_SIZE,
                overlap_size=OVERLAP_SIZE,
                target_fs=TARGET_FS,
                fit_scaling=True
            )
            scaling_params = train_dataset.scaling_params
            save_scaler(scaling_params, scaler_path)
            logger.info(f"[Scaler] Saved scaler params → {scaler_path}")

        # ── 2. Collect train + val windows ──
        logger.info("\n[Windows] Collecting training windows …")
        train_dataset = SIATGaitDataset(
            base_dir=args.base_dir,
            subjects=train_subs,
            movements=args.movements,
            window_size=WINDOW_SIZE,
            overlap_size=OVERLAP_SIZE,
            target_fs=TARGET_FS,
            scaling_params=scaling_params,
            fit_scaling=False
        )
        train_windows = train_dataset.windows_data
        logger.info(f"  Train windows: {len(train_windows)}")

        logger.info("[Windows] Collecting validation windows …")
        val_dataset = SIATGaitDataset(
            base_dir=args.base_dir,
            subjects=val_subs,
            movements=args.movements,
            window_size=WINDOW_SIZE,
            overlap_size=OVERLAP_SIZE,
            target_fs=TARGET_FS,
            scaling_params=scaling_params,
            fit_scaling=False
        )
        val_windows = val_dataset.windows_data
        logger.info(f"  Val windows: {len(val_windows)}")

        # ── 3. Train or load models ──
        active_models: Dict[str, object] = {}

        for model_key in args.models:
            logger.info(f"\n{'─'*55}")
            logger.info(f"  Model: {MODEL_REGISTRY[model_key]}")
            logger.info(f"{'─'*55}")

            if args.skip_training:
                logger.info(f"  Loading saved {MODEL_REGISTRY[model_key]} weights …")
                if model_key == "lstm":
                    active_models["lstm"] = _load_lstm_models(model_dir)
                elif model_key == "transformer":
                    active_models["transformer"] = _load_transformer_models(model_dir)
                elif model_key == "sarima":
                    active_models["sarima"] = _load_sarima_models(model_dir)
            else:
                if model_key == "lstm":
                    active_models["lstm"] = _train_lstm(
                        train_windows, val_windows, model_dir
                    )
                elif model_key == "transformer":
                    active_models["transformer"] = _train_transformer(
                        train_windows, val_windows, model_dir
                    )
                elif model_key == "sarima":
                    active_models["sarima"] = _train_sarima(
                        train_windows, model_dir, args.sarima_max_subjects
                    )

        # ── 4. Compute thresholds on train set ──
        logger.info("\n[Thresholds] Computing anomaly thresholds from train errors …")
        thresholds = compute_train_thresholds(
            active_models, train_windows,
            output_dir=args.output_dir,
            save_errors=getattr(args, "save_train_errors", False),
        )

        # ── 5. Score test set ──
        logger.info("\n[Test] Scoring test subjects …")
        n_test = len(test_subs) * len(args.movements)
        done = 0
        for sub in test_subs:
            for mov in args.movements:
                done += 1
                logger.info(f"  [{done}/{n_test}] {sub}/{mov}")
                score_test_trial(
                    args.base_dir, sub, mov,
                    scaling_params,
                    active_models,
                    thresholds,
                    args.output_dir,
                    anomaly_conditions=anomaly_conditions,
                    severity_tag=severity_tag,
                )

    # ══════════════════════════════════════════════════════════════════════
    # EVALUATION PHASE
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n[Evaluation] Running metrics …")
    run_evaluation(args.output_dir, args.models, test_subs)

    elapsed = time.time() - t0
    logger.info(f"\n✓ Pipeline complete in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
