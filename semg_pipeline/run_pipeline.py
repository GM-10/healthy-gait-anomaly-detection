"""
semg_pipeline/run_pipeline.py

Master orchestration script for the sEMG anomaly detection pipeline.

Runs the complete pipeline end-to-end:
    1. Load + filter + normalize all training trials
    2. Fit Min-Max scaler on train subjects only
    3. Segment into windows (1920 samples, 50% overlap, cycle-safe)
    4. Train SARIMA / LSTM / Transformer models (one per channel)
    5. Compute per-channel thresholds from training reconstruction errors
    6. Process validation set (early stopping for LSTM & Transformer)
    7. Process test set: score clean + synthetically injected windows
    8. Save output CSVs per subject per movement per model
    9. Run evaluation: Recall, F1, RMSE, confusion matrix
   10. Save aggregate evaluation summary

Subject splits (must match kinematics/kinetics teammate exactly):
    Train:      Sub01 – Sub30
    Validation: Sub31 – Sub35
    Test:       Sub36 – Sub40

Output structure:
    outputs/sEMG/
    ├── scaler_params.json
    ├── models/
    │   ├── lstm/          ← LSTM model weights
    │   ├── transformer/   ← Transformer model weights
    │   └── sarima/        ← SARIMA pickle files
    ├── Sub36/
    │   ├── Sub36_WAK_LSTM_scores.csv
    │   ├── Sub36_WAK_Transformer_scores.csv
    │   └── Sub36_WAK_SARIMA_scores.csv
    ├── ...
    └── evaluation_summary.csv

Usage
-----
    # Full pipeline (all models, all movements):
    python -m semg_pipeline.run_pipeline \
        --base_dir SIAT_LLMD20230404/SIAT_LLMD20230404

    # Quick smoke-test (1 subject, 1 movement, LSTM only):
    python -m semg_pipeline.run_pipeline \
        --base_dir SIAT_LLMD20230404/SIAT_LLMD20230404 \
        --movements WAK --models lstm --dry_run

    # Skip training (load saved weights) and only score test set:
    python -m semg_pipeline.run_pipeline \
        --base_dir SIAT_LLMD20230404/SIAT_LLMD20230404 \
        --skip_training --models lstm transformer
"""

import argparse
import logging
import os
import sys
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from semg_pipeline.models.sarima_model import SARIMAModel

import numpy as np
import pandas as pd

# ── Ensure repo root is on sys.path so we can import utils.synthetic_anomalies
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ── sEMG pipeline imports
from semg_pipeline.loader import (
    load_semg_trial,
    build_trial_paths,
    SEMG_CHANNELS,
    MOVEMENTS,
    TRAIN_SUBS,
    VAL_SUBS,
    TEST_SUBS,
)
from semg_pipeline.filter     import apply_semg_filter_chain
from semg_pipeline.normalizer import fit_scaler, apply_scaler, save_scaler, load_scaler
from semg_pipeline.windower   import create_semg_windows
from semg_pipeline.anomaly_scorer import (
    compute_threshold,
    label_windows,
    build_output_rows,
    save_train_errors,
    load_train_errors,
)
from semg_pipeline.evaluator import (
    evaluate_model,
    evaluate_aggregate,
    save_evaluation_report,
    print_evaluation_summary,
)

# ── Shared synthetic anomaly injection (must import from repo root)
from utils.synthetic_anomalies import (
    inject_amplitude_scale,
    inject_time_warp,
    inject_time_shift,
    inject_combined,
    DEFAULT_SEVERITIES,
)

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("semg_pipeline")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

WINDOW_SIZE  = 1920   # 1 second at 1920 Hz
OVERLAP_SIZE = 960    # 50% overlap
FS           = 1920.0

MODEL_REGISTRY = {
    "sarima":       "SARIMA",
    "lstm":         "LSTM",
    "transformer":  "Transformer",
}

# Synthetic anomaly conditions — built at runtime from requested severity levels.
# Each entry: (inject_fn, {kwargs}, anomaly_type_str, severity_label_str)
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


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load + filter + window a single trial
# ─────────────────────────────────────────────────────────────────────────────

def _process_trial(
    base_dir: str,
    subject: str,
    movement: str,
    scaling_params: Optional[Dict] = None,
    filter_only: bool = False,
) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray], Optional[List]]:
    """
    Load → filter → [scale] → window a single trial.

    Returns
    -------
    (filtered_df, windows_array, windows_meta)
    Any of which can be None if the trial files are missing or yield 0 windows.
    """
    data_path, label_path = build_trial_paths(base_dir, subject, movement)

    if not os.path.exists(data_path) or not os.path.exists(label_path):
        return None, None, None

    try:
        df = load_semg_trial(data_path, label_path, active_only=True)
    except Exception as e:
        logger.warning(f"  [load] {subject}/{movement}: {e}")
        return None, None, None

    if df.empty:
        return None, None, None

    # Filter chain (applied at native 1920 Hz)
    df = apply_semg_filter_chain(df, fs=FS)

    if filter_only:
        return df, None, None

    # Scale (apply pre-fitted params; if None, skip scaling)
    if scaling_params is not None:
        df = apply_scaler(df, scaling_params)

    # Window
    windows, meta = create_semg_windows(df, WINDOW_SIZE, OVERLAP_SIZE)
    if len(windows) == 0:
        return df, None, None

    return df, windows, meta


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Collect training windows
# ─────────────────────────────────────────────────────────────────────────────

def collect_split_windows(
    base_dir: str,
    subjects: List[str],
    movements: List[str],
    scaling_params: Optional[Dict],
    split_label: str = "train",
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Load and window all trials for a subject split.

    Returns
    -------
    (all_windows, all_meta) — concatenated across all subjects × movements
    """
    all_windows = []
    all_meta    = []

    n = len(subjects) * len(movements)
    done = 0
    for sub in subjects:
        for mov in movements:
            done += 1
            logger.info(f"  [{split_label}] {sub}/{mov}  ({done}/{n})")
            _, windows, meta = _process_trial(
                base_dir, sub, mov, scaling_params
            )
            if windows is not None and len(windows) > 0:
                all_windows.append(windows)
                all_meta.extend(meta)

    if not all_windows:
        logger.warning(f"No windows collected for {split_label} split.")
        return np.empty((0, WINDOW_SIZE, len(SEMG_CHANNELS)), dtype=np.float32), []

    return np.concatenate(all_windows, axis=0), all_meta


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Build & train models
# ─────────────────────────────────────────────────────────────────────────────

def _train_sarima(
    train_windows: np.ndarray,
    model_dir: str,
    sarima_train_subs: int,
) -> "SARIMAModel":
    from semg_pipeline.models.sarima_model import SARIMAModel, SARIMA_MAX_TRAIN_SUBJECTS

    max_subs = sarima_train_subs or SARIMA_MAX_TRAIN_SUBJECTS
    logger.info(
        f"[SARIMA] Training on up to {max_subs} subjects' windows "
        f"({len(train_windows)} total train windows available)."
    )
    model = SARIMAModel(
        channel_names=SEMG_CHANNELS,
        max_windows_per_channel=200,
        model_dir=os.path.join(model_dir, "sarima"),
    )
    model.fit(train_windows)
    return model


def _train_lstm(
    train_windows: np.ndarray,
    val_windows: np.ndarray,
    model_dir: str,
    max_subjects: Optional[int] = None,
    use_classifier: bool = False,
    lambda_cls: float = 0.5,
    augmentation_fraction: float = 0.20,
    augmentation_severity: str = "moderate",
    augmentation_types: Optional[List] = None,
) -> List:
    from semg_pipeline.models.lstm_model import LSTMModel

    # Optionally subsample training windows to simulate using only
    # max_subjects out of 30 train subjects (proportional window count).
    if max_subjects is not None and max_subjects < len(TRAIN_SUBS):
        ratio      = max_subjects / len(TRAIN_SUBS)
        n_keep     = max(1, int(round(len(train_windows) * ratio)))
        rng        = np.random.default_rng(42)
        indices    = rng.choice(len(train_windows), size=n_keep, replace=False)
        indices.sort()
        n_total = len(train_windows)
        train_windows = train_windows[indices]
        logger.info(
            f"[LSTM] --max_subjects={max_subjects}: "
            f"using {n_keep}/{n_total} windows "
            f"(~{max_subjects}/{len(TRAIN_SUBS)} subjects)."
        )

    models = []
    for ch_idx, ch_name in enumerate(SEMG_CHANNELS):
        logger.info(f"[LSTM] Training channel {ch_idx + 1}/9: {ch_name}")
        m = LSTMModel(
            channel_name=ch_name,
            window_size=WINDOW_SIZE,
            use_classifier=use_classifier,
            lambda_cls=lambda_cls,
            augmentation_fraction=augmentation_fraction,
            augmentation_severity=augmentation_severity,
            augmentation_types=augmentation_types,
        )
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
    use_classifier: bool = False,
    lambda_cls: float = 0.5,
    augmentation_fraction: float = 0.20,
    augmentation_severity: str = "moderate",
    augmentation_types: Optional[List] = None,
) -> List:
    from semg_pipeline.models.transformer_model import TransformerModel

    models = []
    for ch_idx, ch_name in enumerate(SEMG_CHANNELS):
        logger.info(f"[Transformer] Training channel {ch_idx + 1}/9: {ch_name}")
        m = TransformerModel(
            channel_name=ch_name,
            window_size=WINDOW_SIZE,
            use_classifier=use_classifier,
            lambda_cls=lambda_cls,
            augmentation_fraction=augmentation_fraction,
            augmentation_severity=augmentation_severity,
            augmentation_types=augmentation_types,
        )
        train_ch = train_windows[:, :, ch_idx : ch_idx + 1]
        val_ch   = val_windows[:, :, ch_idx : ch_idx + 1] if len(val_windows) > 0 else None
        m.fit(train_ch, val_ch)
        save_path = os.path.join(model_dir, "transformer", f"transformer_ch{ch_idx:02d}.pt")
        m.save(save_path)
        models.append(m)
    return models


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Load pre-trained models
# ─────────────────────────────────────────────────────────────────────────────

def _load_lstm_models(model_dir: str) -> List:
    from semg_pipeline.models.lstm_model import LSTMModel
    models = []
    for ch_idx, ch_name in enumerate(SEMG_CHANNELS):
        m = LSTMModel(channel_name=ch_name, window_size=WINDOW_SIZE)
        path = os.path.join(model_dir, "lstm", f"lstm_ch{ch_idx:02d}.pt")
        if os.path.exists(path):
            m.load(path)
        else:
            logger.warning(f"[LSTM] Weight file not found: {path}")
        models.append(m)
    return models


def _load_transformer_models(model_dir: str) -> List:
    from semg_pipeline.models.transformer_model import TransformerModel
    models = []
    for ch_idx, ch_name in enumerate(SEMG_CHANNELS):
        m = TransformerModel(channel_name=ch_name, window_size=WINDOW_SIZE)
        path = os.path.join(model_dir, "transformer", f"transformer_ch{ch_idx:02d}.pt")
        if os.path.exists(path):
            m.load(path)
        else:
            logger.warning(f"[Transformer] Weight file not found: {path}")
        models.append(m)
    return models


def _load_sarima_models(model_dir: str) -> "SARIMAModel":
    from semg_pipeline.models.sarima_model import SARIMAModel
    m = SARIMAModel(
        channel_names=SEMG_CHANNELS,
        model_dir=os.path.join(model_dir, "sarima"),
    )
    m.load(os.path.join(model_dir, "sarima"))
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Compute train thresholds
# ─────────────────────────────────────────────────────────────────────────────

def compute_train_thresholds(
    active_models: Dict[str, object],
    train_windows: np.ndarray,
    output_dir: Optional[str] = None,
    save_errors: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-model per-channel anomaly thresholds from training windows.

    Optionally saves training errors as .npy files for use by evaluate.py
    when computing percentile thresholds.

    Returns
    -------
    thresholds[model_key][channel_name] = float threshold
    """
    thresholds: Dict[str, Dict[str, float]] = {}

    for model_key, model_or_list in active_models.items():
        model_display = MODEL_REGISTRY[model_key]
        logger.info(f"[Threshold] Computing thresholds for {model_display} …")
        ch_thresholds = {}

        for ch_idx, ch_name in enumerate(SEMG_CHANNELS):
            if model_key == "sarima":
                from semg_pipeline.models.sarima_model import SARIMAModel
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


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5: Score test set with anomaly injection
# ─────────────────────────────────────────────────────────────────────────────

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
    include_classifier: bool = False,
    use_classifier_labeling: bool = False,
) -> None:
    """
    Score one test trial for all active models.
    Generates clean + anomalous windows, scores each, saves output CSVs.

    Parameters
    ----------
    anomaly_conditions : list of (inject_fn, kwargs, atype, severity_label)
        Built by _build_anomaly_conditions() for the requested severity levels.
    severity_tag : str
        Inserted into the output filename before '_scores.csv' when non-empty.
        e.g. 'mild' -> Sub36_WAK_LSTM_mild_scores.csv
        Empty string -> Sub36_WAK_LSTM_scores.csv  (default / backward-compat)
    include_classifier : bool, default False
        When True, calls anomaly_probability() on each LSTM/Transformer model
        and includes the result as the `anomaly_probability` column.
    use_classifier_labeling : bool, default False
        When True (and include_classifier=True), derives predicted_label from
        classifier probability instead of reconstruction threshold.
    """
    _, windows, meta = _process_trial(base_dir, subject, movement, scaling_params)
    if windows is None or len(windows) == 0:
        logger.warning(f"  [test] {subject}/{movement}: no windows — skipping.")
        return

    n_windows = len(windows)
    sub_out_dir = os.path.join(output_dir, subject)
    os.makedirs(sub_out_dir, exist_ok=True)

    for model_key, model_or_list in active_models.items():
        model_display = MODEL_REGISTRY[model_key]
        all_rows: List[Dict] = []

        # ── A) Clean windows ────────────────────────────────────────────
        for ch_idx, ch_name in enumerate(SEMG_CHANNELS):
            th = thresholds[model_key][ch_name]

            if model_key == "sarima":
                errors = model_or_list.score(windows)[:, ch_idx]
                preds  = label_windows(errors, th)
                rows   = build_output_rows(
                    meta, errors, preds, ch_name, subject, movement,
                    model_display,
                    is_synthetic_anomaly=0, anomaly_type="none",
                    window_id_offset=0, severity=0.0,
                )
            else:
                ch_win  = windows[:, :, ch_idx : ch_idx + 1]
                errors  = model_or_list[ch_idx].score(ch_win)

                # Optionally collect classifier probabilities
                cls_probs = None
                if include_classifier:
                    try:
                        cls_probs = model_or_list[ch_idx].anomaly_probability(ch_win)
                    except RuntimeError as exc:
                        logger.warning(
                            f"  [{model_display}] anomaly_probability() failed "
                            f"for {ch_name}: {exc}"
                        )

                if use_classifier_labeling and cls_probs is not None:
                    preds = (cls_probs > 0.5).astype(int)
                else:
                    preds = label_windows(errors, th)

                rows = build_output_rows(
                    meta, errors, preds, ch_name, subject, movement,
                    model_display,
                    is_synthetic_anomaly=0, anomaly_type="none",
                    window_id_offset=0, severity=0.0,
                    classifier_probs=cls_probs,
                )
            all_rows.extend(rows)

        # ── B) Anomalous windows ─────────────────────────────────────────────
        # Apply each anomaly condition to each clean window independently
        for cond_idx, (inject_fn, inject_kwargs, atype, severity_label) in enumerate(anomaly_conditions):
            # Build anomalous version of every window for every channel
            # Shape: (n_windows, WINDOW_SIZE, 9)
            anom_windows = np.empty_like(windows)
            for w_idx in range(n_windows):
                for ch_idx in range(len(SEMG_CHANNELS)):
                    sig = windows[w_idx, :, ch_idx]
                    anom_sig, _ = inject_fn(sig, **inject_kwargs)
                    anom_windows[w_idx, :, ch_idx] = anom_sig

            window_id_offset = n_windows * (cond_idx + 1)

            for ch_idx, ch_name in enumerate(SEMG_CHANNELS):
                th = thresholds[model_key][ch_name]

                if model_key == "sarima":
                    errors = model_or_list.score(anom_windows)[:, ch_idx]
                    preds  = label_windows(errors, th)
                    rows   = build_output_rows(
                        meta, errors, preds, ch_name, subject, movement,
                        model_display,
                        is_synthetic_anomaly=1,
                        anomaly_type=atype,
                        window_id_offset=window_id_offset,
                        severity=float(sev_numeric),
                    )
                else:
                    ch_win  = anom_windows[:, :, ch_idx : ch_idx + 1]
                    errors  = model_or_list[ch_idx].score(ch_win)

                    cls_probs = None
                    if include_classifier:
                        try:
                            cls_probs = model_or_list[ch_idx].anomaly_probability(ch_win)
                        except RuntimeError as exc:
                            logger.warning(
                                f"  [{model_display}] anomaly_probability() failed "
                                f"for {ch_name}: {exc}"
                            )

                    if use_classifier_labeling and cls_probs is not None:
                        preds = (cls_probs > 0.5).astype(int)
                    else:
                        preds = label_windows(errors, th)

                    rows = build_output_rows(
                        meta, errors, preds, ch_name, subject, movement,
                        model_display,
                        is_synthetic_anomaly=1,
                        anomaly_type=atype,
                        window_id_offset=window_id_offset,
                        severity=float(sev_numeric),
                        classifier_probs=cls_probs,
                    )
                all_rows.extend(rows)

        # ── Save output CSV ───────────────────────────────────────────────────
        sev_part = f"_{severity_tag}" if severity_tag else ""
        out_path = os.path.join(
            sub_out_dir, f"{subject}_{movement}_{model_display}{sev_part}_scores.csv"
        )
        out_df = pd.DataFrame(all_rows)

        # Ensure exact column order (include new columns with fallback)
        col_order = [
            "subject_id", "modality", "channel_name", "movement", "window_id",
            "window_start_time", "window_end_time", "reconstruction_error",
            "is_synthetic_anomaly", "anomaly_type", "predicted_label", "model_name",
            "severity", "anomaly_probability",
        ]
        # Only keep columns that exist (backward-compatible)
        col_order = [c for c in col_order if c in out_df.columns]
        out_df = out_df[col_order]
        out_df.to_csv(out_path, index=False)
        logger.info(
            f"  [{model_display}] {subject}/{movement} → {out_path}  "
            f"({len(out_df)} rows)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6: Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(output_dir: str, active_model_keys: List[str]) -> None:
    """Load all test output CSVs and compute evaluation metrics."""
    logger.info("\n[Evaluation] Loading test score CSVs …")

    all_results: Dict[str, pd.DataFrame] = {}

    for model_key in active_model_keys:
        model_display = MODEL_REGISTRY[model_key]
        dfs = []

        for sub in TEST_SUBS:
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

    # Save combined report
    summary_path = save_evaluation_report(all_results, output_dir)
    logger.info(f"\n[Evaluation] Summary saved → {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="sEMG anomaly detection pipeline (SIAT-LLMD)"
    )
    parser.add_argument(
        "--base_dir",
        default=os.path.join("SIAT_LLMD20230404", "SIAT_LLMD20230404"),
        help="Root path of the SIAT-LLMD dataset (default: SIAT_LLMD20230404/SIAT_LLMD20230404)",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join("outputs", "sEMG"),
        help="Root output directory (default: outputs/sEMG)",
    )
    parser.add_argument(
        "--movements",
        nargs="+",
        default=MOVEMENTS,
        help="Movement codes to process (default: all 16)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lstm", "transformer", "sarima"],
        choices=list(MODEL_REGISTRY.keys()),
        help="Models to run (default: lstm transformer sarima)",
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
        help="Number of train subjects to use for SARIMA fitting (default: 5).",
    )
    parser.add_argument(
        "--max_subjects",
        type=int,
        default=None,
        help=(
            "Limit LSTM (and Transformer) training to a proportional subset of "
            "windows equivalent to N subjects out of 30. "
            "Useful for quick experiments (e.g. --max_subjects 5). "
            "Default: None (use all 30 train subjects)."
        ),
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
            "'percentile95'/'percentile99' = percentile-based thresholds. "
            "Note: evaluate.py re-computes all three methods from saved train errors."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Process only Sub01 (train), Sub31 (val), Sub36 (test) for smoke-testing.",
    )
    # ── Semi-supervised options ─────────────────────────────────────────────
    parser.add_argument(
        "--semi_supervised",
        action="store_true",
        help=(
            "Enable semi-supervised training: attach a classifier head to "
            "LSTM/Transformer encoders and train jointly with a combined "
            "reconstruction + classification loss. "
            "See also: --augmentation_fraction, --lambda_cls, "
            "--augmentation_severity, --use_classifier_labeling."
        ),
    )
    parser.add_argument(
        "--augmentation_fraction",
        type=float,
        default=0.20,
        help=(
            "Fraction of training windows replaced by synthetic anomalies during "
            "semi-supervised training (default: 0.20). "
            "Only used when --semi_supervised is set. "
            "Sweep this value (e.g. 0.10, 0.20, 0.30) to find the optimal trade-off."
        ),
    )
    parser.add_argument(
        "--lambda_cls",
        type=float,
        default=0.5,
        help=(
            "Weight of the classification loss in combined loss: "
            "total_loss = recon_loss + lambda_cls * bce_loss (default: 0.5). "
            "Only used when --semi_supervised is set. "
            "Sweep this value (e.g. 0.1, 0.5, 1.0) for ablation study."
        ),
    )
    parser.add_argument(
        "--augmentation_severity",
        default="moderate",
        choices=["mild", "moderate", "severe"],
        help=(
            "Severity level for synthetic anomalies injected during training "
            "(default: moderate=0.35). Only used when --semi_supervised is set."
        ),
    )
    parser.add_argument(
        "--use_classifier_labeling",
        action="store_true",
        help=(
            "At test time, derive predicted_label from classifier probability "
            "(prob > 0.5) instead of reconstruction error threshold. "
            "Only active when --semi_supervised is set. "
            "When not set, reconstruction-error thresholding is used for labels "
            "and the classifier probability is recorded as an extra column only."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0   = time.time()

    logger.info("=" * 65)
    logger.info("  sEMG Anomaly Detection Pipeline — SIAT-LLMD")
    logger.info("=" * 65)
    logger.info(f"  base_dir        : {args.base_dir}")
    logger.info(f"  output_dir      : {args.output_dir}")
    logger.info(f"  movements       : {args.movements}")
    logger.info(f"  models          : {args.models}")
    logger.info(f"  severities      : {args.severities}")
    logger.info(f"  max_subjects    : {args.max_subjects or 'all (30)'}")
    logger.info(f"  sarima_max_subs : {args.sarima_max_subjects}")
    logger.info(f"  dry_run         : {args.dry_run}")
    logger.info(f"  semi_supervised : {args.semi_supervised}")
    if args.semi_supervised:
        logger.info(f"    augmentation_fraction   : {args.augmentation_fraction}")
        logger.info(f"    lambda_cls              : {args.lambda_cls}")
        logger.info(f"    augmentation_severity   : {args.augmentation_severity}")
        logger.info(f"    use_classifier_labeling : {args.use_classifier_labeling}")

    model_dir = os.path.join(args.output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # ── Dry-run: limit subjects ────────────────────────────────────────────
    if args.dry_run:
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
    # Severity tag used in filenames: empty when only moderate (backward-compat)
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

        # ── 1. Fit scaler on raw filtered training data ────────────────────
        if os.path.exists(scaler_path) and args.skip_training:
            logger.info(f"[Scaler] Loading existing scaler from {scaler_path}")
            scaling_params = load_scaler(scaler_path)
        else:
            logger.info("[Scaler] Collecting raw training data to fit scaler …")
            raw_train_dfs = []
            for sub in train_subs:
                for mov in args.movements:
                    df, _, _ = _process_trial(
                        args.base_dir, sub, mov,
                        scaling_params=None,
                        filter_only=True,
                    )
                    if df is not None and not df.empty:
                        raw_train_dfs.append(df)

            if not raw_train_dfs:
                logger.error("No training data found. Check --base_dir.")
                sys.exit(1)

            scaling_params = fit_scaler(raw_train_dfs)
            save_scaler(scaling_params, scaler_path)
            logger.info(f"[Scaler] Fitted and saved → {scaler_path}")

        # ── 2. Collect train + val windows ────────────────────────────────
        logger.info("\n[Windows] Collecting training windows …")
        train_windows, train_meta = collect_split_windows(
            args.base_dir, train_subs, args.movements, scaling_params, "train"
        )
        logger.info(f"  Train windows: {len(train_windows)}")

        logger.info("[Windows] Collecting validation windows …")
        val_windows, val_meta = collect_split_windows(
            args.base_dir, val_subs, args.movements, scaling_params, "val"
        )
        logger.info(f"  Val windows: {len(val_windows)}")

        # ── 3. Train or load models ────────────────────────────────────────
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
                        train_windows, val_windows, model_dir,
                        max_subjects=args.max_subjects,
                        use_classifier=args.semi_supervised,
                        lambda_cls=args.lambda_cls,
                        augmentation_fraction=args.augmentation_fraction,
                        augmentation_severity=args.augmentation_severity,
                    )
                elif model_key == "transformer":
                    active_models["transformer"] = _train_transformer(
                        train_windows, val_windows, model_dir,
                        use_classifier=args.semi_supervised,
                        lambda_cls=args.lambda_cls,
                        augmentation_fraction=args.augmentation_fraction,
                        augmentation_severity=args.augmentation_severity,
                    )
                elif model_key == "sarima":
                    active_models["sarima"] = _train_sarima(
                        train_windows, model_dir, args.sarima_max_subjects
                    )

        # ── 4. Compute thresholds on train set ────────────────────────────
        logger.info("\n[Thresholds] Computing anomaly thresholds from train errors …")
        thresholds = compute_train_thresholds(
            active_models, train_windows,
            output_dir=args.output_dir,
            save_errors=getattr(args, "save_train_errors", False),
        )

        # ── 5. Score test set ─────────────────────────────────────────────
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
                    include_classifier=args.semi_supervised,
                    use_classifier_labeling=getattr(args, "use_classifier_labeling", False),
                )

    # ══════════════════════════════════════════════════════════════════════
    # EVALUATION PHASE
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n[Evaluation] Running metrics …")
    run_evaluation(args.output_dir, args.models)

    elapsed = time.time() - t0
    logger.info(f"\n✓ Pipeline complete in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
