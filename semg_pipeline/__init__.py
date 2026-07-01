"""
semg_pipeline/__init__.py

Public API for the sEMG preprocessing and anomaly-detection pipeline.
"""

from .loader import (
    load_semg_trial,
    SEMG_CHANNELS,
    SEMG_COL_MAP,
    MOVEMENTS,
    TRAIN_SUBS,
    VAL_SUBS,
    TEST_SUBS,
)
from .filter import apply_semg_filter_chain
from .normalizer import fit_scaler, apply_scaler, save_scaler, load_scaler
from .windower import create_semg_windows
from .anomaly_scorer import compute_threshold, label_windows, build_output_rows
from .evaluator import evaluate_model, evaluate_aggregate, save_evaluation_report

__all__ = [
    # loader
    "load_semg_trial",
    "SEMG_CHANNELS",
    "SEMG_COL_MAP",
    "MOVEMENTS",
    "TRAIN_SUBS",
    "VAL_SUBS",
    "TEST_SUBS",
    # filter
    "apply_semg_filter_chain",
    # normalizer
    "fit_scaler",
    "apply_scaler",
    "save_scaler",
    "load_scaler",
    # windower
    "create_semg_windows",
    # anomaly scorer
    "compute_threshold",
    "label_windows",
    "build_output_rows",
    # evaluator
    "evaluate_model",
    "evaluate_aggregate",
    "save_evaluation_report",
]
