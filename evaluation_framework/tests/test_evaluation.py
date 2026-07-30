"""
evaluation_framework/tests/test_evaluation.py

Unit tests for the evaluation metrics engine.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evaluation_framework.evaluation import (
    compute_metrics,
    evaluate_all_thresholds,
    aggregate_by_modality,
    build_comparison_dataframe,
    MetricsResult,
)
from evaluation_framework.thresholds import MeanStdThreshold, PercentileThreshold


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_score_df(
    model_name="LSTM",
    modality="sEMG",
    channel_name="ch1",
    n_clean=200,
    n_anom=200,
    clean_mean=0.01,
    anom_mean=0.05,
    seed=0,
):
    """Build a minimal synthetic score CSV DataFrame."""
    rng = np.random.default_rng(seed)
    clean_errors = rng.normal(clean_mean, 0.003, size=n_clean).clip(min=0)
    anom_errors  = rng.normal(anom_mean,  0.010, size=n_anom).clip(min=0)

    rows = []
    for i, e in enumerate(clean_errors):
        rows.append({
            "model_name": model_name,
            "modality":   modality,
            "channel_name": channel_name,
            "movement": "WAK",
            "subject_id": "Sub36",
            "window_id": i,
            "reconstruction_error": float(e),
            "is_synthetic_anomaly": 0,
            "anomaly_type": "none",
            "predicted_label": 0,
        })
    for i, e in enumerate(anom_errors):
        rows.append({
            "model_name": model_name,
            "modality":   modality,
            "channel_name": channel_name,
            "movement": "WAK",
            "subject_id": "Sub36",
            "window_id": n_clean + i,
            "reconstruction_error": float(e),
            "is_synthetic_anomaly": 1,
            "anomaly_type": "amplitude_scale",
            "predicted_label": 0,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# compute_metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_perfect_classifier(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        m = compute_metrics(y_true, y_pred)
        assert m.precision == pytest.approx(1.0)
        assert m.recall    == pytest.approx(1.0)
        assert m.f1        == pytest.approx(1.0)
        assert m.accuracy  == pytest.approx(1.0)
        assert m.fpr       == pytest.approx(0.0)
        assert m.fnr       == pytest.approx(0.0)

    def test_all_zeros_predictor(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.zeros(4, dtype=int)
        m = compute_metrics(y_true, y_pred)
        assert m.recall    == pytest.approx(0.0)
        assert m.precision == pytest.approx(0.0)
        assert m.f1        == pytest.approx(0.0)

    def test_confusion_matrix_values(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        m = compute_metrics(y_true, y_pred)
        assert m.TP == 1
        assert m.FP == 1
        assert m.TN == 1
        assert m.FN == 1

    def test_roc_auc_with_scores(self):
        rng    = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=200)
        # Scores correlated with y_true → good AUC
        scores = y_true.astype(float) + rng.normal(0, 0.3, size=200)
        y_pred = (scores > 0.5).astype(int)
        m = compute_metrics(y_true, y_pred, scores=scores)
        assert not np.isnan(m.roc_auc)
        assert m.roc_auc > 0.5

    def test_roc_auc_without_scores(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        m = compute_metrics(y_true, y_pred, scores=None)
        assert np.isnan(m.roc_auc)

    def test_metadata_fields(self):
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        m = compute_metrics(
            y_true, y_pred,
            model_name="LSTM",
            modality="sEMG",
            channel_name="ch1",
            threshold_method="mean_std",
            threshold_value=0.05,
        )
        assert m.model_name == "LSTM"
        assert m.modality   == "sEMG"
        assert m.channel_name == "ch1"
        assert m.threshold_method == "mean_std"
        assert m.threshold_value  == pytest.approx(0.05)

    def test_returns_metrics_result(self):
        m = compute_metrics(np.array([0, 1]), np.array([0, 1]))
        assert isinstance(m, MetricsResult)

    def test_to_dict_has_all_keys(self):
        m = compute_metrics(np.array([0, 1]), np.array([0, 1]))
        d = m.to_dict()
        for key in ["precision", "recall", "f1", "accuracy", "fpr", "fnr",
                    "TP", "FP", "TN", "FN", "n_windows", "roc_auc", "pr_auc"]:
            assert key in d


# ─────────────────────────────────────────────────────────────────────────────
# evaluate_all_thresholds
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluateAllThresholds:
    def test_basic_run(self):
        df = _make_score_df()
        # Simulate training errors (clean distribution)
        train_errors = df[df["is_synthetic_anomaly"] == 0]["reconstruction_error"].values
        result = evaluate_all_thresholds(
            score_df=df,
            train_errors_map={"LSTM": {"ch1": train_errors}},
            methods=["mean_std", "percentile95"],
        )
        assert not result.empty
        # One row per method
        assert len(result) == 2

    def test_all_three_methods(self):
        df = _make_score_df()
        train_errors = df[df["is_synthetic_anomaly"] == 0]["reconstruction_error"].values
        result = evaluate_all_thresholds(
            score_df=df,
            train_errors_map={"LSTM": {"ch1": train_errors}},
            methods=["mean_std", "percentile95", "percentile99"],
        )
        methods_found = set(result["threshold_method"].unique())
        assert methods_found == {"mean_std", "percentile95", "percentile99"}

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"model_name": ["LSTM"], "channel_name": ["ch1"]})
        with pytest.raises(ValueError, match="missing required columns"):
            evaluate_all_thresholds(df, {})

    def test_fallback_when_no_train_errors(self):
        """Should not raise; uses test clean windows as proxy."""
        df = _make_score_df()
        result = evaluate_all_thresholds(
            score_df=df,
            train_errors_map={},  # no training errors provided
            methods=["mean_std"],
        )
        assert not result.empty

    def test_recall_between_0_and_1(self):
        df = _make_score_df(anom_mean=0.1)
        train_errors = df[df["is_synthetic_anomaly"] == 0]["reconstruction_error"].values
        result = evaluate_all_thresholds(
            score_df=df,
            train_errors_map={"LSTM": {"ch1": train_errors}},
        )
        assert (result["recall"] >= 0).all()
        assert (result["recall"] <= 1).all()

    def test_multi_channel(self):
        df1 = _make_score_df(channel_name="ch1", seed=0)
        df2 = _make_score_df(channel_name="ch2", seed=1)
        df  = pd.concat([df1, df2], ignore_index=True)
        train_errors_ch1 = df1[df1["is_synthetic_anomaly"] == 0]["reconstruction_error"].values
        train_errors_ch2 = df2[df2["is_synthetic_anomaly"] == 0]["reconstruction_error"].values
        result = evaluate_all_thresholds(
            score_df=df,
            train_errors_map={"LSTM": {"ch1": train_errors_ch1, "ch2": train_errors_ch2}},
            methods=["mean_std"],
        )
        channels = set(result["channel_name"].unique())
        assert "ch1" in channels
        assert "ch2" in channels


# ─────────────────────────────────────────────────────────────────────────────
# aggregate_by_modality
# ─────────────────────────────────────────────────────────────────────────────

class TestAggregateByModality:
    def test_produces_all_channels_row(self):
        df = _make_score_df()
        train_errors = df[df["is_synthetic_anomaly"] == 0]["reconstruction_error"].values
        metrics_df = evaluate_all_thresholds(
            score_df=df,
            train_errors_map={"LSTM": {"ch1": train_errors}},
            methods=["mean_std"],
        )
        agg = aggregate_by_modality(metrics_df)
        assert "ALL_CHANNELS" in agg["channel_name"].values

    def test_aggregate_recall_in_range(self):
        df = _make_score_df()
        train_errors = df[df["is_synthetic_anomaly"] == 0]["reconstruction_error"].values
        metrics_df = evaluate_all_thresholds(
            score_df=df,
            train_errors_map={"LSTM": {"ch1": train_errors}},
        )
        agg = aggregate_by_modality(metrics_df)
        assert (agg["recall"] >= 0).all()
        assert (agg["recall"] <= 1).all()
