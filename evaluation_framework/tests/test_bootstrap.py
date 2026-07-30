"""
evaluation_framework/tests/test_bootstrap.py

Unit tests for the bootstrap statistical module.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evaluation_framework.bootstrap import (
    bootstrap_metric,
    bootstrap_all_metrics,
    bootstrap_comparison,
    mcnemar_test,
    BUILTIN_METRICS,
    BootstrapResult,
    BootstrapComparison,
    McNemarResult,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def perfect_predictions():
    """Perfect classifier: all correct."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=500)
    return y_true, y_true.copy()


@pytest.fixture
def good_predictions():
    """Good classifier: ~90% correct."""
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=500)
    y_pred = y_true.copy()
    # Flip 10% of predictions
    flip_idx = rng.choice(500, size=50, replace=False)
    y_pred[flip_idx] = 1 - y_pred[flip_idx]
    return y_true, y_pred


@pytest.fixture
def random_predictions():
    """Random classifier."""
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 2, size=500)
    y_pred = rng.integers(0, 2, size=500)
    return y_true, y_pred


# ─────────────────────────────────────────────────────────────────────────────
# bootstrap_metric
# ─────────────────────────────────────────────────────────────────────────────

class TestBootstrapMetric:
    def test_returns_bootstrap_result(self, good_predictions):
        y_true, y_pred = good_predictions
        result = bootstrap_metric(y_true, y_pred, "f1", n_bootstrap=100, seed=42)
        assert isinstance(result, BootstrapResult)

    def test_ci_contains_observed(self, good_predictions):
        """The CI should typically contain the observed value (not guaranteed but very likely)."""
        y_true, y_pred = good_predictions
        result = bootstrap_metric(y_true, y_pred, "recall", n_bootstrap=500, ci_level=0.95, seed=42)
        assert result.ci_lower <= result.observed <= result.ci_upper

    def test_ci_ordering(self, good_predictions):
        y_true, y_pred = good_predictions
        result = bootstrap_metric(y_true, y_pred, "f1", n_bootstrap=200, seed=42)
        assert result.ci_lower < result.ci_upper

    def test_perfect_classifier_high_recall(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        result = bootstrap_metric(y_true, y_pred, "recall", n_bootstrap=100, seed=42)
        assert result.observed == pytest.approx(1.0)
        assert result.ci_lower > 0.95

    def test_string_metric(self, good_predictions):
        y_true, y_pred = good_predictions
        result = bootstrap_metric(y_true, y_pred, "precision", n_bootstrap=100, seed=42)
        assert result.metric_name == "precision"

    def test_invalid_metric_string(self, good_predictions):
        y_true, y_pred = good_predictions
        with pytest.raises(ValueError, match="Unknown"):
            bootstrap_metric(y_true, y_pred, "invalid_metric", n_bootstrap=100)

    def test_n_bootstrap_stored(self, good_predictions):
        y_true, y_pred = good_predictions
        result = bootstrap_metric(y_true, y_pred, "f1", n_bootstrap=200, seed=42)
        assert result.n_bootstrap == 200

    def test_reproducible(self, good_predictions):
        y_true, y_pred = good_predictions
        r1 = bootstrap_metric(y_true, y_pred, "f1", n_bootstrap=100, seed=42)
        r2 = bootstrap_metric(y_true, y_pred, "f1", n_bootstrap=100, seed=42)
        assert r1.ci_lower == r2.ci_lower
        assert r1.ci_upper == r2.ci_upper


# ─────────────────────────────────────────────────────────────────────────────
# bootstrap_all_metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestBootstrapAllMetrics:
    def test_returns_all_four_metrics(self, good_predictions):
        y_true, y_pred = good_predictions
        results = bootstrap_all_metrics(y_true, y_pred, n_bootstrap=100, seed=42)
        assert set(results.keys()) == {"recall", "precision", "f1", "accuracy"}

    def test_f1_less_equal_one(self, good_predictions):
        y_true, y_pred = good_predictions
        results = bootstrap_all_metrics(y_true, y_pred, n_bootstrap=100, seed=42)
        assert results["f1"].ci_upper <= 1.0 + 1e-6

    def test_ci_consistent_across_metrics(self, good_predictions):
        """F1 CI should be between precision and recall CIs in typical cases."""
        y_true, y_pred = good_predictions
        results = bootstrap_all_metrics(y_true, y_pred, n_bootstrap=200, seed=42)
        # F1 observed <= max(precision, recall)
        assert results["f1"].observed <= max(
            results["precision"].observed, results["recall"].observed
        ) + 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# bootstrap_comparison
# ─────────────────────────────────────────────────────────────────────────────

class TestBootstrapComparison:
    def test_good_vs_random_significant(self, good_predictions, random_predictions):
        y_true_g, y_pred_g = good_predictions
        y_true_r, y_pred_r = random_predictions
        # Use shared y_true
        y_true = y_true_g
        result = bootstrap_comparison(y_true, y_pred_r, y_pred_g, "f1", n_bootstrap=500, seed=42)
        # Good classifier should be better
        assert result.delta_observed > 0
        assert isinstance(result, BootstrapComparison)

    def test_identical_models_not_significant(self, good_predictions):
        y_true, y_pred = good_predictions
        result = bootstrap_comparison(y_true, y_pred, y_pred, "f1", n_bootstrap=200, seed=42)
        assert abs(result.delta_observed) < 1e-9
        assert result.p_value > 0.05  # should not be significant

    def test_effect_size_positive(self, good_predictions, random_predictions):
        y_true, y_pred_good = good_predictions
        _, y_pred_rand = random_predictions
        result = bootstrap_comparison(y_true, y_pred_rand, y_pred_good, "recall", n_bootstrap=200, seed=42)
        assert result.effect_size >= 0

    def test_returns_comparison_object(self, good_predictions):
        y_true, y_pred = good_predictions
        result = bootstrap_comparison(y_true, y_pred, y_pred, "f1", n_bootstrap=100, seed=42)
        assert isinstance(result, BootstrapComparison)


# ─────────────────────────────────────────────────────────────────────────────
# mcnemar_test
# ─────────────────────────────────────────────────────────────────────────────

class TestMcNemar:
    def test_identical_models_not_significant(self, good_predictions):
        y_true, y_pred = good_predictions
        result = mcnemar_test(y_true, y_pred, y_pred)
        assert result.p_value == pytest.approx(1.0)
        assert not result.is_significant

    def test_very_different_models(self):
        """Model A always correct, Model B always wrong — should be significant."""
        rng    = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=500)
        y_pred_a = y_true.copy()
        y_pred_b = 1 - y_true
        result = mcnemar_test(y_true, y_pred_a, y_pred_b)
        assert result.is_significant
        assert result.p_value < 0.001
        assert result.n_discordant > 0

    def test_returns_mcnemar_result(self, good_predictions):
        y_true, y_pred = good_predictions
        result = mcnemar_test(y_true, y_pred, y_pred)
        assert isinstance(result, McNemarResult)

    def test_b_plus_c_equals_n_discordant(self, good_predictions, random_predictions):
        y_true, y_pred_good = good_predictions
        _, y_pred_rand = random_predictions
        result = mcnemar_test(y_true, y_pred_good, y_pred_rand)
        assert result.b + result.c == result.n_discordant

    def test_no_discordant_pairs(self):
        y_true = np.ones(100, dtype=int)
        y_pred = np.ones(100, dtype=int)
        result = mcnemar_test(y_true, y_pred, y_pred)
        assert result.n_discordant == 0
        assert result.p_value == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Builtin metrics self-tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBuiltinMetrics:
    def test_perfect_recall(self):
        y_true = np.array([1, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0])
        assert BUILTIN_METRICS["recall"](y_true, y_pred) == pytest.approx(1.0)

    def test_zero_recall(self):
        y_true = np.array([1, 1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 0, 0])
        assert BUILTIN_METRICS["recall"](y_true, y_pred) == pytest.approx(0.0)

    def test_perfect_precision(self):
        y_true = np.array([1, 1, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 0])
        assert BUILTIN_METRICS["precision"](y_true, y_pred) == pytest.approx(1.0)

    def test_f1_harmonic_mean(self):
        y_true = np.array([1, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 0])
        p = 1.0  # TP=2, FP=0
        r = 2 / 3  # TP=2, FN=1
        expected_f1 = 2 * p * r / (p + r)
        assert BUILTIN_METRICS["f1"](y_true, y_pred) == pytest.approx(expected_f1)

    def test_accuracy(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1])
        assert BUILTIN_METRICS["accuracy"](y_true, y_pred) == pytest.approx(0.5)
