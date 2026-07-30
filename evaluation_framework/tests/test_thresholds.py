"""
evaluation_framework/tests/test_thresholds.py

Unit tests for the thresholding module.
"""

import numpy as np
import pytest
import sys
import os

# Ensure repo root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evaluation_framework.thresholds import (
    MeanStdThreshold,
    PercentileThreshold,
    get_threshold,
    fit_all_thresholds,
    ThresholdResult,
    THRESHOLD_METHODS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def healthy_errors():
    """Simulate 1000 healthy training reconstruction errors (log-normal)."""
    rng = np.random.default_rng(42)
    return rng.lognormal(mean=-3.0, sigma=0.5, size=1000)


# ─────────────────────────────────────────────────────────────────────────────
# MeanStdThreshold
# ─────────────────────────────────────────────────────────────────────────────

class TestMeanStdThreshold:
    def test_name(self):
        t = MeanStdThreshold()
        assert t.name == "mean_std"

    def test_params_default(self):
        t = MeanStdThreshold()
        assert t.params["n_sigma"] == 3.0

    def test_params_custom(self):
        t = MeanStdThreshold(n_sigma=2.0)
        assert t.params["n_sigma"] == 2.0

    def test_fit_value(self, healthy_errors):
        t = MeanStdThreshold(n_sigma=3.0)
        result = t.fit(healthy_errors)
        expected = np.mean(healthy_errors) + 3.0 * np.std(healthy_errors)
        assert abs(result.value - expected) < 1e-10

    def test_fit_stats_populated(self, healthy_errors):
        result = MeanStdThreshold().fit(healthy_errors)
        assert result.train_n == len(healthy_errors)
        assert result.train_p95 == pytest.approx(np.percentile(healthy_errors, 95), rel=1e-6)
        assert result.train_mean == pytest.approx(np.mean(healthy_errors), rel=1e-6)

    def test_fit_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            MeanStdThreshold().fit(np.array([]))

    def test_predict(self, healthy_errors):
        t = MeanStdThreshold()
        result = t.fit(healthy_errors)
        scores = np.array([result.value - 0.1, result.value + 0.1])
        preds  = t.predict(scores, result.value)
        assert preds[0] == 0
        assert preds[1] == 1

    def test_method_stored_in_result(self, healthy_errors):
        result = MeanStdThreshold().fit(healthy_errors)
        assert result.method == "mean_std"


# ─────────────────────────────────────────────────────────────────────────────
# PercentileThreshold
# ─────────────────────────────────────────────────────────────────────────────

class TestPercentileThreshold:
    def test_name_95(self):
        t = PercentileThreshold(95)
        assert t.name == "percentile95"

    def test_name_99(self):
        t = PercentileThreshold(99)
        assert t.name == "percentile99"

    def test_invalid_percentile(self):
        with pytest.raises(ValueError):
            PercentileThreshold(0)
        with pytest.raises(ValueError):
            PercentileThreshold(101)

    def test_fit_value_95(self, healthy_errors):
        t = PercentileThreshold(95)
        result = t.fit(healthy_errors)
        expected = np.percentile(healthy_errors, 95)
        assert abs(result.value - expected) < 1e-10

    def test_fit_value_99(self, healthy_errors):
        t = PercentileThreshold(99)
        result = t.fit(healthy_errors)
        expected = np.percentile(healthy_errors, 99)
        assert abs(result.value - expected) < 1e-10

    def test_p95_less_than_p99(self, healthy_errors):
        r95 = PercentileThreshold(95).fit(healthy_errors)
        r99 = PercentileThreshold(99).fit(healthy_errors)
        assert r95.value < r99.value

    def test_method_stored(self, healthy_errors):
        result = PercentileThreshold(95).fit(healthy_errors)
        assert result.method == "percentile95"
        assert result.params["percentile"] == 95.0


# ─────────────────────────────────────────────────────────────────────────────
# Factory / Registry
# ─────────────────────────────────────────────────────────────────────────────

class TestRegistry:
    def test_get_threshold_mean_std(self):
        t = get_threshold("mean_std")
        assert isinstance(t, MeanStdThreshold)

    def test_get_threshold_percentile95(self):
        t = get_threshold("percentile95")
        assert isinstance(t, PercentileThreshold)

    def test_get_threshold_percentile99(self):
        t = get_threshold("percentile99")
        assert isinstance(t, PercentileThreshold)

    def test_get_threshold_invalid(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_threshold("invalid_method")

    def test_all_methods_in_registry(self):
        for name in ["mean_std", "percentile95", "percentile99"]:
            assert name in THRESHOLD_METHODS

    def test_fit_all_thresholds(self, healthy_errors):
        results = fit_all_thresholds(healthy_errors)
        assert set(results.keys()) == {"mean_std", "percentile95", "percentile99"}
        for k, v in results.items():
            assert isinstance(v, ThresholdResult)
            assert v.value > 0

    def test_fit_all_subset(self, healthy_errors):
        results = fit_all_thresholds(healthy_errors, methods=["mean_std", "percentile95"])
        assert "percentile99" not in results
        assert len(results) == 2

    def test_threshold_ordering(self, healthy_errors):
        """P95 threshold < P99 threshold, and both should be >= mean_std for typical errors."""
        results = fit_all_thresholds(healthy_errors)
        assert results["percentile95"].value < results["percentile99"].value


# ─────────────────────────────────────────────────────────────────────────────
# Serialization
# ─────────────────────────────────────────────────────────────────────────────

class TestSerialization:
    def test_to_dict(self, healthy_errors):
        result = MeanStdThreshold().fit(healthy_errors)
        d = result.to_dict()
        assert d["method"] == "mean_std"
        assert isinstance(d["value"], float)
        assert isinstance(d["train_n"], int)

    def test_to_json_from_json(self, healthy_errors, tmp_path):
        result = MeanStdThreshold().fit(healthy_errors)
        path   = str(tmp_path / "threshold.json")
        result.to_json(path)
        loaded = ThresholdResult.from_json(path)
        assert abs(loaded.value - result.value) < 1e-10
        assert loaded.method == result.method
        assert loaded.train_n == result.train_n
