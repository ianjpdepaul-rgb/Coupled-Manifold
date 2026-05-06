"""Tests for graceful.variance — variance structure characterization (Pass 2)."""

import math
import statistics

from graceful.variance import (
    compute_heteroscedasticity,
    compute_burstiness,
    compute_autocorrelation,
    compute_extremes,
    characterize_variance,
    format_variance,
)


class TestHeteroscedasticity:
    """Variance stability across segments."""

    def test_constant_variance(self):
        import random
        random.seed(42)
        # Need enough samples per segment to stabilize variance estimates
        values = [random.gauss(0, 10) for _ in range(100)]
        result = compute_heteroscedasticity(values)
        assert result["heteroscedastic"] is False
        assert result["ratio"] < 4.0

    def test_increasing_variance(self):
        import random
        random.seed(42)
        # First 20: stdev=1, last 20: stdev=20
        values = [random.gauss(0, 1) for _ in range(20)] + [random.gauss(0, 20) for _ in range(20)]
        result = compute_heteroscedasticity(values)
        assert result["heteroscedastic"] is True
        assert result["ratio"] > 4.0
        assert result["trend"] == "increasing"

    def test_decreasing_variance(self):
        import random
        random.seed(42)
        values = [random.gauss(0, 20) for _ in range(20)] + [random.gauss(0, 1) for _ in range(20)]
        result = compute_heteroscedasticity(values)
        assert result["heteroscedastic"] is True
        assert result["trend"] == "decreasing"

    def test_insufficient_data(self):
        result = compute_heteroscedasticity([1, 2, 3])
        assert result["heteroscedastic"] is False
        assert result["segments"] == []

    def test_flat_sequence(self):
        values = [5.0] * 20
        result = compute_heteroscedasticity(values)
        assert result["heteroscedastic"] is False

    def test_segments_count(self):
        import random
        random.seed(42)
        values = [random.gauss(0, 5) for _ in range(40)]
        result = compute_heteroscedasticity(values, n_segments=4)
        assert len(result["segments"]) == 4


class TestBurstiness:
    """Clustered vs uniform variance patterns."""

    def test_uniform_noise(self):
        import random
        random.seed(42)
        # Enough samples for rolling variances to stabilize
        values = [random.gauss(50, 5) for _ in range(100)]
        result = compute_burstiness(values)
        assert result["bursty"] is False

    def test_bursty_sequence(self):
        # Calm with a sudden burst
        values = [50.0] * 15 + [10.0, 90.0, 5.0, 95.0, 15.0, 85.0] + [50.0] * 19
        result = compute_burstiness(values, window=5)
        assert result["bursty"] is True
        assert result["cv"] > 1.5

    def test_insufficient_data(self):
        result = compute_burstiness([1, 2, 3], window=5)
        assert result["bursty"] is False
        assert result["fano"] == 1.0

    def test_constant_sequence(self):
        values = [50.0] * 30
        result = compute_burstiness(values)
        assert result["bursty"] is False
        assert result["fano"] == 1.0

    def test_max_burst_index(self):
        # Burst in the middle
        values = [50.0] * 10 + [10.0, 90.0, 10.0, 90.0, 10.0] + [50.0] * 10
        result = compute_burstiness(values, window=5)
        # The max burst should be around index 10 (where the oscillation is)
        assert 7 <= result["max_burst_idx"] <= 13


class TestAutocorrelation:
    """Temporal correlation structure."""

    def test_random_walk_persistent(self):
        import random
        random.seed(42)
        # Random walk is highly autocorrelated
        values = [0.0]
        for _ in range(50):
            values.append(values[-1] + random.gauss(0, 1))
        result = compute_autocorrelation(values)
        assert result["persistent"] is True
        assert result["lag_1"] > 0.5

    def test_alternating_antipersistent(self):
        # Alternating high/low — negative autocorrelation
        values = [10.0, 90.0] * 20
        result = compute_autocorrelation(values)
        assert result["antipersistent"] is True
        assert result["lag_1"] < -0.3

    def test_white_noise(self):
        import random
        random.seed(42)
        values = [random.gauss(0, 10) for _ in range(100)]
        result = compute_autocorrelation(values)
        # White noise: lag_1 should be near 0
        assert abs(result["lag_1"]) < 0.3
        assert result["persistent"] is False
        assert result["antipersistent"] is False

    def test_insufficient_data(self):
        result = compute_autocorrelation([1, 2, 3])
        assert result["lag_1"] == 0.0
        assert result["lags"] == {}

    def test_constant_sequence(self):
        values = [5.0] * 20
        result = compute_autocorrelation(values)
        assert result["lag_1"] == 0.0

    def test_multiple_lags_returned(self):
        import random
        random.seed(42)
        values = [random.gauss(0, 5) for _ in range(30)]
        result = compute_autocorrelation(values, max_lag=5)
        assert len(result["lags"]) == 5
        assert 1 in result["lags"]
        assert 5 in result["lags"]


class TestExtremes:
    """Tail behavior characterization."""

    def test_normal_distribution(self):
        import random
        random.seed(42)
        values = [random.gauss(50, 10) for _ in range(200)]
        result = compute_extremes(values)
        # Normal has kurtosis ~0
        assert abs(result["kurtosis"]) < 1.0
        # About 5% of values beyond 2 stdev for normal
        assert result["extreme_fraction"] < 0.1

    def test_heavy_tailed(self):
        import random
        random.seed(42)
        # Cauchy-like: mostly normal but occasional extreme values
        values = [random.gauss(50, 5) for _ in range(90)]
        values += [random.gauss(50, 50) for _ in range(10)]  # outliers
        result = compute_extremes(values)
        # Should have higher kurtosis and more extremes
        assert result["kurtosis"] > 0

    def test_uniform_light_tailed(self):
        # Uniform distribution has negative excess kurtosis
        values = [float(i) for i in range(100)]
        result = compute_extremes(values)
        assert result["kurtosis"] < 0  # Uniform = platykurtic

    def test_insufficient_data(self):
        result = compute_extremes([1, 2])
        assert result["kurtosis"] == 0.0
        assert result["extreme_count"] == 0

    def test_constant_sequence(self):
        values = [42.0] * 10
        result = compute_extremes(values)
        assert result["kurtosis"] == 0.0
        assert result["max_deviation"] == 0.0

    def test_max_deviation(self):
        # One clear outlier
        values = [50.0] * 19 + [150.0]
        result = compute_extremes(values)
        assert result["max_deviation"] > 2.0
        assert result["extreme_count"] >= 1


class TestCharacterizeVariance:
    """Integration test for main entry point."""

    def test_empty_history(self):
        result = characterize_variance([])
        assert result["summary"] == "no data"

    def test_short_history(self):
        history = [{"turn": i, "trace": 50.0, "drift": 0.2, "mode": "M", "model": "test"}
                   for i in range(5)]
        result = characterize_variance(history)
        assert "accumulating" in result["summary"]

    def test_well_behaved(self):
        import random
        random.seed(42)
        # Need enough samples for variance estimates to stabilize
        history = [{"turn": i, "trace": 50.0 + random.gauss(0, 5), "drift": 0.2 + random.gauss(0, 0.02),
                    "mode": "M", "model": "test"} for i in range(100)]
        result = characterize_variance(history)
        assert "trace" in result
        assert "well-behaved" in result["summary"]

    def test_heteroscedastic_trace(self):
        import random
        random.seed(42)
        history = ([{"turn": i, "trace": 50.0 + random.gauss(0, 2), "drift": 0.2,
                     "mode": "M", "model": "test"} for i in range(20)] +
                   [{"turn": i + 20, "trace": 50.0 + random.gauss(0, 30), "drift": 0.2,
                     "mode": "M", "model": "test"} for i in range(20)])
        result = characterize_variance(history)
        assert result["trace"]["heteroscedasticity"]["heteroscedastic"] is True
        assert "het" in result["summary"]

    def test_none_traces_handled(self):
        history = [{"turn": i, "trace": None, "drift": 0.2, "mode": "M", "model": "test"}
                   for i in range(30)]
        result = characterize_variance(history)
        assert result["trace"] == {}

    def test_negative_drift_excluded(self):
        history = [{"turn": i, "trace": 50.0, "drift": -1, "mode": "M", "model": "test"}
                   for i in range(30)]
        result = characterize_variance(history)
        assert result["drift"] == {}

    def test_basic_stats_included(self):
        history = [{"turn": i, "trace": float(50 + i), "drift": 0.3,
                    "mode": "M", "model": "test"} for i in range(20)]
        result = characterize_variance(history)
        assert "basic" in result["trace"]
        assert result["trace"]["basic"]["n"] == 20
        assert result["trace"]["basic"]["mean"] > 0


class TestFormatVariance:
    """Display formatting."""

    def test_no_data(self):
        assert format_variance({}) == (None, None)
        assert format_variance({"summary": "no data"}) == (None, None)

    def test_accumulating(self):
        result = {"summary": "accumulating (5 traces, 3 drifts)", "trace": {}, "drift": {}}
        assert format_variance(result) == (None, None)

    def test_well_behaved_returns_none(self):
        result = {
            "summary": "trace: well-behaved",
            "trace": {
                "heteroscedasticity": {"heteroscedastic": False},
                "burstiness": {"bursty": False},
                "autocorrelation": {"persistent": False},
                "extremes": {"kurtosis": 0.5},
            },
            "drift": {},
        }
        assert format_variance(result) == (None, None)

    def test_anomalous_surfaces(self):
        result = {
            "summary": "trace: het(increasing), bursty",
            "trace": {
                "heteroscedasticity": {"heteroscedastic": True},
                "burstiness": {"bursty": True},
                "autocorrelation": {"persistent": False},
                "extremes": {"kurtosis": 1.0},
            },
            "drift": {},
        }
        summary, detail = format_variance(result)
        assert summary is not None
        assert "burst" in summary

    def test_persistent_surfaces(self):
        result = {
            "summary": "trace: persistent",
            "trace": {
                "heteroscedasticity": {"heteroscedastic": False},
                "burstiness": {"bursty": False},
                "autocorrelation": {"persistent": True},
                "extremes": {"kurtosis": 0.5},
            },
            "drift": {},
        }
        summary, detail = format_variance(result)
        assert summary is not None
        assert "corr" in summary
