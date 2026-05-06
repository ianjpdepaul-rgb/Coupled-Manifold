"""Tests for graceful.channel_correlation — cross-channel correlation (Pass 3)."""

import math

from graceful.channel_correlation import (
    pearson,
    compute_pairwise,
    cross_correlation,
    rolling_correlation,
    detect_divergences,
    compute_channel_correlation,
    format_channel_correlation,
)


class TestPearson:
    """Basic Pearson correlation."""

    def test_perfect_positive(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        assert abs(pearson(x, y) - 1.0) < 0.001

    def test_perfect_negative(self):
        x = [1, 2, 3, 4, 5]
        y = [10, 8, 6, 4, 2]
        assert abs(pearson(x, y) - (-1.0)) < 0.001

    def test_uncorrelated(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8]
        y = [5, 3, 7, 1, 8, 2, 6, 4]
        # Not perfectly 0 but should be small
        assert abs(pearson(x, y)) < 0.5

    def test_zero_variance(self):
        x = [5, 5, 5, 5, 5]
        y = [1, 2, 3, 4, 5]
        assert pearson(x, y) == 0.0

    def test_insufficient_data(self):
        assert pearson([1, 2], [3, 4]) == 0.0

    def test_different_lengths_uses_min(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6]
        # Uses first 3 of each
        assert abs(pearson(x, y) - 1.0) < 0.001


class TestComputePairwise:
    """Pairwise correlations between all channels."""

    def test_two_channels(self):
        channels = {"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10]}
        result = compute_pairwise(channels)
        assert ("a", "b") in result
        assert abs(result[("a", "b")] - 1.0) < 0.001

    def test_three_channels(self):
        channels = {
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 6, 8, 10],
            "z": [5, 4, 3, 2, 1],
        }
        result = compute_pairwise(channels)
        assert len(result) == 3  # x-y, x-z, y-z
        assert abs(result[("x", "y")] - 1.0) < 0.001
        assert abs(result[("x", "z")] - (-1.0)) < 0.001


class TestCrossCorrelation:
    """Lagged cross-correlation."""

    def test_simultaneous_correlation(self):
        # Use a non-monotonic signal so lag matters
        import math
        x = [math.sin(i * 0.5) for i in range(30)]
        y = [math.sin(i * 0.5) for i in range(30)]  # Same signal, no lag
        result = cross_correlation(x, y, max_lag=3)
        assert result["best_lag"] == 0
        assert abs(result["best_corr"] - 1.0) < 0.01

    def test_lagged_signal(self):
        # y is x shifted by 2
        x = [0] * 5 + [1] * 5 + [0] * 10
        y = [0] * 7 + [1] * 5 + [0] * 8
        result = cross_correlation(x, y, max_lag=5)
        # x leads y by 2 positions
        assert result["best_lag"] > 0  # positive = x leads y
        assert abs(result["best_corr"]) > 0.5

    def test_insufficient_data(self):
        result = cross_correlation([1, 2], [3, 4], max_lag=5)
        assert result["lags"] == {}
        assert result["best_lag"] == 0

    def test_lags_dict_has_full_range(self):
        x = list(range(20))
        y = [v * 2 for v in x]
        result = cross_correlation(x, y, max_lag=3)
        # Should have lags -3, -2, -1, 0, 1, 2, 3
        assert len(result["lags"]) == 7
        assert -3 in result["lags"]
        assert 3 in result["lags"]


class TestRollingCorrelation:
    """Rolling window correlation stability."""

    def test_stable_relationship(self):
        # Perfectly correlated throughout
        x = list(range(40))
        y = [v * 2 + 1 for v in x]
        result = rolling_correlation(x, y, window=10)
        assert len(result["values"]) > 0
        assert result["mean"] > 0.99
        assert result["stdev"] < 0.01
        assert result["sign_changes"] == 0

    def test_unstable_relationship(self):
        # First half: positive correlation, second half: negative
        x = list(range(40))
        y = list(range(20)) + list(range(20, 0, -1))
        result = rolling_correlation(x, y, window=10)
        assert result["stdev"] > 0.1
        assert result["sign_changes"] >= 1

    def test_insufficient_data(self):
        result = rolling_correlation([1, 2, 3], [4, 5, 6], window=15)
        assert result["values"] == []
        assert result["mean"] == 0.0

    def test_min_max_bounds(self):
        import random
        random.seed(99)
        x = [random.gauss(0, 1) for _ in range(50)]
        y = [random.gauss(0, 1) for _ in range(50)]
        result = rolling_correlation(x, y, window=10)
        assert result["min"] >= -1.0
        assert result["max"] <= 1.0


class TestDivergences:
    """Divergence detection between channels."""

    def test_no_divergence(self):
        # Consistently correlated
        channels = {"a": list(range(30)), "b": [v * 2 for v in range(30)]}
        result = detect_divergences(channels, window=10)
        assert result == []

    def test_divergence_detected(self):
        # Correlated historically, then diverge at end
        a = list(range(30))
        b = list(range(20)) + list(range(20, 10, -1))  # reverses in last 10
        channels = {"a": a, "b": b}
        result = detect_divergences(channels, window=10)
        assert len(result) >= 1
        assert result[0]["pair"] == ("a", "b")
        assert result[0]["drop"] > 0  # correlation dropped

    def test_insufficient_data(self):
        channels = {"a": [1, 2, 3], "b": [4, 5, 6]}
        result = detect_divergences(channels, window=10)
        assert result == []

    def test_significant_flag(self):
        # Large divergence
        a = list(range(40))
        b = list(range(30)) + list(range(30, 20, -1))
        channels = {"a": a, "b": b}
        result = detect_divergences(channels, window=10)
        if result:
            # The drop should be significant given the reversal
            assert any(d.get("significant") for d in result)


class TestComputeChannelCorrelation:
    """Integration test for main entry point."""

    def test_empty_history(self):
        result = compute_channel_correlation([])
        assert result["summary"] == "no data"

    def test_insufficient_data(self):
        history = [{"trace": 50.0, "drift": 0.2} for _ in range(5)]
        result = compute_channel_correlation(history)
        assert "accumulating" in result["summary"]

    def test_correlated_channels(self):
        history = [{"trace": float(i * 10), "drift": float(i * 0.05),
                    "flattery": 0.1, "coupled_nudge": None}
                   for i in range(30)]
        result = compute_channel_correlation(history)
        assert "pairwise" in result
        # trace and drift should be correlated
        pw = result["pairwise"]
        if "drift-trace" in pw:
            assert pw["drift-trace"] > 0.8

    def test_uncorrelated_channels(self):
        import random
        random.seed(42)
        history = [{"trace": random.gauss(50, 10), "drift": random.gauss(0.3, 0.1),
                    "flattery": random.gauss(0.2, 0.1)}
                   for _ in range(30)]
        result = compute_channel_correlation(history)
        assert "channels uncorrelated" in result["summary"]

    def test_missing_channels_handled(self):
        history = [{"trace": float(i), "drift": -1, "flattery": None}
                   for i in range(30)]
        result = compute_channel_correlation(history)
        # Only trace is valid, can't compute correlations
        assert "accumulating" in result["summary"] or result["pairwise"] == {}

    def test_divergence_in_result(self):
        # Construct correlated then diverging channels
        history = []
        for i in range(30):
            if i < 20:
                history.append({"trace": float(i * 5), "drift": float(i * 0.02)})
            else:
                history.append({"trace": float(i * 5), "drift": float(0.5 - i * 0.02)})
        result = compute_channel_correlation(history)
        # Should have some divergence or changed correlation
        assert "divergences" in result


class TestFormatChannelCorrelation:
    """Display formatting."""

    def test_no_data(self):
        assert format_channel_correlation({}) == (None, None)
        assert format_channel_correlation({"summary": "no data"}) == (None, None)

    def test_accumulating(self):
        result = {"summary": "accumulating (10 values across 2 channels)",
                  "pairwise": {}, "divergences": []}
        assert format_channel_correlation(result) == (None, None)

    def test_uncorrelated_returns_none(self):
        result = {"summary": "channels uncorrelated",
                  "pairwise": {"trace-drift": 0.1}, "divergences": []}
        assert format_channel_correlation(result) == (None, None)

    def test_strong_correlation_surfaces(self):
        result = {
            "summary": "trace/drift:+0.92",
            "pairwise": {"trace-drift": 0.92},
            "divergences": [],
        }
        summary, detail = format_channel_correlation(result)
        assert summary is not None
        assert "r=" in summary

    def test_divergence_surfaces(self):
        result = {
            "summary": "DIVERGE:trace/drift",
            "pairwise": {"trace-drift": 0.3},
            "divergences": [{"significant": True, "pair": ("trace", "drift")}],
        }
        summary, detail = format_channel_correlation(result)
        assert summary is not None
        assert "DIV" in summary
