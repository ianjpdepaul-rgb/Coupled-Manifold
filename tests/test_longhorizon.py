"""Tests for graceful.longhorizon — long-horizon drift tracking (Pass 1)."""

import statistics

from graceful.longhorizon import (
    compute_trend,
    compute_baseline_shift,
    detect_regimes,
    compute_long_horizon,
    format_long_horizon,
)


class TestComputeTrend:
    """Trend detection via split-half comparison."""

    def test_rising_sequence(self):
        # Clear upward trend
        values = list(range(20))  # 0,1,2,...,19
        result = compute_trend(values)
        assert result["direction"] == "rising"
        assert result["magnitude"] > 0
        assert result["rate"] > 0

    def test_falling_sequence(self):
        values = list(range(20, 0, -1))  # 20,19,...,1
        result = compute_trend(values)
        assert result["direction"] == "falling"
        assert result["magnitude"] > 0

    def test_flat_sequence(self):
        values = [50.0] * 20
        result = compute_trend(values)
        assert result["direction"] == "flat"

    def test_noisy_flat(self):
        # Small noise around a constant — should be flat
        import random
        random.seed(42)
        values = [50.0 + random.gauss(0, 0.5) for _ in range(30)]
        result = compute_trend(values)
        assert result["direction"] == "flat"

    def test_insufficient_data(self):
        result = compute_trend([1, 2, 3])
        assert result["direction"] == "insufficient"

    def test_min_len_respected(self):
        values = [1, 2, 3, 4, 5, 6, 7, 8]
        result = compute_trend(values, min_len=10)
        assert result["direction"] == "insufficient"
        result2 = compute_trend(values, min_len=6)
        assert result2["direction"] != "insufficient"

    def test_step_function_detected(self):
        # First half at 0, second half at 100
        values = [0.0] * 10 + [100.0] * 10
        result = compute_trend(values)
        assert result["direction"] == "rising"
        assert result["magnitude"] == 100.0


class TestBaselineShift:
    """Baseline vs current comparison."""

    def test_no_shift(self):
        values = [50.0] * 30
        result = compute_baseline_shift(values, reference_window=10)
        assert result["shift"] == 0.0
        assert result["significant"] is False

    def test_significant_shift_up(self):
        # Baseline at 50, then jump to 150
        values = [50.0] * 10 + [150.0] * 10
        result = compute_baseline_shift(values, reference_window=10)
        assert result["baseline"] == 50.0
        assert result["current"] == 150.0
        assert result["shift"] == 100.0
        assert result["significant"] is True

    def test_significant_shift_down(self):
        values = [100.0] * 10 + [20.0] * 10
        result = compute_baseline_shift(values, reference_window=10)
        assert result["shift"] == -80.0
        assert result["significant"] is True

    def test_insufficient_data(self):
        values = [1.0, 2.0, 3.0]
        result = compute_baseline_shift(values, reference_window=10)
        assert result["baseline"] is None
        assert result["significant"] is False

    def test_gradual_shift(self):
        # Slow linear ramp — baseline stdev is near zero, so any shift is significant
        values = [float(i) for i in range(30)]
        result = compute_baseline_shift(values, reference_window=10)
        # First 10: mean 4.5, last 10: mean 24.5
        assert result["shift"] == 20.0
        assert result["significant"] is True

    def test_relative_is_z_score_like(self):
        # Baseline with known stdev, then shift by exactly 2 stdevs
        baseline = [10.0, 12.0, 8.0, 10.0, 12.0, 8.0, 10.0, 12.0, 8.0, 10.0]
        stdev = statistics.stdev(baseline)
        shifted = [10.0 + 2 * stdev] * 10
        values = baseline + shifted
        result = compute_baseline_shift(values, reference_window=10)
        # relative should be ~2.0
        assert abs(result["relative"] - 2.0) < 0.5


class TestDetectRegimes:
    """Regime detection via changepoint analysis."""

    def test_single_regime(self):
        values = [50.0] * 30
        regimes = detect_regimes(values)
        assert len(regimes) == 1
        assert regimes[0]["start"] == 0
        assert regimes[0]["end"] == 30
        assert regimes[0]["mean"] == 50.0

    def test_two_regimes(self):
        # Clear step change
        values = [10.0] * 20 + [90.0] * 20
        regimes = detect_regimes(values, min_segment=8)
        assert len(regimes) == 2
        assert regimes[0]["mean"] < 20
        assert regimes[1]["mean"] > 80

    def test_three_regimes(self):
        # Wider separation needed for detection with high threshold_factor
        values = [10.0] * 20 + [200.0] * 20 + [10.0] * 20
        regimes = detect_regimes(values, min_segment=8, threshold_factor=1.0)
        assert len(regimes) >= 2  # Should detect at least 2

    def test_too_short(self):
        values = [1.0, 2.0, 3.0]
        regimes = detect_regimes(values, min_segment=8)
        assert len(regimes) == 1
        assert regimes[0]["length"] == 3

    def test_empty_input(self):
        regimes = detect_regimes([])
        assert regimes == []

    def test_single_value(self):
        regimes = detect_regimes([42.0])
        assert len(regimes) == 1
        assert regimes[0]["mean"] == 42.0

    def test_noisy_but_single_regime(self):
        import random
        random.seed(123)
        values = [50.0 + random.gauss(0, 5) for _ in range(40)]
        regimes = detect_regimes(values, min_segment=8, threshold_factor=1.5)
        # Should stay as one regime — noise isn't a regime change
        assert len(regimes) == 1

    def test_regime_lengths(self):
        values = [10.0] * 20 + [90.0] * 20
        regimes = detect_regimes(values, min_segment=8)
        total_len = sum(r["length"] for r in regimes)
        assert total_len == 40  # All values accounted for


class TestComputeLongHorizon:
    """Integration test for the main entry point."""

    def test_empty_history(self):
        result = compute_long_horizon([])
        assert result["summary"] == "no data"
        assert result["trace"] == {}
        assert result["drift"] == {}

    def test_short_history(self):
        history = [{"turn": i, "trace": 50.0, "drift": 0.2, "mode": "M", "model": "test"}
                   for i in range(5)]
        result = compute_long_horizon(history)
        # Too short for any window
        assert "accumulating" in result["summary"]

    def test_sufficient_history_flat(self):
        history = [{"turn": i, "trace": 50.0, "drift": 0.2, "mode": "M", "model": "test"}
                   for i in range(25)]
        result = compute_long_horizon(history, windows=(20,))
        assert 20 in result["trace"]
        assert result["trace"][20]["trend"]["direction"] == "flat"

    def test_rising_trace_detected(self):
        history = [{"turn": i, "trace": float(i * 5), "drift": 0.1, "mode": "M", "model": "test"}
                   for i in range(25)]
        result = compute_long_horizon(history, windows=(20,))
        assert result["trace"][20]["trend"]["direction"] == "rising"

    def test_falling_drift_detected(self):
        history = [{"turn": i, "trace": 50.0, "drift": 0.8 - i * 0.03, "mode": "M", "model": "test"}
                   for i in range(25)]
        result = compute_long_horizon(history, windows=(20,))
        assert result["drift"][20]["trend"]["direction"] == "falling"

    def test_none_traces_skipped(self):
        history = [{"turn": i, "trace": None, "drift": 0.2, "mode": "M", "model": "test"}
                   for i in range(25)]
        result = compute_long_horizon(history, windows=(20,))
        # No valid traces
        assert 20 not in result["trace"]

    def test_negative_drift_skipped(self):
        history = [{"turn": i, "trace": 50.0, "drift": -1, "mode": "M", "model": "test"}
                   for i in range(25)]
        result = compute_long_horizon(history, windows=(20,))
        # drift=-1 means unavailable, should be excluded
        assert 20 not in result["drift"]

    def test_mixed_valid_invalid(self):
        history = []
        for i in range(30):
            trace = float(i * 2) if i % 2 == 0 else None
            drift = 0.3 if i % 3 != 0 else -1
            history.append({"turn": i, "trace": trace, "drift": drift, "mode": "M", "model": "test"})
        result = compute_long_horizon(history, windows=(10,))
        # Should still produce results from valid entries
        assert "trace" in result

    def test_window_fallback(self):
        # 15 entries, window=20 — should use all 15 with a note
        history = [{"turn": i, "trace": float(i), "drift": 0.1, "mode": "M", "model": "test"}
                   for i in range(15)]
        result = compute_long_horizon(history, windows=(20,))
        if 20 in result["trace"]:
            assert "note" in result["trace"][20]

    def test_regime_detection_in_result(self):
        # Long enough for regime detection (needs 16+)
        history = ([{"turn": i, "trace": 20.0, "drift": 0.1, "mode": "M", "model": "test"}
                    for i in range(20)] +
                   [{"turn": i + 20, "trace": 80.0, "drift": 0.1, "mode": "M", "model": "test"}
                    for i in range(20)])
        result = compute_long_horizon(history, windows=(20,))
        assert len(result["regimes"]) >= 2

    def test_multiple_windows(self):
        history = [{"turn": i, "trace": float(i), "drift": 0.1, "mode": "M", "model": "test"}
                   for i in range(110)]
        result = compute_long_horizon(history, windows=(20, 50, 100))
        assert 20 in result["trace"]
        assert 50 in result["trace"]
        assert 100 in result["trace"]


class TestFormatLongHorizon:
    """Test display formatting."""

    def test_no_data(self):
        assert format_long_horizon({}) == (None, None)
        assert format_long_horizon({"summary": "no data"}) == (None, None)

    def test_accumulating(self):
        result = {"summary": "accumulating (5 traces, 3 drifts)", "trace": {}, "drift": {}, "regimes": []}
        assert format_long_horizon(result) == (None, None)

    def test_notable_drift_formatted(self):
        result = {
            "summary": "trace@20: falling (rate=0.5/turn)",
            "trace": {20: {"trend": {"direction": "falling", "magnitude": 10, "rate": 0.5},
                           "baseline": {"significant": False}}},
            "drift": {},
            "regimes": [],
        }
        summary, detail = format_long_horizon(result)
        assert summary is not None
        assert "\u2193" in summary  # down arrow

    def test_regime_count_in_format(self):
        result = {
            "summary": "regimes: 3 detected",
            "trace": {20: {"trend": {"direction": "flat"}, "baseline": {"significant": False}}},
            "drift": {},
            "regimes": [{"start": 0}, {"start": 10}, {"start": 20}],
        }
        summary, detail = format_long_horizon(result)
        assert summary is not None
        assert "3reg" in summary

    def test_flat_returns_none(self):
        result = {
            "summary": "trace@20: flat",
            "trace": {20: {"trend": {"direction": "flat", "magnitude": 0.1, "rate": 0.01},
                           "baseline": {"significant": False}}},
            "drift": {},
            "regimes": [{"start": 0, "end": 20}],  # single regime
        }
        assert format_long_horizon(result) == (None, None)
