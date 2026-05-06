"""Tests for graceful.coupled_trace — coupled measurement (Move 2)."""

from graceful.coupled_trace import couple, summarize, _compute_nudge


class TestCoupleBasic:
    def test_none_trace_returns_none(self):
        assert couple(None, {"keystroke_count": 50}) is None

    def test_empty_state_returns_trace(self):
        assert couple(42.0, {}) == 42.0

    def test_none_trace_empty_state(self):
        assert couple(None, {}) is None

    def test_both_present_returns_float(self):
        result = couple(50.0, {"keystroke_count": 30})
        assert isinstance(result, float)


class TestNudgeDirection:
    def test_high_engagement_nudges_positive(self):
        # Many keystrokes, fast typing → positive nudge
        hs = {"keystroke_count": 100, "typing_duration_ms": 5000, "edit_ratio": 0.02}
        nudge = _compute_nudge(hs)
        assert nudge > 0

    def test_high_struggle_nudges_negative(self):
        # High edit ratio, long pauses, lots of scrolling
        hs = {"keystroke_count": 20, "edit_ratio": 0.5, "median_pause_ms": 2000,
              "scroll_ups": 8, "read_time_ms": 60000}
        nudge = _compute_nudge(hs)
        assert nudge < 0

    def test_neutral_input_near_zero(self):
        # Baseline values — should be near zero
        hs = {"keystroke_count": 30, "edit_ratio": 0.1, "median_pause_ms": 300,
              "typing_duration_ms": 10000}
        nudge = _compute_nudge(hs)
        assert abs(nudge) < 5

    def test_nudge_bounded(self):
        # Extreme values should still be bounded
        hs_extreme_pos = {"keystroke_count": 500, "typing_duration_ms": 2000,
                          "edit_ratio": 0, "read_time_ms": 500}
        hs_extreme_neg = {"keystroke_count": 3, "edit_ratio": 0.8,
                          "median_pause_ms": 5000, "max_pause_ms": 30000,
                          "scroll_ups": 20, "read_time_ms": 120000,
                          "typing_duration_ms": 60000}
        pos = _compute_nudge(hs_extreme_pos)
        neg = _compute_nudge(hs_extreme_neg)
        assert -20 <= pos <= 20
        assert -20 <= neg <= 20


class TestCoupleIntegration:
    def test_engaged_user_raises_trace(self):
        model_trace = 50.0
        hs = {"keystroke_count": 80, "typing_duration_ms": 8000,
              "edit_ratio": 0.03, "median_pause_ms": 150}
        coupled = couple(model_trace, hs)
        assert coupled > model_trace

    def test_struggling_user_lowers_trace(self):
        model_trace = 50.0
        hs = {"keystroke_count": 10, "edit_ratio": 0.4,
              "median_pause_ms": 2000, "scroll_ups": 6, "read_time_ms": 45000}
        coupled = couple(model_trace, hs)
        assert coupled < model_trace

    def test_nudge_does_not_flip_sign_on_healthy(self):
        # Healthy trace should stay positive even with struggling user
        model_trace = 100.0
        hs = {"keystroke_count": 5, "edit_ratio": 0.6,
              "median_pause_ms": 3000, "scroll_ups": 10}
        coupled = couple(model_trace, hs)
        assert coupled > 0  # max nudge is -20, 100-20=80


class TestSummarize:
    def test_none_trace(self):
        n, d = summarize(None, {}, None)
        assert n is None and d is None

    def test_no_coupling(self):
        n, d = summarize(50.0, {}, 50.0)
        assert n is None

    def test_positive_nudge(self):
        n, d = summarize(50.0, {"keystroke_count": 80}, 55.0)
        assert n.startswith("+")
        assert "ks:80" in d

    def test_negative_nudge(self):
        n, d = summarize(50.0, {"keystroke_count": 10, "scroll_ups": 5}, 45.0)
        assert n.startswith("-")

    def test_read_time_in_detail(self):
        n, d = summarize(50.0, {"read_time_ms": 8000}, 47.0)
        assert "read:" in d

    def test_edit_ratio_in_detail(self):
        n, d = summarize(50.0, {"edit_ratio": 0.3, "keystroke_count": 20}, 45.0)
        assert "edit:" in d
