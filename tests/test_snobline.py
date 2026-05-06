"""Characterization tests for graceful.snobline."""

from graceful.snobline import SnobLine


class TestSnobLineBasic:
    def test_initial_mode_is_lora(self):
        ctrl = SnobLine()
        assert ctrl.mode == "lora"

    def test_none_trace_is_noop(self):
        ctrl = SnobLine()
        result = ctrl.step(None, 1)
        assert result == "lora"
        assert len(ctrl.all_traces) == 0

    def test_zero_trace_is_processed(self):
        ctrl = SnobLine()
        ctrl.step(0.0, 1)
        assert len(ctrl.all_traces) == 1


class TestSnobLineCircuitBreakers:
    def test_absolute_floor_triggers_anti(self):
        ctrl = SnobLine()
        ctrl.step(-200, 1)
        assert ctrl.mode == "anti"

    def test_sustained_negative_triggers_anti(self):
        ctrl = SnobLine()
        for i in range(3):
            ctrl.step(-120, i + 1)
        assert ctrl.mode == "anti"


class TestSnobLinePercentile:
    def test_normal_traces_stay_lora(self):
        ctrl = SnobLine()
        for i in range(10):
            ctrl.step(50 + i, i + 1)
        assert ctrl.mode == "lora"

    def test_drop_below_percentile_triggers_anti(self):
        ctrl = SnobLine()
        # Build baseline
        for i in range(10):
            ctrl.step(100 + i, i + 1)
        # Drop well below
        ctrl.step(-50, 11)
        assert ctrl.mode == "anti"

    def test_max_anti_duration_returns_to_lora(self):
        ctrl = SnobLine(max_anti=2)
        for i in range(10):
            ctrl.step(100 + i, i + 1)
        ctrl.step(-50, 11)  # triggers anti
        assert ctrl.mode == "anti"
        ctrl.step(-50, 12)  # anti_count=1
        ctrl.step(-50, 13)  # anti_count=2 → back to lora
        assert ctrl.mode == "lora"


class TestSnobLineTrend:
    def test_building_with_few_traces(self):
        ctrl = SnobLine()
        ctrl.step(10, 1)
        label, avg, slope = ctrl.trend()
        assert label == "building"

    def test_stable_trend(self):
        ctrl = SnobLine()
        for i in range(10):
            ctrl.step(50, i + 1)
        label, avg, slope = ctrl.trend()
        assert label == "stable"


class TestSnobLineTermination:
    def test_sustained_pathological_terminates(self):
        ctrl = SnobLine()
        ctrl.consec_patho = 3
        terminate, reason = ctrl.should_terminate()
        assert terminate is True
        assert reason == "sustained_pathological"

    def test_normal_does_not_terminate(self):
        ctrl = SnobLine()
        for i in range(10):
            ctrl.step(50, i + 1)
        terminate, _ = ctrl.should_terminate()
        assert terminate is False


class TestSnobLineManualMode:
    def test_manual_override(self):
        ctrl = SnobLine()
        ctrl.manual_mode = "anti"
        result = ctrl.step(1000, 1)
        assert result == "anti"
        assert len(ctrl.all_traces) == 0  # manual mode skips processing


class TestSnobLineCitationalRegulation:
    """Tests for model-writable threshold adjustment (Move 1)."""

    def test_set_thresholds_applies_in_lora_mode(self):
        ctrl = SnobLine()
        result = ctrl.set_thresholds(-120, 8, "high-stakes turn")
        assert result["applied"] is True
        assert ctrl.model_low == -120
        assert ctrl.model_high == 8
        assert len(ctrl.thresholds_history) == 1

    def test_set_thresholds_deferred_in_anti_mode(self):
        ctrl = SnobLine()
        ctrl.mode = "anti"
        result = ctrl.set_thresholds(-120, 8, "tightening")
        assert result["applied"] is False
        assert "anti-LoRA active" in result["defer_reason"]
        assert ctrl.model_low is None  # not applied yet
        assert len(ctrl._threshold_queue) == 1

    def test_set_thresholds_deferred_at_high_patho(self):
        ctrl = SnobLine()
        ctrl.consec_patho = 2
        result = ctrl.set_thresholds(-80, 10, "exploring")
        assert result["applied"] is False
        assert "patho" in result["defer_reason"]

    def test_queue_drains_on_recovery(self):
        ctrl = SnobLine()
        # Build baseline
        for i in range(10):
            ctrl.step(100 + i, i + 1)
        # Trigger anti
        ctrl.step(-50, 11)
        assert ctrl.mode == "anti"
        # Queue a threshold adjustment while in anti
        ctrl.set_thresholds(-90, 15, "post-recovery plan")
        assert ctrl.model_low is None
        # Max duration recovery (2 turns in anti)
        ctrl.step(-50, 12)  # anti_count=1
        ctrl.step(-50, 13)  # anti_count=2 → lora, queue drained
        assert ctrl.mode == "lora"
        assert ctrl.model_low == -90
        assert ctrl.model_high == 15

    def test_model_thresholds_used_in_routing(self):
        ctrl = SnobLine()
        # Build baseline around 100
        for i in range(10):
            ctrl.step(100, i + 1)
        # Set tight model thresholds
        ctrl.set_thresholds(-10, 5, "tight for testing")
        # Trace of 50 is above percentile low but potentially could trigger
        # with very tight model thresholds — trace=50 is above model_low=-10
        assert ctrl.mode == "lora"
        ctrl.step(50, 11)
        assert ctrl.mode == "lora"  # 50 > -10, stays lora
        # Drop below model_low
        ctrl.step(-20, 12)
        assert ctrl.mode == "anti"

    def test_get_thresholds_returns_model_values(self):
        ctrl = SnobLine()
        for i in range(10):
            ctrl.step(100, i + 1)
        ctrl.set_thresholds(-100, 6, "testing get_thresholds")
        low, high = ctrl.get_thresholds()
        assert low == -100.0
        assert high == 6.0

    def test_reset_thresholds(self):
        ctrl = SnobLine()
        ctrl.set_thresholds(-80, 10, "temp")
        assert ctrl.model_low == -80
        ctrl.reset_thresholds()
        assert ctrl.model_low is None
        assert ctrl.model_high is None
        assert len(ctrl.thresholds_history) == 2  # set + reset

    def test_validation_clamps_ranges(self):
        ctrl = SnobLine()
        result = ctrl.set_thresholds(-300, 100, "extreme")
        assert result["applied"] is True
        assert ctrl.model_low == -200  # clamped
        assert ctrl.model_high == 50   # clamped

    def test_validation_rejects_low_gte_high(self):
        ctrl = SnobLine()
        result = ctrl.set_thresholds(0, 0, "invalid")
        assert result["applied"] is False
        assert ctrl.model_low is None

    def test_absolute_floor_overrides_model_thresholds(self):
        """_ABSOLUTE_FLOOR at -150 fires regardless of model thresholds."""
        ctrl = SnobLine()
        for i in range(5):
            ctrl.step(100, i + 1)
        ctrl.set_thresholds(-200, 5, "lenient low")
        # Even with model_low=-200, absolute floor at -150 still fires
        ctrl.step(-160, 6)
        assert ctrl.mode == "anti"

    def test_only_last_queued_adjustment_applied(self):
        ctrl = SnobLine()
        ctrl.mode = "anti"
        ctrl.set_thresholds(-120, 8, "first attempt")
        ctrl.set_thresholds(-80, 12, "second attempt")
        assert len(ctrl._threshold_queue) == 2
        ctrl._drain_threshold_queue()
        assert ctrl.model_low == -80   # last one wins
        assert ctrl.model_high == 12


class TestSnobLineAntiStrength:
    def test_none_trace_default(self):
        ctrl = SnobLine()
        assert ctrl.get_anti_strength(None) == 0.1

    def test_strength_increases_with_depth(self):
        ctrl = SnobLine()
        for i in range(10):
            ctrl.step(100, i + 1)
        s_shallow = ctrl.get_anti_strength(90)
        s_deep = ctrl.get_anti_strength(-200)
        assert s_deep > s_shallow
