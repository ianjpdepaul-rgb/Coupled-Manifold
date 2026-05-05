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
