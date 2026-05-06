"""Tests for graceful.interoception — apparatus state visibility (Move 4)."""

from graceful.interoception import build_apparatus_block


class TestApparatusStateInContext:
    """Verify all apparatus fields appear in the interoceptive block."""

    def _make_state(self, **overrides):
        base = {
            "temperature": 0.70,
            "top_p": 0.95,
            "threshold_low": -17,
            "threshold_high": 30,
            "threshold_source": "percentile",
            "threshold_set_turn": None,
            "current_turn": 10,
            "anti_lora_active": False,
            "anti_lora_last_turn": None,
            "fw_confidence": 0.85,
            "fw_provisional": False,
            "fw_disagreements": [],
            "fw_channels": {"trace": "healthy", "drift": "on-topic",
                            "flattery": "authentic", "stability": "stable"},
        }
        base.update(overrides)
        return base

    def test_sampling_params_present(self):
        state = self._make_state(temperature=0.30, top_p=0.95)
        block, _ = build_apparatus_block(state)
        assert "temperature=0.30" in block
        assert "top_p=0.95" in block

    def test_thresholds_present(self):
        state = self._make_state(threshold_low=-17, threshold_high=30)
        block, _ = build_apparatus_block(state)
        assert "low=-17" in block
        assert "high=30" in block

    def test_threshold_source_percentile(self):
        state = self._make_state(threshold_source="percentile")
        block, _ = build_apparatus_block(state)
        assert "percentile" in block

    def test_threshold_source_model_set_with_age(self):
        state = self._make_state(
            threshold_source="model-set",
            threshold_set_turn=5,
            current_turn=17,
        )
        block, _ = build_apparatus_block(state)
        assert "model-set 12 turns ago" in block

    def test_anti_lora_inactive(self):
        state = self._make_state(anti_lora_active=False, anti_lora_last_turn=2,
                                 current_turn=10)
        block, _ = build_apparatus_block(state)
        assert "inactive (last fired 8 turns ago)" in block

    def test_anti_lora_active(self):
        state = self._make_state(anti_lora_active=True)
        block, _ = build_apparatus_block(state)
        assert "ACTIVE" in block

    def test_anti_lora_never_fired(self):
        state = self._make_state(anti_lora_active=False, anti_lora_last_turn=None)
        block, _ = build_apparatus_block(state)
        assert "anti_lora: inactive" in block
        assert "last fired" not in block

    def test_framework_confidence_present(self):
        state = self._make_state(fw_confidence=0.40, fw_provisional=True,
                                 fw_disagreements=["trace_healthy_but_drifting",
                                                    "volatile_trace_no_flattery"])
        block, _ = build_apparatus_block(state)
        assert "40%" in block
        assert "PROVISIONAL" in block
        assert "trace-healthy-but-drifting" in block
        assert "volatile-trace-no-flattery" in block

    def test_framework_confidence_confident(self):
        state = self._make_state(fw_confidence=0.90, fw_provisional=False)
        block, _ = build_apparatus_block(state)
        assert "90%" in block
        assert "CONFIDENT" in block

    def test_channels_agreeing(self):
        # trace and drift both "healthy"/"on-topic" won't agree since different values,
        # but if two channels share the same assessment they should be noted
        state = self._make_state(
            fw_channels={"trace": "healthy", "drift": "healthy",
                         "flattery": "authentic", "stability": "stable"})
        block, _ = build_apparatus_block(state)
        assert "agreeing" in block

    def test_channels_mixed(self):
        state = self._make_state(
            fw_channels={"trace": "healthy", "drift": "off-topic",
                         "flattery": "sycophantic", "stability": "volatile"})
        block, _ = build_apparatus_block(state)
        assert "mixed" in block

    def test_all_fields_present(self):
        """Comprehensive check that every required section appears."""
        state = self._make_state()
        block, _ = build_apparatus_block(state)
        assert "APPARATUS STATE:" in block
        assert "sampling:" in block
        assert "thresholds:" in block
        assert "anti_lora:" in block
        assert "framework_confidence:" in block
        assert "channels:" in block


class TestChangeAnnotation:
    """Verify change notes appear when apparatus values change between turns."""

    def _make_state(self, **overrides):
        base = {
            "temperature": 0.70,
            "top_p": 0.95,
            "threshold_low": -17,
            "threshold_high": 30,
            "threshold_source": "percentile",
            "threshold_set_turn": None,
            "current_turn": 10,
            "anti_lora_active": False,
            "anti_lora_last_turn": None,
            "fw_confidence": 0.85,
            "fw_provisional": False,
            "fw_disagreements": [],
            "fw_channels": {},
        }
        base.update(overrides)
        return base

    def test_temperature_change_noted(self):
        state1 = self._make_state(temperature=0.70)
        _, snap1 = build_apparatus_block(state1)

        state2 = self._make_state(temperature=1.10)
        block2, _ = build_apparatus_block(state2, prev_state=snap1)
        assert "APPARATUS CHANGED THIS TURN:" in block2
        assert "temperature" in block2
        assert "0.7" in block2
        assert "1.1" in block2

    def test_no_change_no_annotation(self):
        state = self._make_state()
        _, snap = build_apparatus_block(state)
        block, _ = build_apparatus_block(state, prev_state=snap)
        assert "APPARATUS CHANGED THIS TURN:" not in block

    def test_anti_lora_activation_noted(self):
        state1 = self._make_state(anti_lora_active=False)
        _, snap1 = build_apparatus_block(state1)

        state2 = self._make_state(anti_lora_active=True)
        block2, _ = build_apparatus_block(state2, prev_state=snap1)
        assert "APPARATUS CHANGED THIS TURN:" in block2
        assert "anti_lora" in block2

    def test_confidence_change_noted(self):
        state1 = self._make_state(fw_confidence=0.85)
        _, snap1 = build_apparatus_block(state1)

        state2 = self._make_state(fw_confidence=0.40)
        block2, _ = build_apparatus_block(state2, prev_state=snap1)
        assert "APPARATUS CHANGED THIS TURN:" in block2
        assert "framework_confidence" in block2

    def test_multiple_changes_all_noted(self):
        state1 = self._make_state(temperature=0.70, top_p=0.95)
        _, snap1 = build_apparatus_block(state1)

        state2 = self._make_state(temperature=1.10, top_p=0.99)
        block2, _ = build_apparatus_block(state2, prev_state=snap1)
        assert "temperature" in block2
        assert "top_p" in block2

    def test_first_turn_no_prev_no_annotation(self):
        state = self._make_state()
        block, _ = build_apparatus_block(state, prev_state=None)
        assert "APPARATUS CHANGED THIS TURN:" not in block


class TestModelCanAcknowledgeApparatus:
    """Verify the block contains actual numeric values a model could reference."""

    def _make_state(self, **overrides):
        base = {
            "temperature": 0.30,
            "top_p": 0.95,
            "threshold_low": -17,
            "threshold_high": 30,
            "threshold_source": "model-set",
            "threshold_set_turn": 5,
            "current_turn": 17,
            "anti_lora_active": False,
            "anti_lora_last_turn": 9,
            "fw_confidence": 0.40,
            "fw_provisional": True,
            "fw_disagreements": ["trace_healthy_but_drifting",
                                 "volatile_trace_no_flattery"],
            "fw_channels": {"trace": "healthy", "diversity": "healthy"},
        }
        base.update(overrides)
        return base

    def test_actual_temperature_value_in_block(self):
        state = self._make_state(temperature=0.30)
        block, _ = build_apparatus_block(state)
        assert "0.30" in block

    def test_actual_threshold_values_in_block(self):
        state = self._make_state(threshold_low=-17, threshold_high=30)
        block, _ = build_apparatus_block(state)
        assert "-17" in block
        assert "30" in block

    def test_actual_confidence_value_in_block(self):
        state = self._make_state(fw_confidence=0.40)
        block, _ = build_apparatus_block(state)
        assert "40%" in block

    def test_actual_anti_lora_age_in_block(self):
        state = self._make_state(anti_lora_last_turn=9, current_turn=17)
        block, _ = build_apparatus_block(state)
        assert "8 turns ago" in block

    def test_actual_disagreement_reasons_in_block(self):
        state = self._make_state(
            fw_disagreements=["trace_healthy_but_drifting", "volatile_trace_no_flattery"])
        block, _ = build_apparatus_block(state)
        assert "trace-healthy-but-drifting" in block
        assert "volatile-trace-no-flattery" in block

    def test_snapshot_preserves_values_for_next_turn(self):
        state = self._make_state(temperature=0.30)
        _, snap = build_apparatus_block(state)
        assert snap["temperature"] == 0.30
        assert snap["top_p"] == 0.95
        assert snap["anti_lora_active"] is False
