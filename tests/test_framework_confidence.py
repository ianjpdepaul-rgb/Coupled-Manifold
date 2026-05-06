"""Tests for graceful.framework_confidence — visible failure (Move 3)."""

from graceful.framework_confidence import (
    compute_confidence, format_confidence, build_failure_record, _compute_stability
)


class TestComputeConfidence:
    def test_all_healthy_high_confidence(self):
        result = compute_confidence(
            trace=80, drift=0.1, flattery_score=0.05,
            trace_history=[70, 75, 80, 85, 78])
        assert result["confidence"] >= 0.85
        assert result["provisional"] is False
        assert len(result["disagreements"]) == 0

    def test_all_unavailable_low_confidence(self):
        result = compute_confidence(
            trace=None, drift=-1, flattery_score=0.0,
            trace_history=[])
        assert result["confidence"] < 0.8
        assert "unavailable" in result["channels"]["trace"]

    def test_trace_healthy_flattery_high_disagrees(self):
        result = compute_confidence(
            trace=100, drift=0.2, flattery_score=0.8,
            trace_history=[90, 95, 100])
        assert "trace_healthy_but_flattering" in result["disagreements"]
        assert result["confidence"] < 0.8

    def test_pathological_trace_on_topic_disagrees(self):
        result = compute_confidence(
            trace=-80, drift=0.1, flattery_score=0.1,
            trace_history=[-70, -80, -90])
        assert "trace_pathological_but_on_topic" in result["disagreements"]

    def test_healthy_trace_off_topic_disagrees(self):
        result = compute_confidence(
            trace=100, drift=0.7, flattery_score=0.1,
            trace_history=[90, 95, 100])
        assert "trace_healthy_but_drifting" in result["disagreements"]

    def test_provisional_when_low_confidence(self):
        # Pathological trace + on-topic drift + sycophantic = multiple disagreements
        result = compute_confidence(
            trace=-80, drift=0.1, flattery_score=0.75,
            trace_history=[-50, 100, -80, 120, -60],
            coupled_nudge="-18")
        assert result["provisional"] is True

    def test_provisional_when_many_disagreements(self):
        result = compute_confidence(
            trace=100, drift=0.7, flattery_score=0.8,
            trace_history=[90, 95, 100])
        assert result["provisional"] is True
        assert len(result["disagreements"]) >= 2

    def test_strong_coupled_nudge_reduces_confidence(self):
        result_without = compute_confidence(
            trace=80, drift=0.1, flattery_score=0.05,
            trace_history=[70, 75, 80])
        result_with = compute_confidence(
            trace=80, drift=0.1, flattery_score=0.05,
            trace_history=[70, 75, 80], coupled_nudge="+18")
        assert result_with["confidence"] < result_without["confidence"]

    def test_dance_coherence_channel_coherent(self):
        result = compute_confidence(
            trace=80, drift=0.1, flattery_score=0.05,
            trace_history=[70, 75, 80], dance_coherence=0.8)
        assert result["channels"]["dance"] == "coherent"

    def test_dance_coherence_channel_incoherent(self):
        result = compute_confidence(
            trace=80, drift=0.1, flattery_score=0.05,
            trace_history=[70, 75, 80], dance_coherence=0.2)
        assert result["channels"]["dance"] == "incoherent"

    def test_dance_incoherent_reduces_confidence(self):
        result_no_dance = compute_confidence(
            trace=80, drift=0.1, flattery_score=0.05,
            trace_history=[70, 75, 80])
        result_incoherent = compute_confidence(
            trace=80, drift=0.1, flattery_score=0.05,
            trace_history=[70, 75, 80], dance_coherence=0.2)
        assert result_incoherent["confidence"] < result_no_dance["confidence"]

    def test_dance_incoherent_trace_healthy_disagrees(self):
        result = compute_confidence(
            trace=80, drift=0.1, flattery_score=0.05,
            trace_history=[70, 75, 80], dance_coherence=0.2)
        assert "dance_incoherent_trace_healthy" in result["disagreements"]

    def test_dance_none_is_building(self):
        result = compute_confidence(
            trace=80, drift=0.1, flattery_score=0.05,
            trace_history=[70, 75, 80], dance_coherence=None)
        assert result["channels"]["dance"] == "building"


class TestStability:
    def test_few_traces_building(self):
        s = _compute_stability([10, 20])
        assert s["label"] == "building"

    def test_stable_traces(self):
        s = _compute_stability([50, 52, 48, 51, 49])
        assert s["label"] == "stable"

    def test_volatile_traces(self):
        s = _compute_stability([100, -100, 150, -50, 200])
        assert s["label"] == "volatile"

    def test_shifting_traces(self):
        s = _compute_stability([50, 120, 30, 110, 20])
        assert s["label"] == "shifting"

    def test_none_traces_filtered(self):
        s = _compute_stability([None, 50, None, 52, 48, 51, 49])
        assert s["label"] == "stable"


class TestFormatConfidence:
    def test_high_confidence_returns_none(self):
        result = compute_confidence(
            trace=80, drift=0.1, flattery_score=0.05,
            trace_history=[70, 75, 80, 85, 78])
        conf_str, detail = format_confidence(result)
        assert conf_str is None

    def test_low_confidence_returns_string(self):
        result = compute_confidence(
            trace=-80, drift=0.7, flattery_score=0.8,
            trace_history=[-50, 100, -80, 120, -60])
        conf_str, detail = format_confidence(result)
        assert conf_str is not None
        assert "%" in conf_str

    def test_provisional_in_detail(self):
        result = {"confidence": 0.4, "provisional": True,
                  "channels": {}, "disagreements": ["test"]}
        conf_str, detail = format_confidence(result)
        assert "PROVISIONAL" in detail


class TestFailureRecord:
    def test_no_record_when_confident(self):
        result = {"confidence": 0.9, "provisional": False,
                  "channels": {}, "disagreements": []}
        record = build_failure_record(5, result, 80, 0.1, 0.05)
        assert record is None

    def test_record_when_provisional(self):
        result = {"confidence": 0.5, "provisional": True,
                  "channels": {"trace": "stressed"}, "disagreements": ["test_disagree"]}
        record = build_failure_record(5, result, -30, 0.6, 0.7)
        assert record is not None
        assert record["turn"] == 5
        assert record["provisional"] is True
        assert "test_disagree" in record["disagreements"]

    def test_record_when_low_confidence(self):
        result = {"confidence": 0.55, "provisional": True,
                  "channels": {}, "disagreements": []}
        record = build_failure_record(3, result, 10, 0.4, 0.3)
        assert record is not None
