"""Tests for graceful.consolidation — sleep/consolidation passes (Pass 4)."""

import json
import os
import tempfile

from graceful.consolidation import (
    summarize_session,
    detect_patterns,
    update_baselines,
    prune_history,
    generate_report,
    run_consolidation,
    load_prior_summaries,
    load_prior_baselines,
)


class TestSummarizeSession:
    """Session summarization."""

    def test_empty_history(self):
        result = summarize_session([])
        assert result["turns"] == 0
        assert result["status"] == "empty"

    def test_basic_summary(self):
        history = [{"turn": i, "trace": float(50 + i), "drift": 0.2,
                    "mode": "M", "model": "test"} for i in range(20)]
        result = summarize_session(history)
        assert result["turns"] == 20
        assert result["first_turn"] == 0
        assert result["last_turn"] == 19
        assert "trace" in result
        assert result["trace"]["n"] == 20
        assert result["trace"]["mean"] == 59.5  # mean of 50..69

    def test_trace_statistics(self):
        history = [{"turn": i, "trace": float(i * 10), "drift": -1,
                    "mode": "M", "model": "test"} for i in range(10)]
        result = summarize_session(history)
        assert result["trace"]["min"] == 0.0
        assert result["trace"]["max"] == 90.0
        assert result["trace"]["stdev"] > 0

    def test_drift_excluded_when_negative(self):
        history = [{"turn": i, "trace": 50.0, "drift": -1,
                    "mode": "M", "model": "test"} for i in range(10)]
        result = summarize_session(history)
        assert "drift" not in result

    def test_drift_included_when_valid(self):
        history = [{"turn": i, "trace": 50.0, "drift": 0.3,
                    "mode": "M", "model": "test"} for i in range(10)]
        result = summarize_session(history)
        assert "drift" in result
        assert result["drift"]["mean"] == 0.3

    def test_mode_distribution(self):
        history = ([{"turn": i, "trace": 50.0, "drift": 0.2, "mode": "M"}
                    for i in range(5)] +
                   [{"turn": i + 5, "trace": 50.0, "drift": 0.2, "mode": "A"}
                    for i in range(3)])
        result = summarize_session(history)
        assert result["modes"] == {"M": 5, "A": 3}

    def test_start_end_means(self):
        history = [{"turn": i, "trace": float(i * 10), "drift": 0.1,
                    "mode": "M", "model": "test"} for i in range(20)]
        result = summarize_session(history)
        # First 5 traces: 0,10,20,30,40 → mean 20
        assert result["trace"]["start_mean"] == 20.0
        # Last 5 traces: 150,160,170,180,190 → mean 170
        assert result["trace"]["end_mean"] == 170.0

    def test_none_traces_skipped(self):
        history = [{"turn": i, "trace": None, "drift": 0.2, "mode": "M"}
                   for i in range(10)]
        result = summarize_session(history)
        assert result["turns"] == 10
        assert "trace" not in result


class TestDetectPatterns:
    """Cross-session pattern detection."""

    def test_single_session(self):
        result = detect_patterns([{"turns": 20, "trace": {"mean": 50}}])
        assert result["trends"] == {}

    def test_stable_trace(self):
        summaries = [{"turns": 20, "trace": {"mean": 50.0}} for _ in range(5)]
        result = detect_patterns(summaries)
        assert result["trends"]["trace"] == "stable"

    def test_rising_trace(self):
        summaries = [{"turns": 20, "trace": {"mean": float(i * 20)}} for i in range(6)]
        result = detect_patterns(summaries)
        assert result["trends"]["trace"] == "rising"

    def test_falling_trace(self):
        summaries = [{"turns": 20, "trace": {"mean": float(100 - i * 20)}} for i in range(6)]
        result = detect_patterns(summaries)
        assert result["trends"]["trace"] == "falling"

    def test_anomalous_session_detected(self):
        summaries = [{"turns": 20, "trace": {"mean": 50.0}} for _ in range(8)]
        summaries[4] = {"turns": 20, "trace": {"mean": 200.0}}  # outlier
        result = detect_patterns(summaries)
        assert 4 in result["anomalous_sessions"]

    def test_stable_ranges_computed(self):
        summaries = [{"turns": 20, "trace": {"mean": 50.0 + i}} for i in range(5)]
        result = detect_patterns(summaries)
        assert "trace" in result["stable_ranges"]
        assert result["stable_ranges"]["trace"]["low"] < result["stable_ranges"]["trace"]["high"]

    def test_drift_patterns(self):
        summaries = [{"turns": 20, "trace": {"mean": 50.0},
                      "drift": {"mean": float(i * 0.05)}} for i in range(6)]
        result = detect_patterns(summaries)
        assert "drift" in result["trends"]


class TestUpdateBaselines:
    """Baseline EMA updates."""

    def test_first_session_sets_baseline(self):
        summaries = [{"trace": {"mean": 50.0, "stdev": 10.0}}]
        result = update_baselines({}, summaries)
        assert result["trace"]["mean"] == 50.0
        assert result["trace"]["stdev"] == 10.0
        assert result["trace"]["n_sessions"] == 1

    def test_ema_update(self):
        existing = {"trace": {"mean": 50.0, "stdev": 10.0, "n_sessions": 5}}
        summaries = [{"trace": {"mean": 80.0, "stdev": 15.0}}]
        result = update_baselines(existing, summaries, decay=0.3)
        # EMA: 50 * 0.7 + 80 * 0.3 = 35 + 24 = 59
        assert abs(result["trace"]["mean"] - 59.0) < 0.1
        assert result["trace"]["n_sessions"] == 6

    def test_decay_1_replaces_completely(self):
        existing = {"trace": {"mean": 50.0, "stdev": 10.0, "n_sessions": 5}}
        summaries = [{"trace": {"mean": 100.0, "stdev": 20.0}}]
        result = update_baselines(existing, summaries, decay=1.0)
        assert abs(result["trace"]["mean"] - 100.0) < 0.1

    def test_decay_0_keeps_existing(self):
        existing = {"trace": {"mean": 50.0, "stdev": 10.0, "n_sessions": 5}}
        summaries = [{"trace": {"mean": 100.0, "stdev": 20.0}}]
        result = update_baselines(existing, summaries, decay=0.0)
        assert abs(result["trace"]["mean"] - 50.0) < 0.1

    def test_drift_baseline(self):
        summaries = [{"drift": {"mean": 0.25, "stdev": 0.05}}]
        result = update_baselines({}, summaries)
        assert "drift" in result
        assert result["drift"]["mean"] == 0.25

    def test_missing_channel_skipped(self):
        summaries = [{"turns": 20}]  # No trace or drift
        result = update_baselines({}, summaries)
        assert result == {}


class TestPruneHistory:
    """History compression."""

    def test_already_short(self):
        history = [{"turn": i, "trace": 50.0} for i in range(50)]
        result = prune_history(history, max_entries=100)
        assert len(result) == 50

    def test_prunes_to_target(self):
        history = [{"turn": i, "trace": 50.0} for i in range(500)]
        result = prune_history(history, max_entries=100)
        assert len(result) <= 100

    def test_preserves_first_and_last(self):
        history = [{"turn": i, "trace": float(i)} for i in range(200)]
        result = prune_history(history, max_entries=50)
        assert result[0]["turn"] == 0
        assert result[-1]["turn"] == 199

    def test_preserves_boundaries(self):
        # Flat with one big jump
        history = [{"turn": i, "trace": 50.0} for i in range(100)]
        history[50] = {"turn": 50, "trace": 200.0}  # big jump
        result = prune_history(history, max_entries=30, keep_boundaries=True)
        # The jump point should be preserved
        jump_turns = [h["turn"] for h in result]
        assert 50 in jump_turns or 49 in jump_turns

    def test_empty_history(self):
        assert prune_history([], max_entries=100) == []

    def test_none_traces_handled(self):
        history = [{"turn": i, "trace": None} for i in range(200)]
        # Should not crash
        result = prune_history(history, max_entries=50)
        assert len(result) <= 50


class TestGenerateReport:
    """Report generation."""

    def test_basic_report(self):
        summary = {"turns": 25, "trace": {"mean": 45.0, "stdev": 12.0,
                                           "min": 20.0, "max": 80.0,
                                           "start_mean": 30.0, "end_mean": 60.0}}
        patterns = {"trends": {"trace": "rising"}, "stable_ranges": {},
                    "anomalous_sessions": []}
        baselines = {"trace": {"mean": 50.0, "stdev": 10.0}}
        report = generate_report(summary, patterns, baselines)
        assert "25 turns" in report
        assert "45.0" in report
        assert "rising" in report
        assert "Baselines" in report

    def test_empty_summary(self):
        report = generate_report({"turns": 0, "status": "empty"}, {}, {})
        assert "0 turns" in report

    def test_includes_drift(self):
        summary = {"turns": 10, "drift": {"mean": 0.35, "max": 0.6}}
        report = generate_report(summary, {}, {})
        assert "0.35" in report


class TestRunConsolidation:
    """Full orchestration."""

    def test_full_run(self):
        with tempfile.TemporaryDirectory() as td:
            history = [{"turn": i, "trace": float(50 + i), "drift": 0.2,
                        "mode": "M", "model": "test"} for i in range(30)]
            result = run_consolidation(history, td)
            assert result["session_summary"]["turns"] == 30
            assert "report" in result
            assert len(result["pruned_history"]) <= 100

    def test_persists_to_disk(self):
        with tempfile.TemporaryDirectory() as td:
            history = [{"turn": i, "trace": float(50 + i), "drift": 0.2,
                        "mode": "M", "model": "test"} for i in range(20)]
            run_consolidation(history, td)

            # Check files were created
            assert os.path.isfile(os.path.join(td, "consolidation", "session_summaries.jsonl"))
            assert os.path.isfile(os.path.join(td, "consolidation", "baselines.json"))
            assert os.path.isfile(os.path.join(td, "consolidation", "reports.jsonl"))

    def test_incorporates_prior_summaries(self):
        with tempfile.TemporaryDirectory() as td:
            prior = [{"turns": 20, "trace": {"mean": 30.0, "stdev": 5.0}}]
            history = [{"turn": i, "trace": 80.0, "drift": 0.1, "mode": "M"}
                       for i in range(20)]
            result = run_consolidation(history, td, prior_summaries=prior)
            # Should detect pattern across 2 sessions
            assert "trends" in result["patterns"]

    def test_updates_existing_baselines(self):
        with tempfile.TemporaryDirectory() as td:
            prior_baselines = {"trace": {"mean": 40.0, "stdev": 8.0, "n_sessions": 3}}
            history = [{"turn": i, "trace": 80.0, "drift": 0.2, "mode": "M"}
                       for i in range(20)]
            result = run_consolidation(history, td, prior_baselines=prior_baselines)
            # Baseline should move toward 80
            assert result["baselines"]["trace"]["mean"] > 40.0
            assert result["baselines"]["trace"]["n_sessions"] == 4

    def test_empty_history_consolidation(self):
        with tempfile.TemporaryDirectory() as td:
            result = run_consolidation([], td)
            assert result["session_summary"]["turns"] == 0


class TestLoadFunctions:
    """Loading prior state from disk."""

    def test_load_summaries_empty(self):
        with tempfile.TemporaryDirectory() as td:
            result = load_prior_summaries(td)
            assert result == []

    def test_load_summaries_after_write(self):
        with tempfile.TemporaryDirectory() as td:
            history = [{"turn": i, "trace": 50.0, "drift": 0.2, "mode": "M"}
                       for i in range(10)]
            run_consolidation(history, td)
            summaries = load_prior_summaries(td)
            assert len(summaries) == 1
            assert summaries[0]["turns"] == 10

    def test_load_baselines_empty(self):
        with tempfile.TemporaryDirectory() as td:
            result = load_prior_baselines(td)
            assert result == {}

    def test_load_baselines_after_write(self):
        with tempfile.TemporaryDirectory() as td:
            history = [{"turn": i, "trace": 50.0, "drift": 0.2, "mode": "M"}
                       for i in range(10)]
            run_consolidation(history, td)
            baselines = load_prior_baselines(td)
            assert "trace" in baselines

    def test_multiple_consolidations_accumulate(self):
        with tempfile.TemporaryDirectory() as td:
            # First consolidation
            h1 = [{"turn": i, "trace": 30.0, "drift": 0.2, "mode": "M"}
                   for i in range(10)]
            run_consolidation(h1, td)

            # Second consolidation using priors from disk
            prior_summaries = load_prior_summaries(td)
            prior_baselines = load_prior_baselines(td)
            h2 = [{"turn": i, "trace": 70.0, "drift": 0.3, "mode": "M"}
                   for i in range(10)]
            run_consolidation(h2, td, prior_summaries=prior_summaries,
                              prior_baselines=prior_baselines)

            # Should now have 2 summaries
            summaries = load_prior_summaries(td)
            assert len(summaries) == 2

            # Baseline should be between 30 and 70
            baselines = load_prior_baselines(td)
            assert 30 < baselines["trace"]["mean"] < 70
