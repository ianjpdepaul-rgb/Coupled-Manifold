"""Characterization tests for trace_analytics.py."""

import json
from trace_analytics import TraceAnalytics, format_spectrum_report


class TestTraceAnalytics:
    def test_empty_traces(self, tmp_data_dir):
        ta = TraceAnalytics(str(tmp_data_dir))
        summary = ta.session_summary()
        assert summary["sessions"] == 0
        assert summary["total_turns"] == 0

    def test_load_traces(self, tmp_data_dir):
        traces_path = tmp_data_dir / "logs" / "traces.jsonl"
        traces_path.parent.mkdir(parents=True, exist_ok=True)
        with open(traces_path, "w") as f:
            for i in range(5):
                f.write(json.dumps({"session": "s1", "turn": i, "trace": 50.0 + i, "mode": "lora", "trend": "stable"}) + "\n")

        ta = TraceAnalytics(str(tmp_data_dir))
        traces = ta.load_all_traces()
        assert len(traces) == 5

    def test_session_summary(self, tmp_data_dir):
        traces_path = tmp_data_dir / "logs" / "traces.jsonl"
        traces_path.parent.mkdir(parents=True, exist_ok=True)
        with open(traces_path, "w") as f:
            for i in range(5):
                f.write(json.dumps({"session": "s1", "turn": i, "trace": 50.0, "mode": "lora", "trend": "stable"}) + "\n")

        ta = TraceAnalytics(str(tmp_data_dir))
        summary = ta.session_summary()
        assert summary["sessions"] == 1
        assert summary["total_turns"] == 5
        assert abs(summary["global_mean"] - 50.0) < 0.01

    def test_collapse_patterns(self, tmp_data_dir):
        traces_path = tmp_data_dir / "logs" / "traces.jsonl"
        traces_path.parent.mkdir(parents=True, exist_ok=True)
        with open(traces_path, "w") as f:
            # Create a collapse pattern: 60, 30, -80
            for trace_val in [60, 30, -80]:
                f.write(json.dumps({"session": "s1", "turn": 0, "trace": trace_val}) + "\n")

        ta = TraceAnalytics(str(tmp_data_dir))
        collapses = ta.collapse_patterns()
        assert len(collapses) == 1

    def test_format_report(self, tmp_data_dir):
        ta = TraceAnalytics(str(tmp_data_dir))
        report = ta.format_report()
        assert "Cross-Session Trace Analytics" in report


class TestFormatSpectrumReport:
    def test_error_case(self):
        report = format_spectrum_report({"error": "no adapters found", "n_adapters": 0})
        assert "no adapters found" in report
