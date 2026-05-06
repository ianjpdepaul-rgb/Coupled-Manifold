"""
Consolidation / sleep passes — measurement expansion Pass 4.

Offline processing that runs between sessions or during idle periods.
Compresses, summarizes, and integrates accumulated measurements.

Passes:
  1. summarize_session — compress trace_history into session summary
  2. detect_patterns — identify recurring drift/variance/correlation patterns
  3. update_baselines — revise reference baselines from accumulated data
  4. prune_history — remove redundant entries, keep regime boundaries
  5. generate_report — produce human-readable consolidation report

Unlike Passes 1-3, this module has controlled side effects (writes to disk).
Functions still accept explicit paths rather than using globals.
"""

import json
import os
import statistics
import time
from typing import Any


# ═══════════════════════════════════════════════════
# SESSION SUMMARY
# ═══════════════════════════════════════════════════

def summarize_session(history: list[dict]) -> dict:
    """Compress a full session's trace history into a summary.

    history: list of {turn, trace, drift, mode, model, ...} entries.

    Returns a compact summary dict suitable for long-term storage.
    """
    if not history:
        return {"turns": 0, "status": "empty"}

    traces = [h["trace"] for h in history if h.get("trace") is not None
              and isinstance(h["trace"], (int, float))]
    drifts = [h["drift"] for h in history if isinstance(h.get("drift"), (int, float))
              and h["drift"] >= 0]

    summary = {
        "turns": len(history),
        "first_turn": history[0].get("turn", 0),
        "last_turn": history[-1].get("turn", 0),
        "ts": time.time(),
    }

    if traces:
        summary["trace"] = {
            "mean": round(statistics.mean(traces), 2),
            "stdev": round(statistics.stdev(traces), 2) if len(traces) >= 2 else 0.0,
            "min": round(min(traces), 2),
            "max": round(max(traces), 2),
            "n": len(traces),
            "start_mean": round(statistics.mean(traces[:5]), 2) if len(traces) >= 5 else None,
            "end_mean": round(statistics.mean(traces[-5:]), 2) if len(traces) >= 5 else None,
        }

    if drifts:
        summary["drift"] = {
            "mean": round(statistics.mean(drifts), 4),
            "stdev": round(statistics.stdev(drifts), 4) if len(drifts) >= 2 else 0.0,
            "max": round(max(drifts), 4),
            "n": len(drifts),
        }

    # Mode distribution
    modes = [h.get("mode") for h in history if h.get("mode")]
    if modes:
        mode_counts = {}
        for m in modes:
            mode_counts[m] = mode_counts.get(m, 0) + 1
        summary["modes"] = mode_counts

    return summary


# ═══════════════════════════════════════════════════
# PATTERN DETECTION
# ═══════════════════════════════════════════════════

def detect_patterns(session_summaries: list[dict]) -> dict:
    """Identify recurring patterns across multiple session summaries.

    session_summaries: list of summaries from summarize_session().

    Returns:
        trends: cross-session trend directions
        stable_ranges: value ranges that appear stable across sessions
        anomalous_sessions: indices of sessions that differ significantly
    """
    if len(session_summaries) < 2:
        return {"trends": {}, "stable_ranges": {}, "anomalous_sessions": []}

    # Extract per-session trace means
    trace_means = []
    drift_means = []
    for s in session_summaries:
        if "trace" in s and s["trace"].get("mean") is not None:
            trace_means.append(s["trace"]["mean"])
        if "drift" in s and s["drift"].get("mean") is not None:
            drift_means.append(s["drift"]["mean"])

    trends = {}
    stable_ranges = {}
    anomalous = []

    # Trace trend across sessions
    if len(trace_means) >= 3:
        mid = len(trace_means) // 2
        first_half = statistics.mean(trace_means[:mid])
        second_half = statistics.mean(trace_means[mid:])
        diff = second_half - first_half
        stdev = statistics.stdev(trace_means) if len(trace_means) >= 2 else 1.0
        if stdev > 0 and abs(diff) > stdev * 0.5:
            trends["trace"] = "rising" if diff > 0 else "falling"
        else:
            trends["trace"] = "stable"

        # Stable range
        stable_ranges["trace"] = {
            "low": round(statistics.mean(trace_means) - statistics.stdev(trace_means), 2),
            "high": round(statistics.mean(trace_means) + statistics.stdev(trace_means), 2),
        }

        # Anomalous sessions: > 2 stdev from mean
        mean_t = statistics.mean(trace_means)
        for i, tm in enumerate(trace_means):
            if abs(tm - mean_t) > 2 * stdev:
                anomalous.append(i)

    if len(drift_means) >= 3:
        mid = len(drift_means) // 2
        first_half = statistics.mean(drift_means[:mid])
        second_half = statistics.mean(drift_means[mid:])
        diff = second_half - first_half
        stdev = statistics.stdev(drift_means) if len(drift_means) >= 2 else 1.0
        if stdev > 0 and abs(diff) > stdev * 0.5:
            trends["drift"] = "rising" if diff > 0 else "falling"
        else:
            trends["drift"] = "stable"

        stable_ranges["drift"] = {
            "low": round(statistics.mean(drift_means) - statistics.stdev(drift_means), 4),
            "high": round(statistics.mean(drift_means) + statistics.stdev(drift_means), 4),
        }

    return {
        "trends": trends,
        "stable_ranges": stable_ranges,
        "anomalous_sessions": sorted(set(anomalous)),
    }


# ═══════════════════════════════════════════════════
# BASELINE UPDATE
# ═══════════════════════════════════════════════════

def update_baselines(current_baselines: dict, session_summaries: list[dict],
                     decay: float = 0.3) -> dict:
    """Revise reference baselines using exponential moving average.

    current_baselines: existing baselines {channel: {mean, stdev}}
    session_summaries: recent sessions to incorporate
    decay: weight given to new data (0-1). Higher = more responsive.

    Returns: updated baselines dict.
    """
    baselines = dict(current_baselines)

    for summary in session_summaries:
        if "trace" in summary and summary["trace"].get("mean") is not None:
            new_mean = summary["trace"]["mean"]
            new_stdev = summary["trace"].get("stdev", 0)
            if "trace" in baselines:
                old = baselines["trace"]
                baselines["trace"] = {
                    "mean": round(old["mean"] * (1 - decay) + new_mean * decay, 3),
                    "stdev": round(old["stdev"] * (1 - decay) + new_stdev * decay, 3),
                    "updated": time.time(),
                    "n_sessions": old.get("n_sessions", 0) + 1,
                }
            else:
                baselines["trace"] = {
                    "mean": round(new_mean, 3),
                    "stdev": round(new_stdev, 3),
                    "updated": time.time(),
                    "n_sessions": 1,
                }

        if "drift" in summary and summary["drift"].get("mean") is not None:
            new_mean = summary["drift"]["mean"]
            new_stdev = summary["drift"].get("stdev", 0)
            if "drift" in baselines:
                old = baselines["drift"]
                baselines["drift"] = {
                    "mean": round(old["mean"] * (1 - decay) + new_mean * decay, 4),
                    "stdev": round(old["stdev"] * (1 - decay) + new_stdev * decay, 4),
                    "updated": time.time(),
                    "n_sessions": old.get("n_sessions", 0) + 1,
                }
            else:
                baselines["drift"] = {
                    "mean": round(new_mean, 4),
                    "stdev": round(new_stdev, 4),
                    "updated": time.time(),
                    "n_sessions": 1,
                }

    return baselines


# ═══════════════════════════════════════════════════
# HISTORY PRUNING
# ═══════════════════════════════════════════════════

def prune_history(history: list[dict], max_entries: int = 100,
                  keep_boundaries: bool = True) -> list[dict]:
    """Prune history to max_entries while preserving important points.

    Keeps: first entry, last entry, regime boundaries (large jumps),
    and evenly sampled points between them.

    history: full trace history
    max_entries: target size after pruning
    keep_boundaries: if True, preserves entries where trace jumps > 2 stdev

    Returns: pruned history list.
    """
    if len(history) <= max_entries:
        return list(history)

    if not history:
        return []

    # Identify important indices
    important = {0, len(history) - 1}

    if keep_boundaries:
        traces = [h.get("trace") for h in history]
        valid_traces = [t for t in traces if t is not None and isinstance(t, (int, float))]
        if len(valid_traces) >= 4:
            stdev = statistics.stdev(valid_traces)
            threshold = stdev * 2
            for i in range(1, len(traces)):
                curr = traces[i]
                prev = traces[i - 1]
                if (curr is not None and prev is not None and
                        isinstance(curr, (int, float)) and isinstance(prev, (int, float))):
                    if abs(curr - prev) > threshold:
                        important.add(i)
                        if i > 0:
                            important.add(i - 1)

    # Fill remaining slots with evenly spaced indices
    remaining = max_entries - len(important)
    if remaining > 0:
        step = len(history) / remaining
        for i in range(remaining):
            idx = int(i * step)
            important.add(min(idx, len(history) - 1))

    # Sort and take up to max_entries
    indices = sorted(important)[:max_entries]
    return [history[i] for i in indices]


# ═══════════════════════════════════════════════════
# CONSOLIDATION REPORT
# ═══════════════════════════════════════════════════

def generate_report(session_summary: dict, patterns: dict,
                    baselines: dict) -> str:
    """Generate human-readable consolidation report.

    Returns a multi-line string summarizing what happened during
    consolidation.
    """
    lines = ["— Consolidation Report —"]

    # Session summary
    turns = session_summary.get("turns", 0)
    lines.append(f"Session: {turns} turns")

    if "trace" in session_summary:
        t = session_summary["trace"]
        lines.append(f"  Trace: mean={t['mean']} stdev={t['stdev']} range=[{t['min']}, {t['max']}]")
        if t.get("start_mean") is not None and t.get("end_mean") is not None:
            delta = t["end_mean"] - t["start_mean"]
            lines.append(f"  Session drift: {delta:+.1f} (start={t['start_mean']}, end={t['end_mean']})")

    if "drift" in session_summary:
        d = session_summary["drift"]
        lines.append(f"  Corpus drift: mean={d['mean']:.3f} max={d['max']:.3f}")

    # Cross-session patterns
    if patterns.get("trends"):
        trends_str = ", ".join(f"{k}={v}" for k, v in patterns["trends"].items())
        lines.append(f"Patterns: {trends_str}")
    if patterns.get("anomalous_sessions"):
        lines.append(f"  Anomalous sessions: {patterns['anomalous_sessions']}")

    # Baselines
    if baselines:
        bl_parts = []
        for ch, bl in baselines.items():
            bl_parts.append(f"{ch}={bl.get('mean', '?')}±{bl.get('stdev', '?')}")
        lines.append(f"Baselines: {', '.join(bl_parts)}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════
# ORCHESTRATOR — runs full consolidation pass
# ═══════════════════════════════════════════════════

def run_consolidation(history: list[dict], data_dir: str,
                      prior_summaries: list[dict] | None = None,
                      prior_baselines: dict | None = None) -> dict:
    """Run a full consolidation pass.

    history: current session's trace history
    data_dir: directory for reading/writing state
    prior_summaries: previously stored session summaries (optional)
    prior_baselines: previously stored baselines (optional)

    Returns:
        session_summary: this session's summary
        patterns: detected cross-session patterns
        baselines: updated baselines
        pruned_history: compressed history
        report: human-readable report string
    """
    # Step 1: Summarize current session
    session_summary = summarize_session(history)

    # Step 2: Gather prior summaries
    summaries = list(prior_summaries or [])
    if session_summary.get("turns", 0) > 0:
        summaries.append(session_summary)

    # Step 3: Detect patterns
    patterns = detect_patterns(summaries)

    # Step 4: Update baselines
    baselines = update_baselines(prior_baselines or {}, [session_summary])

    # Step 5: Prune history
    pruned = prune_history(history, max_entries=100)

    # Step 6: Generate report
    report = generate_report(session_summary, patterns, baselines)

    # Step 7: Persist (controlled side effect)
    _persist_consolidation(data_dir, session_summary, summaries, baselines, report)

    return {
        "session_summary": session_summary,
        "patterns": patterns,
        "baselines": baselines,
        "pruned_history": pruned,
        "report": report,
    }


def _persist_consolidation(data_dir: str, session_summary: dict,
                           all_summaries: list[dict], baselines: dict,
                           report: str):
    """Write consolidation results to disk."""
    consolidation_dir = os.path.join(data_dir, "consolidation")
    os.makedirs(consolidation_dir, exist_ok=True)

    # Append session summary to summaries.jsonl
    summaries_path = os.path.join(consolidation_dir, "session_summaries.jsonl")
    try:
        with open(summaries_path, "a") as f:
            f.write(json.dumps(session_summary, default=str) + "\n")
    except OSError:
        pass

    # Write current baselines
    baselines_path = os.path.join(consolidation_dir, "baselines.json")
    try:
        with open(baselines_path, "w") as f:
            json.dump(baselines, f, indent=2, default=str)
    except OSError:
        pass

    # Append report to reports log
    reports_path = os.path.join(consolidation_dir, "reports.jsonl")
    try:
        entry = {"ts": time.time(), "report": report}
        with open(reports_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except OSError:
        pass


# ═══════════════════════════════════════════════════
# LOADING PRIOR STATE
# ═══════════════════════════════════════════════════

def load_prior_summaries(data_dir: str) -> list[dict]:
    """Load previously stored session summaries from disk."""
    path = os.path.join(data_dir, "consolidation", "session_summaries.jsonl")
    if not os.path.isfile(path):
        return []
    summaries = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    summaries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return summaries


def load_prior_baselines(data_dir: str) -> dict:
    """Load previously stored baselines from disk."""
    path = os.path.join(data_dir, "consolidation", "baselines.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
