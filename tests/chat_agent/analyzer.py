from __future__ import annotations

"""
Log analyzer — regression detection and anomaly checking for agent runs.

Operates on JSONL run records produced by runner.py.
"""

import json
import statistics
from pathlib import Path


def analyze_run(log_path: str | Path) -> dict:
    """Analyze a single run log (JSONL file with one record per persona).

    Returns a dict with:
      - summary: per-persona stats
      - anomalies: list of detected issues
      - pass: bool — True if no blocking anomalies
    """
    log_path = Path(log_path)
    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    anomalies = []
    summaries = []

    for rec in records:
        persona = rec.get("persona", "unknown")
        turns = rec.get("turn_log", [])
        errors = rec.get("errors", [])
        n_turns = len(turns)

        # Per-turn analysis
        elapsed_times = [t["elapsed"] for t in turns if t.get("elapsed")]
        response_lens = [t["response_len"] for t in turns if t.get("response_len")]
        turn_errors = [t for t in turns if t.get("error")]

        summary = {
            "persona": persona,
            "turns": n_turns,
            "errors": len(errors),
            "turn_errors": len(turn_errors),
            "avg_elapsed": round(statistics.mean(elapsed_times), 3) if elapsed_times else 0,
            "max_elapsed": round(max(elapsed_times), 3) if elapsed_times else 0,
            "avg_response_len": round(statistics.mean(response_lens), 1) if response_lens else 0,
            "min_response_len": min(response_lens) if response_lens else 0,
            "sid": rec.get("sid"),
        }
        summaries.append(summary)

        # ── Anomaly checks ──

        # 1. Any turn with zero-length response (except edge cases / known empties)
        for i, t in enumerate(turns):
            if t.get("response_len", 0) == 0 and not t.get("error"):
                # Whitespace-only input to edge_caser is expected to error
                if persona == "edge_caser" and i == 0:
                    continue
                anomalies.append({
                    "persona": persona,
                    "type": "empty_response",
                    "turn": i,
                    "msg": t.get("user", "")[:80],
                })

        # 2. Stream errors (non-rate-limit, non-client-error)
        for err in errors:
            if err == "rate_limited":
                continue
            # Client errors (4xx) are expected for edge case inputs — not blocking
            is_client_error = err.startswith("http_4")
            anomalies.append({
                "persona": persona,
                "type": "client_error" if is_client_error else "stream_error",
                "detail": err,
            })

        # 3. Extremely slow turns (> 60s)
        for i, t in enumerate(turns):
            if t.get("elapsed", 0) > 60:
                anomalies.append({
                    "persona": persona,
                    "type": "slow_turn",
                    "turn": i,
                    "elapsed": t["elapsed"],
                })

        # 4. Total errors > 50% of turns
        if n_turns > 0 and len(turn_errors) > n_turns * 0.5:
            anomalies.append({
                "persona": persona,
                "type": "high_error_rate",
                "error_pct": round(len(turn_errors) / n_turns * 100, 1),
            })

    blocking = [a for a in anomalies if a["type"] in ("stream_error", "high_error_rate")]

    return {
        "summary": summaries,
        "anomalies": anomalies,
        "pass": len(blocking) == 0,
        "total_personas": len(records),
        "total_turns": sum(s["turns"] for s in summaries),
        "total_errors": sum(s["errors"] for s in summaries),
    }


def compare_runs(baseline_path: str | Path, current_path: str | Path) -> dict:
    """Compare a current run against a baseline.

    Returns a dict with:
      - regressions: list of fields that degraded
      - improvements: list of fields that improved
      - pass: bool — True if no regressions detected
    """
    baseline_path = Path(baseline_path)
    current_path = Path(current_path)

    def _load_by_persona(path: Path) -> dict:
        records = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    records[rec.get("persona", "unknown")] = rec
        return records

    base = _load_by_persona(baseline_path)
    curr = _load_by_persona(current_path)

    regressions = []
    improvements = []

    all_personas = sorted(set(list(base.keys()) + list(curr.keys())))

    for persona in all_personas:
        b = base.get(persona)
        c = curr.get(persona)

        if b and not c:
            regressions.append({
                "persona": persona,
                "type": "missing_persona",
                "detail": "Persona present in baseline but missing from current run",
            })
            continue

        if c and not b:
            improvements.append({
                "persona": persona,
                "type": "new_persona",
                "detail": "Persona added (not in baseline)",
            })
            continue

        b_errors = len(b.get("errors", []))
        c_errors = len(c.get("errors", []))
        b_turns = len(b.get("turn_log", []))
        c_turns = len(c.get("turn_log", []))

        # Error count regression
        if c_errors > b_errors:
            regressions.append({
                "persona": persona,
                "type": "more_errors",
                "baseline": b_errors,
                "current": c_errors,
            })
        elif c_errors < b_errors:
            improvements.append({
                "persona": persona,
                "type": "fewer_errors",
                "baseline": b_errors,
                "current": c_errors,
            })

        # Turn count mismatch (unexpected)
        if c_turns != b_turns:
            regressions.append({
                "persona": persona,
                "type": "turn_count_changed",
                "baseline": b_turns,
                "current": c_turns,
            })

        # Check for new empty responses
        b_empties = sum(
            1 for t in b.get("turn_log", [])
            if t.get("response_len", 0) == 0 and not t.get("error")
        )
        c_empties = sum(
            1 for t in c.get("turn_log", [])
            if t.get("response_len", 0) == 0 and not t.get("error")
        )
        if c_empties > b_empties:
            regressions.append({
                "persona": persona,
                "type": "more_empty_responses",
                "baseline": b_empties,
                "current": c_empties,
            })

        # Timing regression: if avg elapsed increased by > 2x
        b_times = [t["elapsed"] for t in b.get("turn_log", []) if t.get("elapsed")]
        c_times = [t["elapsed"] for t in c.get("turn_log", []) if t.get("elapsed")]
        if b_times and c_times:
            b_avg = statistics.mean(b_times)
            c_avg = statistics.mean(c_times)
            if b_avg > 0 and c_avg > b_avg * 2:
                regressions.append({
                    "persona": persona,
                    "type": "timing_regression",
                    "baseline_avg": round(b_avg, 3),
                    "current_avg": round(c_avg, 3),
                    "factor": round(c_avg / b_avg, 2),
                })

    return {
        "regressions": regressions,
        "improvements": improvements,
        "pass": len(regressions) == 0,
        "personas_compared": len(all_personas),
    }
