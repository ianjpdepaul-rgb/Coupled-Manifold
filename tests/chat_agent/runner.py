from __future__ import annotations

"""
CLI runner for the synthetic user agent.

Usage:
    python -m tests.chat_agent.runner --persona normal_user
    python -m tests.chat_agent.runner --all
    python -m tests.chat_agent.runner --analyze logs/run_2026-05-05_1430.jsonl
    python -m tests.chat_agent.runner --compare-baseline baselines/baseline.jsonl logs/latest.jsonl
    python -m tests.chat_agent.runner --all --save-baseline
"""

import argparse
import datetime
import json
import shutil
import time
import sys
from pathlib import Path

from .agent import GracefulAgent
from .analyzer import analyze_run, compare_runs
from .personas import ALL_PERSONAS

LOGS_DIR = Path(__file__).parent / "logs"
BASELINES_DIR = Path(__file__).parent / "baselines"


def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _run_persona(name: str, agent: GracefulAgent) -> dict:
    """Run a single persona and return its result dict."""
    # Reset per-run state between personas (agent is reused across the suite)
    agent.turn_log = []
    agent.errors = []
    fn = ALL_PERSONAS[name]
    print(f"  [{name}] running...", end="", flush=True)
    t0 = time.time()
    try:
        result = fn(agent)
    except Exception as e:
        result = {
            "persona": name,
            "error": str(e),
            "turns": 0,
            "errors": [str(e)],
            "turn_log": [],
        }
    elapsed = time.time() - t0
    result["wall_time"] = round(elapsed, 2)
    n_turns = len(result.get("turn_log", []))
    n_errors = len(result.get("errors", []))
    status = "OK" if n_errors == 0 else f"{n_errors} errors"
    print(f" {n_turns} turns, {status} ({elapsed:.1f}s)")
    return result


def run_personas(names: list[str], base_url: str) -> Path:
    """Run the given personas and write a JSONL log file.  Returns the log path."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    agent = GracefulAgent(base_url)
    print(f"Connecting to {base_url}...")
    if not agent.wait_ready(timeout=30):
        print("ERROR: server not reachable.  Start it with `make start-test`.", file=sys.stderr)
        sys.exit(1)
    print("Server ready.\n")

    ts = _timestamp()
    log_path = LOGS_DIR / f"run_{ts}.jsonl"

    with open(log_path, "w") as f:
        for name in names:
            result = _run_persona(name, agent)
            f.write(json.dumps(result, default=str) + "\n")
            f.flush()

    print(f"\nLog written: {log_path}")
    return log_path


def main():
    parser = argparse.ArgumentParser(
        description="Synthetic user agent — behavioral testing for Graceful"
    )
    parser.add_argument(
        "--persona", type=str, default=None,
        help=f"Run a single persona. Choices: {', '.join(ALL_PERSONAS.keys())}",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all personas",
    )
    parser.add_argument(
        "--analyze", type=str, default=None, metavar="LOG",
        help="Analyze a JSONL log file",
    )
    parser.add_argument(
        "--save-baseline", action="store_true",
        help="Copy the run log to baselines/ after running",
    )
    parser.add_argument(
        "--compare-baseline", nargs=2, metavar=("BASELINE", "CURRENT"),
        help="Compare a current run against a baseline",
    )
    parser.add_argument(
        "--url", type=str, default="http://localhost:7860",
        help="Graceful server URL (default: http://localhost:7860)",
    )

    args = parser.parse_args()

    # ── Analyze mode ──
    if args.analyze:
        result = analyze_run(args.analyze)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["pass"] else 1)

    # ── Compare mode ──
    if args.compare_baseline:
        result = compare_runs(args.compare_baseline[0], args.compare_baseline[1])
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["pass"] else 1)

    # ── Run mode ──
    if args.persona:
        if args.persona not in ALL_PERSONAS:
            print(f"Unknown persona: {args.persona}", file=sys.stderr)
            print(f"Available: {', '.join(ALL_PERSONAS.keys())}", file=sys.stderr)
            sys.exit(1)
        names = [args.persona]
    elif args.all:
        names = list(ALL_PERSONAS.keys())
    else:
        parser.print_help()
        sys.exit(0)

    print(f"Running {len(names)} persona(s)...")
    log_path = run_personas(names, args.url)

    # Analyze the run
    print("\n── Analysis ──")
    result = analyze_run(log_path)
    for s in result["summary"]:
        status = "PASS" if s["errors"] == 0 else f"WARN ({s['errors']} errors)"
        print(f"  {s['persona']:30s} {s['turns']:2d} turns  "
              f"avg {s['avg_elapsed']:.1f}s  {status}")

    if result["anomalies"]:
        print(f"\n  Anomalies ({len(result['anomalies'])}):")
        for a in result["anomalies"]:
            print(f"    [{a['persona']}] {a['type']}: {a.get('detail', a.get('msg', ''))}")

    verdict = "PASS" if result["pass"] else "FAIL"
    print(f"\n  Result: {verdict}")

    # Save baseline
    if args.save_baseline:
        BASELINES_DIR.mkdir(parents=True, exist_ok=True)
        baseline_path = BASELINES_DIR / "baseline.jsonl"
        shutil.copy2(log_path, baseline_path)
        print(f"  Baseline saved: {baseline_path}")

    sys.exit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
