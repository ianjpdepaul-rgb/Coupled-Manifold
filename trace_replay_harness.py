#!/usr/bin/env python3
"""
Trace Replay Harness — exercises the None contract across all trace consumers.

Requires: GRACEFUL_TEST_MODE=1 server running on localhost:7860.
Injects {-500, -50, 0.0, 50, 500, None} through /api/_test/trace_inject.
Reports per-consumer pass/fail for each value.

Expected pre-commit-3 failures:
  - learn_gate: crashes on None (trace > -50 TypeError)
  - interoceptive_block: crashes on None (t["trace"] > 0 TypeError)
  - temp_adjust: crashes on None (tv[-1] - tv[0] TypeError)

These are real bugs the harness is designed to catch. They get fixed in commit 3.

Usage:
    # Terminal 1: start server in test mode
    GRACEFUL_TEST_MODE=1 python app.py

    # Terminal 2: run harness
    python trace_replay_harness.py [--base http://127.0.0.1:7860]

    # Or via Makefile
    make test-harness
"""

import argparse
import json
import subprocess
import sys
import signal
import time
import os

try:
    import httpx
except ImportError:
    print("pip install httpx")
    sys.exit(1)

parser = argparse.ArgumentParser(description="Trace replay harness")
parser.add_argument("--base", default="http://127.0.0.1:7860", help="Server base URL")
parser.add_argument("--managed", action="store_true",
                    help="Start/stop server automatically (GRACEFUL_TEST_MODE=1)")
parser.add_argument("--port", type=int, default=7860, help="Port for managed server")
parser.add_argument("--verbose", "-v", action="store_true", help="Print full consumer details")
args = parser.parse_args()

BASE = args.base
INJECT_URL = f"{BASE}/api/_test/trace_inject"

# Test values: the full range the trace contract must handle
TEST_TRACES = [
    (-500,  "extreme negative (below absolute floor)"),
    (-50,   "moderate negative (typical pathological)"),
    (0.0,   "zero (real measured zero, not contamination)"),
    (50,    "moderate positive (healthy)"),
    (500,   "extreme positive"),
    (None,  "null (no measurement / skipped)"),
]

# Consumers that are expected to fail pre-commit-3 on None input.
# Map: consumer_name -> set of trace values that cause expected failure.
EXPECTED_FAILURES = {
    "learn_gate":           {None},     # trace > -50 TypeError
    "interoceptive_block":  {None},     # t["trace"] > 0 TypeError
    "temp_adjust":          {None},     # tv[-1] - tv[0] TypeError
}


def alive():
    for attempt in range(5):
        try:
            r = httpx.get(f"{BASE}/api/health", timeout=5 + attempt * 3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def check_endpoint_exists():
    """Verify the test injection endpoint is registered."""
    try:
        r = httpx.post(INJECT_URL, json={"trace": 0.0, "turn": 0}, timeout=10)
        if r.status_code == 200:
            return True
        if r.status_code == 404:
            print("ERROR: /api/_test/trace_inject not found.")
            print("       Server must be started with GRACEFUL_TEST_MODE=1")
            return False
        print(f"ERROR: unexpected status {r.status_code}: {r.text[:200]}")
        return False
    except Exception as e:
        print(f"ERROR: could not reach endpoint: {e}")
        return False


def inject(trace_val, turn):
    """Inject a trace value and return the response dict."""
    body = {"trace": trace_val, "turn": turn}
    r = httpx.post(INJECT_URL, json=body, timeout=15)
    assert r.status_code == 200, f"inject failed: {r.status_code} {r.text[:200]}"
    return r.json()


def run_harness():
    results = []
    all_consumers = set()
    pass_count = 0
    fail_count = 0
    expected_fail_count = 0

    print("=" * 70)
    print("  TRACE REPLAY HARNESS")
    print("=" * 70)
    print(f"  Target: {BASE}")
    print(f"  Test values: {len(TEST_TRACES)}")
    print()

    for i, (trace_val, desc) in enumerate(TEST_TRACES):
        turn = i + 1
        trace_label = "None" if trace_val is None else str(trace_val)
        print(f"  [{turn}/{len(TEST_TRACES)}] trace={trace_label:>6s}  ({desc})")

        try:
            resp = inject(trace_val, turn)
        except Exception as e:
            print(f"    INJECT FAILED: {e}")
            fail_count += 1
            results.append({"trace": trace_val, "error": str(e)})
            continue

        consumers = resp.get("consumers", {})

        for name, result in sorted(consumers.items()):
            all_consumers.add(name)
            ok = result.get("ok", False)
            is_expected_fail = (
                not ok
                and name in EXPECTED_FAILURES
                and trace_val in EXPECTED_FAILURES[name]
            )

            if ok:
                symbol = "\u2713"
                pass_count += 1
            elif is_expected_fail:
                symbol = "\u2717*"
                expected_fail_count += 1
            else:
                symbol = "\u2717"
                fail_count += 1

            detail = ""
            if args.verbose:
                if ok:
                    # Show key values
                    for k in ["mode", "value", "gate_b_passed", "tslope", "length"]:
                        if k in result:
                            detail = f"  -> {k}={result[k]}"
                            break
                else:
                    detail = f"  -> {result.get('error', '?')}"

            if not ok or args.verbose:
                print(f"    {symbol} {name:30s}{detail}")

        results.append({"trace": trace_val, "consumers": consumers})

    # -- Specific assertions --
    print()
    print("-" * 70)
    print("  ASSERTIONS")
    print("-" * 70)

    assertion_failures = []

    # A1: No 500 errors (all inject calls succeeded)
    inject_ok = all("error" not in r for r in results)
    _a1 = "\u2713" if inject_ok else "\u2717"
    print(f"  {_a1} No HTTP 500 errors on inject")
    if not inject_ok:
        assertion_failures.append("HTTP 500 on inject")

    # A2: None -> "?" in status string
    none_result = results[-1]  # None is last
    none_status = none_result.get("consumers", {}).get("status_str", {})
    if none_status.get("ok"):
        status_val = none_status.get("value", "")
        has_question = "?" in status_val.split("\u00b7")[1] if len(status_val.split("\u00b7")) > 1 else "?" in status_val
        _a2 = "\u2713" if has_question else "\u2717"
        print(f"  {_a2} None trace -> '?' in status string")
        if not has_question:
            assertion_failures.append("None trace missing '?' in status")
    else:
        print(f"  \u2717 None trace -> status_str build failed: {none_status.get('error')}")
        assertion_failures.append("status_str crashed on None")

    # A3: Real floats -> numeric in status string
    for r in results[:-1]:  # all except None
        trace_val = r.get("trace")
        status = r.get("consumers", {}).get("status_str", {})
        if status.get("ok"):
            parts = status["value"].split("\u00b7")
            if len(parts) >= 2:
                trace_part = parts[1].strip()
                try:
                    float(trace_part)
                except ValueError:
                    _a3 = "\u2717"
                    print(f"  {_a3} trace={trace_val} -> non-numeric in status: '{trace_part}'")
                    assertion_failures.append(f"trace={trace_val} non-numeric in status")
                    break
    else:
        print(f"  \u2713 Real float traces -> numeric in status string")

    # A4: session_log entry has correct keys
    for r in results:
        sl = r.get("consumers", {}).get("session_log", {})
        if sl.get("ok"):
            entry = sl.get("entry", {})
            required_keys = {"turn", "trace", "mode", "absolute_floor_triggered",
                             "sustained_negative_triggered", "trace_compute_ms"}
            missing = required_keys - set(entry.keys())
            if missing:
                print(f"  \u2717 session_log missing keys: {missing}")
                assertion_failures.append(f"session_log missing: {missing}")
                break
    else:
        print(f"  \u2713 session_log entries have all required keys")

    # A5: learn_gate.gate_b_passed is False when trace is None
    none_gate = none_result.get("consumers", {}).get("learn_gate", {})
    if none_gate.get("ok"):
        gate_b = none_gate.get("gate_b_passed")
        _a5 = "\u2713" if gate_b is False else "\u2717"
        print(f"  {_a5} learn_gate.gate_b_passed is False for None trace (got {gate_b})")
        if gate_b is not False:
            assertion_failures.append("learn_gate passed on None trace")
    else:
        # Expected pre-commit-3 failure
        print(f"  \u2717* learn_gate crashed on None (expected pre-commit-3): {none_gate.get('error')}")

    # A6: SnobLine.step returns valid mode for all inputs
    for r in results:
        step = r.get("consumers", {}).get("snobline_step", {})
        if not step.get("ok"):
            print(f"  \u2717 snobline_step failed for trace={r.get('trace')}: {step.get('error')}")
            assertion_failures.append(f"snobline_step crash")
            break
        if step.get("mode") not in ("lora", "anti"):
            print(f"  \u2717 snobline_step returned invalid mode: {step.get('mode')}")
            assertion_failures.append("invalid mode")
            break
    else:
        print(f"  \u2713 SnobLine.step returns valid mode for all inputs")

    # A7: get_anti_strength returns float in [0.05, 0.3] for all inputs
    for r in results:
        gas = r.get("consumers", {}).get("get_anti_strength", {})
        if not gas.get("ok"):
            print(f"  \u2717 get_anti_strength failed for trace={r.get('trace')}: {gas.get('error')}")
            assertion_failures.append("get_anti_strength crash")
            break
        v = gas.get("value")
        if not (0.05 <= v <= 0.3):
            # 0.1 is the None default, which is in range
            print(f"  \u2717 get_anti_strength out of range for trace={r.get('trace')}: {v}")
            assertion_failures.append("get_anti_strength out of range")
            break
    else:
        print(f"  \u2713 get_anti_strength returns value in [0.05, 0.3] for all inputs")

    # -- State restoration check --
    print()
    print("-" * 70)
    print("  STATE RESTORATION CHECK")
    print("-" * 70)

    try:
        state = httpx.get(f"{BASE}/api/app_state", timeout=10).json()
        live_traces = state.get("live_traces", [])
        # If state restoration works, no test values should be in live_traces
        test_traces_leaked = any(
            t in [-500.0, -50.0, 500.0] for t in live_traces if t is not None
        )
        if test_traces_leaked:
            print(f"  \u2717 Test traces leaked into live_traces: {live_traces[-6:]}")
            assertion_failures.append("state restoration failed")
        else:
            print(f"  \u2713 No test traces in live_traces (state restored)")
    except Exception as e:
        print(f"  \u2717 Could not verify state restoration: {e}")
        assertion_failures.append("state check failed")

    # -- Summary --
    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    total_consumer_checks = pass_count + fail_count + expected_fail_count
    print(f"  Consumer checks:     {total_consumer_checks}")
    print(f"    Passed:            {pass_count}")
    print(f"    Expected failures: {expected_fail_count}  (pre-commit-3)")
    print(f"    Unexpected fails:  {fail_count}")
    print(f"  Assertions:          {7 - len(assertion_failures)}/7 passed")
    if assertion_failures:
        for af in assertion_failures:
            print(f"    FAIL: {af}")
    print(f"  \u2717* = expected failure (fixed in commit 3)")
    print("=" * 70)

    if fail_count > 0 or assertion_failures:
        return 1
    return 0


def main():
    server_proc = None

    if args.managed:
        # Start server with GRACEFUL_TEST_MODE=1
        print("Starting server in test mode...")
        env = os.environ.copy()
        env["GRACEFUL_TEST_MODE"] = "1"
        env["PORT"] = str(args.port)
        server_proc = subprocess.Popen(
            [sys.executable, "app.py", "--port", str(args.port)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        # Wait for server to be ready
        if not alive():
            print("ERROR: Server did not start within timeout")
            server_proc.terminate()
            sys.exit(1)
        print(f"Server running (PID {server_proc.pid})")

    try:
        if not alive():
            print("ERROR: Server not reachable at", BASE)
            print("       Start with: GRACEFUL_TEST_MODE=1 python app.py")
            sys.exit(1)
        print("  Server reachable \u2713")

        if not check_endpoint_exists():
            sys.exit(1)
        print("  Injection endpoint available \u2713")
        print()

        exit_code = run_harness()
    finally:
        if server_proc:
            print("\nStopping managed server...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
