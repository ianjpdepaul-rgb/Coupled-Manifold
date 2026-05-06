"""
Long-horizon drift tracking — measurement expansion Pass 1.

Detects slow drift patterns that short-window analysis misses.
Operates on trace_history_live entries: {turn, trace, mode, model, drift}.

Three dimensions:
  1. Trend — systematic rise/fall over 20/50/100 turn windows
  2. Baseline shift — running mean displacement from earlier reference
  3. Regime detection — distinct operational phases

Pure functions — no side effects, no state.
"""

import statistics
from typing import Any


# ═══════════════════════════════════════════════════
# TREND ANALYSIS
# ═════════════��══════════��══════════════════════════

def compute_trend(values: list[float], min_len: int = 6) -> dict:
    """Compute linear trend over a sequence of values.

    Uses split-half comparison (mean of first half vs second half).
    More robust than regression for noisy sequences.

    Returns:
        direction: "rising", "falling", "flat"
        magnitude: absolute difference of half-means
        rate: magnitude / (len / 2) — per-turn change
    """
    if len(values) < min_len:
        return {"direction": "insufficient", "magnitude": 0.0, "rate": 0.0}

    mid = len(values) // 2
    first_half = values[:mid]
    second_half = values[mid:]

    mean_first = statistics.mean(first_half)
    mean_second = statistics.mean(second_half)
    diff = mean_second - mean_first
    magnitude = abs(diff)
    rate = magnitude / max(mid, 1)

    # Threshold: require meaningful shift relative to spread
    spread = statistics.stdev(values) if len(values) >= 2 else 1.0
    if spread == 0:
        spread = 1.0

    # Shift must be at least 0.3 stdev to count as directional
    if magnitude < spread * 0.3:
        direction = "flat"
    elif diff > 0:
        direction = "rising"
    else:
        direction = "falling"

    return {"direction": direction, "magnitude": round(magnitude, 3), "rate": round(rate, 4)}


# ══════════��═══════════════════��════════════════════
# BASELINE SHIFT
# ══════════��══════════════════════���═════════════════

def compute_baseline_shift(values: list[float], reference_window: int = 10) -> dict:
    """Compare current running mean against an earlier reference.

    reference_window: how many initial values form the baseline.

    Returns:
        baseline: mean of first reference_window values
        current: mean of last reference_window values
        shift: current - baseline
        relative: shift / baseline_stdev (z-score-like)
        significant: bool — |relative| > 2.0
    """
    if len(values) < reference_window * 2:
        return {
            "baseline": None, "current": None, "shift": 0.0,
            "relative": 0.0, "significant": False,
        }

    baseline_vals = values[:reference_window]
    current_vals = values[-reference_window:]

    baseline_mean = statistics.mean(baseline_vals)
    current_mean = statistics.mean(current_vals)
    shift = current_mean - baseline_mean

    baseline_stdev = statistics.stdev(baseline_vals) if len(baseline_vals) >= 2 else 1.0
    if baseline_stdev == 0:
        baseline_stdev = 1.0

    relative = shift / baseline_stdev

    return {
        "baseline": round(baseline_mean, 3),
        "current": round(current_mean, 3),
        "shift": round(shift, 3),
        "relative": round(relative, 2),
        "significant": abs(relative) > 2.0,
    }


# ═══════════════════════════════════════════════════
# REGIME DETECTION
# ══════���═══════��═══════════════════════════════���════

def detect_regimes(values: list[float], min_segment: int = 8,
                   threshold_factor: float = 1.5) -> list[dict]:
    """Detect distinct operational regimes (changepoints).

    Uses a simple sliding-window mean comparison to find points where
    the distribution shifts significantly.

    min_segment: minimum turns in a regime before considering a split.
    threshold_factor: how many stdevs the means must differ by.

    Returns list of regimes:
        start: index of regime start
        end: index of regime end (exclusive)
        mean: mean value in regime
        stdev: stdev in regime
        length: number of values
    """
    n = len(values)
    if n < min_segment * 2:
        if n >= 2:
            return [{
                "start": 0, "end": n,
                "mean": round(statistics.mean(values), 3),
                "stdev": round(statistics.stdev(values), 3),
                "length": n,
            }]
        elif n == 1:
            return [{"start": 0, "end": 1, "mean": values[0], "stdev": 0.0, "length": 1}]
        return []

    # Find changepoints via maximum mean-difference scan
    changepoints = _find_changepoints(values, min_segment, threshold_factor)

    # Build regimes from changepoints
    boundaries = [0] + changepoints + [n]
    regimes = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        segment = values[start:end]
        if len(segment) >= 2:
            regimes.append({
                "start": start, "end": end,
                "mean": round(statistics.mean(segment), 3),
                "stdev": round(statistics.stdev(segment), 3),
                "length": len(segment),
            })
        elif len(segment) == 1:
            regimes.append({
                "start": start, "end": end,
                "mean": segment[0], "stdev": 0.0,
                "length": 1,
            })

    return regimes


def _find_changepoints(values: list[float], min_segment: int,
                       threshold_factor: float) -> list[int]:
    """Recursive binary segmentation for changepoint detection."""
    n = len(values)
    if n < min_segment * 2:
        return []

    overall_stdev = statistics.stdev(values) if len(values) >= 2 else 0.0
    if overall_stdev == 0:
        return []

    best_idx = -1
    best_diff = 0.0

    for i in range(min_segment, n - min_segment + 1):
        left = values[:i]
        right = values[i:]
        diff = abs(statistics.mean(left) - statistics.mean(right))
        if diff > best_diff:
            best_diff = diff
            best_idx = i

    if best_idx < 0 or best_diff < overall_stdev * threshold_factor:
        return []

    # Recurse on both halves
    left_cps = _find_changepoints(values[:best_idx], min_segment, threshold_factor)
    right_cps = _find_changepoints(values[best_idx:], min_segment, threshold_factor)
    right_cps = [cp + best_idx for cp in right_cps]

    return left_cps + [best_idx] + right_cps


# ═══════════════════════���═══════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════

def compute_long_horizon(history: list[dict],
                         windows: tuple[int, ...] = (20, 50, 100)) -> dict:
    """Compute long-horizon drift metrics from trace history.

    history: list of {turn, trace, mode, model, drift} entries.
    windows: tuple of window sizes to analyze.

    Returns dict with:
        trace: per-window analysis of Hessian trace
        drift: per-window analysis of corpus drift
        regimes: regime detection results for trace
        summary: human-readable summary string
    """
    if not history:
        return {"trace": {}, "drift": {}, "regimes": [], "summary": "no data"}

    # Extract valid trace values
    traces = []
    for h in history:
        t = h.get("trace")
        if t is not None and isinstance(t, (int, float)):
            traces.append(float(t))

    # Extract valid drift values (0-1, skip -1 which means unavailable)
    drifts = []
    for h in history:
        d = h.get("drift", -1)
        if isinstance(d, (int, float)) and d >= 0:
            drifts.append(float(d))

    # Per-window analysis
    trace_windows = {}
    drift_windows = {}

    for w in windows:
        # Trace analysis for this window
        if len(traces) >= w:
            window_traces = traces[-w:]
            trace_windows[w] = {
                "trend": compute_trend(window_traces),
                "baseline": compute_baseline_shift(window_traces),
                "mean": round(statistics.mean(window_traces), 2),
                "stdev": round(statistics.stdev(window_traces), 2),
            }
        elif len(traces) >= 6:
            trace_windows[w] = {
                "trend": compute_trend(traces),
                "baseline": compute_baseline_shift(traces),
                "mean": round(statistics.mean(traces), 2),
                "stdev": round(statistics.stdev(traces), 2) if len(traces) >= 2 else 0.0,
                "note": f"using all {len(traces)} (< {w})",
            }

        # Drift analysis for this window
        if len(drifts) >= w:
            window_drifts = drifts[-w:]
            drift_windows[w] = {
                "trend": compute_trend(window_drifts),
                "baseline": compute_baseline_shift(window_drifts),
                "mean": round(statistics.mean(window_drifts), 3),
            }
        elif len(drifts) >= 6:
            drift_windows[w] = {
                "trend": compute_trend(drifts),
                "baseline": compute_baseline_shift(drifts),
                "mean": round(statistics.mean(drifts), 3),
                "note": f"using all {len(drifts)} (< {w})",
            }

    # Regime detection on full trace history
    regimes = detect_regimes(traces) if len(traces) >= 16 else []

    # Build summary
    summary = _build_summary(trace_windows, drift_windows, regimes, len(traces), len(drifts))

    return {
        "trace": trace_windows,
        "drift": drift_windows,
        "regimes": regimes,
        "summary": summary,
    }


def _build_summary(trace_windows: dict, drift_windows: dict,
                   regimes: list[dict], n_traces: int, n_drifts: int) -> str:
    """Build a concise human-readable summary of long-horizon state."""
    parts = []

    # Report the smallest window that has data
    for w in sorted(trace_windows.keys()):
        tw = trace_windows[w]
        trend = tw["trend"]
        if trend["direction"] != "insufficient":
            parts.append(f"trace@{w}: {trend['direction']} (rate={trend['rate']}/turn)")
            if tw["baseline"].get("significant"):
                parts.append(f"  baseline shifted {tw['baseline']['shift']:+.1f}")
            break

    for w in sorted(drift_windows.keys()):
        dw = drift_windows[w]
        trend = dw["trend"]
        if trend["direction"] != "insufficient":
            parts.append(f"drift@{w}: {trend['direction']} (rate={trend['rate']:.4f}/turn)")
            break

    if regimes and len(regimes) > 1:
        parts.append(f"regimes: {len(regimes)} detected")
        latest = regimes[-1]
        parts.append(f"  current: mean={latest['mean']} len={latest['length']}")

    if not parts:
        return f"accumulating ({n_traces} traces, {n_drifts} drifts)"

    return " | ".join(parts)


# ═══════════════════════════════════════════════════
# FORMAT FOR DISPLAY
# ════════���══════════════���═══════════════════════════

def format_long_horizon(result: dict) -> tuple[str | None, str | None]:
    """Format long-horizon result for medulla/interoception display.

    Returns (summary_str, detail_str) or (None, None) if nothing notable.
    """
    if not result or result.get("summary", "").startswith("no data"):
        return None, None

    summary = result.get("summary", "")
    if summary.startswith("accumulating"):
        return None, None

    # Only surface if there's actual drift or regime change
    has_drift = False
    for w_data in result.get("trace", {}).values():
        trend = w_data.get("trend", {})
        if trend.get("direction") in ("rising", "falling"):
            has_drift = True
            break
        if w_data.get("baseline", {}).get("significant"):
            has_drift = True
            break

    if not has_drift and len(result.get("regimes", [])) <= 1:
        return None, None

    # Build compact display
    parts = []
    for w in sorted(result.get("trace", {}).keys()):
        tw = result["trace"][w]
        trend = tw.get("trend", {})
        d = trend.get("direction", "flat")
        if d in ("rising", "falling"):
            arrow = "\u2191" if d == "rising" else "\u2193"
            parts.append(f"t{arrow}{w}")
            break

    for w in sorted(result.get("drift", {}).keys()):
        dw = result["drift"][w]
        trend = dw.get("trend", {})
        d = trend.get("direction", "flat")
        if d in ("rising", "falling"):
            arrow = "\u2191" if d == "rising" else "\u2193"
            parts.append(f"d{arrow}{w}")
            break

    n_regimes = len(result.get("regimes", []))
    if n_regimes > 1:
        parts.append(f"{n_regimes}reg")

    if not parts:
        return None, None

    summary_str = " ".join(parts)
    return summary_str, summary
