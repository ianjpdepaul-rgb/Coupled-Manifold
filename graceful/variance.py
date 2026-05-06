"""
Variance structure characterization — measurement expansion Pass 2.

Goes beyond scalar stdev: characterizes the shape of variance in
trace/drift time series. Answers: is the noise bursty or smooth?
Is variance changing over time? Are there periodic patterns?

Operates on the same trace_history_live entries as longhorizon.py.

Pure functions — no side effects, no state.
"""

import math
import statistics
from typing import Any


# ═══════════════════════════════════════════════════
# HETEROSCEDASTICITY — does variance change over time?
# ═══════════════════════════════════════════════════

def compute_heteroscedasticity(values: list[float], n_segments: int = 4) -> dict:
    """Test whether variance is stable or changing across segments.

    Splits the series into n_segments equal chunks and compares variance.
    High ratio (max_var / min_var) indicates heteroscedasticity.

    Returns:
        ratio: max_variance / min_variance across segments
        segments: list of per-segment {mean, variance, n}
        heteroscedastic: bool — True if ratio > 4.0
        trend: "increasing", "decreasing", or "stable" — direction of variance change
    """
    if len(values) < n_segments * 3:
        return {"ratio": 1.0, "segments": [], "heteroscedastic": False, "trend": "stable"}

    seg_len = len(values) // n_segments
    segments = []
    for i in range(n_segments):
        start = i * seg_len
        end = start + seg_len if i < n_segments - 1 else len(values)
        seg = values[start:end]
        if len(seg) >= 2:
            segments.append({
                "mean": round(statistics.mean(seg), 3),
                "variance": round(statistics.variance(seg), 3),
                "n": len(seg),
            })
        else:
            segments.append({"mean": seg[0] if seg else 0, "variance": 0.0, "n": len(seg)})

    variances = [s["variance"] for s in segments if s["variance"] > 0]
    if not variances:
        return {"ratio": 1.0, "segments": segments, "heteroscedastic": False, "trend": "stable"}

    min_var = min(variances)
    max_var = max(variances)
    ratio = max_var / min_var if min_var > 0 else float("inf")

    # Determine direction: is variance growing or shrinking?
    if len(variances) >= 3:
        first_half = variances[:len(variances) // 2]
        second_half = variances[len(variances) // 2:]
        mean_first = statistics.mean(first_half)
        mean_second = statistics.mean(second_half)
        if mean_second > mean_first * 1.5:
            trend = "increasing"
        elif mean_first > mean_second * 1.5:
            trend = "decreasing"
        else:
            trend = "stable"
    else:
        trend = "stable"

    return {
        "ratio": round(ratio, 2),
        "segments": segments,
        "heteroscedastic": ratio > 4.0,
        "trend": trend,
    }


# ═══════════════════════════════════════════════════
# BURSTINESS — clustered vs uniform variance
# ═══════════════════════════════════════════════════

def compute_burstiness(values: list[float], window: int = 5) -> dict:
    """Measure whether high-variance events cluster or spread uniformly.

    Uses the Fano factor (variance-to-mean ratio of local variances)
    and the coefficient of variation of rolling variances.

    Returns:
        fano: Fano factor of local variances (>1 = bursty, <1 = regular)
        cv: coefficient of variation of local variances
        bursty: bool — True if Fano > 2.0
        max_burst_idx: index of the maximum local variance window
    """
    if len(values) < window * 2:
        return {"fano": 1.0, "cv": 0.0, "bursty": False, "max_burst_idx": 0}

    # Compute rolling variances
    local_vars = []
    for i in range(len(values) - window + 1):
        chunk = values[i:i + window]
        if len(chunk) >= 2:
            local_vars.append(statistics.variance(chunk))

    if not local_vars or all(v == 0 for v in local_vars):
        return {"fano": 1.0, "cv": 0.0, "bursty": False, "max_burst_idx": 0}

    mean_var = statistics.mean(local_vars)
    if mean_var == 0:
        return {"fano": 1.0, "cv": 0.0, "bursty": False, "max_burst_idx": 0}

    var_of_vars = statistics.variance(local_vars) if len(local_vars) >= 2 else 0
    fano = var_of_vars / mean_var

    stdev_vars = math.sqrt(var_of_vars) if var_of_vars > 0 else 0
    cv = stdev_vars / mean_var

    max_burst_idx = local_vars.index(max(local_vars))

    return {
        "fano": round(fano, 3),
        "cv": round(cv, 3),
        "bursty": cv > 1.5,
        "max_burst_idx": max_burst_idx,
    }


# ═══════════════════════════════════════════════════
# AUTOCORRELATION — is noise correlated or white?
# ═══════════════════════════════════════════════════

def compute_autocorrelation(values: list[float], max_lag: int = 5) -> dict:
    """Compute lag-1 through lag-max_lag autocorrelation.

    High lag-1 autocorrelation means changes are smooth/persistent.
    Low autocorrelation means changes are random/unpredictable.

    Returns:
        lag_1: lag-1 autocorrelation (-1 to 1)
        lags: dict of lag -> correlation
        persistent: bool — lag_1 > 0.5
        antipersistent: bool — lag_1 < -0.3
    """
    n = len(values)
    if n < max_lag + 3:
        return {"lag_1": 0.0, "lags": {}, "persistent": False, "antipersistent": False}

    mean = statistics.mean(values)
    # Denominator: variance * n
    denom = sum((v - mean) ** 2 for v in values)
    if denom == 0:
        return {"lag_1": 0.0, "lags": {}, "persistent": False, "antipersistent": False}

    lags = {}
    for lag in range(1, min(max_lag + 1, n)):
        numerator = sum((values[i] - mean) * (values[i + lag] - mean) for i in range(n - lag))
        lags[lag] = round(numerator / denom, 3)

    lag_1 = lags.get(1, 0.0)

    return {
        "lag_1": lag_1,
        "lags": lags,
        "persistent": lag_1 > 0.5,
        "antipersistent": lag_1 < -0.3,
    }


# ═══════════════════════════════════════════════════
# EXTREMES — tail behavior
# ═══════════════════════════════════════════════════

def compute_extremes(values: list[float]) -> dict:
    """Characterize extreme value behavior.

    Returns:
        kurtosis: excess kurtosis (>0 = heavy tails, <0 = light tails)
        extreme_count: number of values beyond 2 stdev from mean
        extreme_fraction: extreme_count / total
        max_deviation: largest absolute deviation from mean (in stdevs)
    """
    n = len(values)
    if n < 4:
        return {"kurtosis": 0.0, "extreme_count": 0, "extreme_fraction": 0.0, "max_deviation": 0.0}

    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    if stdev == 0:
        return {"kurtosis": 0.0, "extreme_count": 0, "extreme_fraction": 0.0, "max_deviation": 0.0}

    # Excess kurtosis
    m4 = sum((v - mean) ** 4 for v in values) / n
    m2 = sum((v - mean) ** 2 for v in values) / n
    kurtosis = (m4 / (m2 ** 2)) - 3.0 if m2 > 0 else 0.0

    # Extremes beyond 2 stdev
    extreme_count = sum(1 for v in values if abs(v - mean) > 2 * stdev)
    extreme_fraction = extreme_count / n

    # Maximum deviation
    max_dev = max(abs(v - mean) for v in values) / stdev

    return {
        "kurtosis": round(kurtosis, 3),
        "extreme_count": extreme_count,
        "extreme_fraction": round(extreme_fraction, 3),
        "max_deviation": round(max_dev, 2),
    }


# ═══════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════

def characterize_variance(history: list[dict]) -> dict:
    """Full variance characterization from trace history.

    history: list of {turn, trace, mode, model, drift} entries.

    Returns dict with:
        trace: variance analysis of Hessian trace values
        drift: variance analysis of corpus drift values
        summary: human-readable summary
    """
    if not history:
        return {"trace": {}, "drift": {}, "summary": "no data"}

    # Extract valid values
    traces = [float(h["trace"]) for h in history
              if h.get("trace") is not None and isinstance(h["trace"], (int, float))]
    drifts = [float(h["drift"]) for h in history
              if isinstance(h.get("drift"), (int, float)) and h["drift"] >= 0]

    trace_analysis = _analyze_channel(traces) if len(traces) >= 8 else {}
    drift_analysis = _analyze_channel(drifts) if len(drifts) >= 8 else {}

    summary = _build_summary(trace_analysis, drift_analysis, len(traces), len(drifts))

    return {
        "trace": trace_analysis,
        "drift": drift_analysis,
        "summary": summary,
    }


def _analyze_channel(values: list[float]) -> dict:
    """Run full variance analysis on a single channel."""
    return {
        "heteroscedasticity": compute_heteroscedasticity(values),
        "burstiness": compute_burstiness(values),
        "autocorrelation": compute_autocorrelation(values),
        "extremes": compute_extremes(values),
        "basic": {
            "mean": round(statistics.mean(values), 3),
            "stdev": round(statistics.stdev(values), 3) if len(values) >= 2 else 0.0,
            "n": len(values),
        },
    }


def _build_summary(trace_analysis: dict, drift_analysis: dict,
                   n_traces: int, n_drifts: int) -> str:
    """Build concise summary of variance structure."""
    parts = []

    if trace_analysis:
        t_parts = []
        het = trace_analysis.get("heteroscedasticity", {})
        if het.get("heteroscedastic"):
            t_parts.append(f"het({het['trend']})")
        burst = trace_analysis.get("burstiness", {})
        if burst.get("bursty"):
            t_parts.append("bursty")
        ac = trace_analysis.get("autocorrelation", {})
        if ac.get("persistent"):
            t_parts.append("persistent")
        elif ac.get("antipersistent"):
            t_parts.append("antipersistent")
        ext = trace_analysis.get("extremes", {})
        if ext.get("kurtosis", 0) > 3:
            t_parts.append("heavy-tailed")

        if t_parts:
            parts.append("trace: " + ", ".join(t_parts))
        else:
            parts.append("trace: well-behaved")

    if drift_analysis:
        d_parts = []
        het = drift_analysis.get("heteroscedasticity", {})
        if het.get("heteroscedastic"):
            d_parts.append(f"het({het['trend']})")
        burst = drift_analysis.get("burstiness", {})
        if burst.get("bursty"):
            d_parts.append("bursty")

        if d_parts:
            parts.append("drift: " + ", ".join(d_parts))

    if not parts:
        return f"accumulating ({n_traces} traces, {n_drifts} drifts)"

    return " | ".join(parts)


# ═══════════════════════════════════════════════════
# FORMAT FOR DISPLAY
# ═══════════════════════════════════════════════════

def format_variance(result: dict) -> tuple[str | None, str | None]:
    """Format variance result for medulla/interoception display.

    Returns (summary_str, detail_str) or (None, None) if nothing notable.
    """
    if not result or result.get("summary", "").startswith("no data"):
        return None, None

    summary = result.get("summary", "")
    if summary.startswith("accumulating"):
        return None, None

    # Only surface if there's something anomalous
    trace = result.get("trace", {})
    has_anomaly = False

    if trace:
        if trace.get("heteroscedasticity", {}).get("heteroscedastic"):
            has_anomaly = True
        if trace.get("burstiness", {}).get("bursty"):
            has_anomaly = True
        if trace.get("autocorrelation", {}).get("persistent"):
            has_anomaly = True
        if trace.get("extremes", {}).get("kurtosis", 0) > 3:
            has_anomaly = True

    if not has_anomaly:
        return None, None

    # Build compact display
    parts = []
    if trace.get("heteroscedasticity", {}).get("heteroscedastic"):
        parts.append("var\u2260")  # ≠ symbol
    if trace.get("burstiness", {}).get("bursty"):
        parts.append("burst")
    if trace.get("autocorrelation", {}).get("persistent"):
        parts.append("corr")

    summary_str = " ".join(parts) if parts else None
    return summary_str, summary
