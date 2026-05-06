"""
Cross-channel correlation tracking — measurement expansion Pass 3.

Measures relationships between measurement channels:
  trace, drift, flattery, coupled_nudge

Answers: do channels move together? Do they diverge (signal
disagreement)? Are there lagged relationships (one predicts another)?

Operates on multi-channel history where each entry may have:
  {turn, trace, drift, flattery, coupled_nudge}

Pure functions — no side effects, no state.
"""

import math
import statistics
from typing import Any


# ═══════════════════════════════════════════════════
# PAIRWISE CORRELATION
# ═══════════════════════════════════════════════════

def pearson(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient between two series.

    Returns 0.0 if either series has zero variance or they differ in length.
    """
    n = min(len(x), len(y))
    if n < 3:
        return 0.0

    x = x[:n]
    y = y[:n]
    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)

    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denom_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    denom_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))

    if denom_x == 0 or denom_y == 0:
        return 0.0
    return num / (denom_x * denom_y)


def compute_pairwise(channels: dict[str, list[float]]) -> dict:
    """Compute pairwise Pearson correlations between all channel pairs.

    channels: dict mapping channel name to list of values (same length).

    Returns dict of {(ch_a, ch_b): correlation} for all pairs.
    """
    names = sorted(channels.keys())
    result = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            corr = pearson(channels[a], channels[b])
            result[(a, b)] = round(corr, 3)
    return result


# ═══════════════════════════════════════════════════
# LAGGED CROSS-CORRELATION
# ═══════════════════════════════════════════════════

def cross_correlation(x: list[float], y: list[float], max_lag: int = 5) -> dict:
    """Compute cross-correlation between x and y at various lags.

    Positive lag means x leads y (x[t] correlates with y[t+lag]).
    Negative lag means y leads x.

    Returns:
        lags: dict of lag -> correlation
        best_lag: the lag with highest absolute correlation
        best_corr: the correlation at best_lag
    """
    n = min(len(x), len(y))
    if n < max_lag + 3:
        return {"lags": {}, "best_lag": 0, "best_corr": 0.0}

    x = x[:n]
    y = y[:n]
    lags = {}

    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x_slice = x[:n - lag]
            y_slice = y[lag:]
        else:
            x_slice = x[-lag:]
            y_slice = y[:n + lag]
        corr = pearson(x_slice, y_slice)
        lags[lag] = round(corr, 3)

    best_lag = max(lags, key=lambda k: abs(lags[k]))
    best_corr = lags[best_lag]

    return {"lags": lags, "best_lag": best_lag, "best_corr": best_corr}


# ═══════════════════════════════════════════════════
# ROLLING CORRELATION — stability of relationships
# ═══════════════════════════════════════════════════

def rolling_correlation(x: list[float], y: list[float],
                        window: int = 15) -> dict:
    """Compute rolling Pearson correlation to detect relationship changes.

    Returns:
        values: list of rolling correlations
        mean: mean rolling correlation
        stdev: stdev of rolling correlation (high = unstable relationship)
        min: minimum correlation observed
        max: maximum correlation observed
        sign_changes: number of times correlation crosses zero
    """
    n = min(len(x), len(y))
    if n < window + 3:
        return {"values": [], "mean": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0, "sign_changes": 0}

    x = x[:n]
    y = y[:n]
    values = []

    for i in range(n - window + 1):
        x_win = x[i:i + window]
        y_win = y[i:i + window]
        values.append(pearson(x_win, y_win))

    if not values:
        return {"values": [], "mean": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0, "sign_changes": 0}

    sign_changes = 0
    for i in range(1, len(values)):
        if (values[i] >= 0) != (values[i - 1] >= 0):
            sign_changes += 1

    return {
        "values": [round(v, 3) for v in values],
        "mean": round(statistics.mean(values), 3),
        "stdev": round(statistics.stdev(values), 3) if len(values) >= 2 else 0.0,
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "sign_changes": sign_changes,
    }


# ═══════════════════════════════════════════════════
# DIVERGENCE DETECTION
# ═══════════════════════════════════════════════════

def detect_divergences(channels: dict[str, list[float]],
                       window: int = 10) -> list[dict]:
    """Detect points where channels that normally agree suddenly diverge.

    Compares recent correlation (last window) against historical correlation
    for each pair. Large drops indicate divergence events.

    Returns list of divergence events:
        pair: (channel_a, channel_b)
        historical_corr: correlation over full history
        recent_corr: correlation over last window
        drop: historical - recent
        significant: bool — True if drop > 0.5
    """
    names = sorted(channels.keys())
    divergences = []

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            va = channels[a]
            vb = channels[b]
            n = min(len(va), len(vb))
            if n < window * 2:
                continue

            historical = pearson(va[:n], vb[:n])
            recent = pearson(va[n - window:n], vb[n - window:n])
            drop = historical - recent

            if abs(drop) > 0.3:
                divergences.append({
                    "pair": (a, b),
                    "historical_corr": round(historical, 3),
                    "recent_corr": round(recent, 3),
                    "drop": round(drop, 3),
                    "significant": abs(drop) > 0.5,
                })

    return divergences


# ═══════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════

def compute_channel_correlation(history: list[dict]) -> dict:
    """Full cross-channel correlation analysis from trace history.

    history: list of entries, each with optional keys:
        trace, drift, flattery, coupled_nudge

    Returns:
        pairwise: current pairwise correlations
        lagged: lagged cross-correlations for key pairs
        divergences: detected divergence events
        stability: rolling correlation stats for key pairs
        summary: human-readable summary
    """
    if not history:
        return {"pairwise": {}, "lagged": {}, "divergences": [],
                "stability": {}, "summary": "no data"}

    # Extract channels — only include entries where value is valid
    channels = _extract_channels(history)

    # Need at least 2 channels with enough data
    valid_channels = {k: v for k, v in channels.items() if len(v) >= 10}
    if len(valid_channels) < 2:
        n = sum(len(v) for v in channels.values())
        return {"pairwise": {}, "lagged": {}, "divergences": [],
                "stability": {}, "summary": f"accumulating ({n} values across {len(channels)} channels)"}

    # Align channels to common length (trim to shortest)
    min_len = min(len(v) for v in valid_channels.values())
    aligned = {k: v[:min_len] for k, v in valid_channels.items()}

    # Pairwise correlations
    pairwise = compute_pairwise(aligned)

    # Lagged cross-correlations for key pairs
    lagged = {}
    key_pairs = [("drift", "trace"), ("flattery", "trace"), ("drift", "flattery")]
    for a, b in key_pairs:
        if a in aligned and b in aligned:
            lagged[(a, b)] = cross_correlation(aligned[a], aligned[b])

    # Divergence detection
    divergences = detect_divergences(aligned)

    # Rolling correlation stability for main pair
    stability = {}
    if "trace" in aligned and "drift" in aligned:
        stability[("trace", "drift")] = rolling_correlation(
            aligned["trace"], aligned["drift"])

    # Build summary
    summary = _build_summary(pairwise, lagged, divergences, stability)

    return {
        "pairwise": {f"{k[0]}-{k[1]}": v for k, v in pairwise.items()},
        "lagged": {f"{k[0]}-{k[1]}": v for k, v in lagged.items()},
        "divergences": divergences,
        "stability": {f"{k[0]}-{k[1]}": v for k, v in stability.items()},
        "summary": summary,
    }


def _extract_channels(history: list[dict]) -> dict[str, list[float]]:
    """Extract aligned channel values from history entries.

    Only includes entries where ALL channels have valid data,
    to keep series aligned for correlation.
    """
    traces = []
    drifts = []
    flatteries = []
    nudges = []

    for h in history:
        t = h.get("trace")
        d = h.get("drift", -1)
        f = h.get("flattery")
        n = h.get("coupled_nudge")

        has_trace = t is not None and isinstance(t, (int, float))
        has_drift = isinstance(d, (int, float)) and d >= 0
        has_flattery = f is not None and isinstance(f, (int, float))
        has_nudge = n is not None and isinstance(n, (int, float))

        # Only append if at least trace + one other channel
        if has_trace:
            traces.append(float(t))
            if has_drift:
                drifts.append(float(d))
            if has_flattery:
                flatteries.append(float(f))
            if has_nudge:
                nudges.append(float(n))

    channels = {}
    if traces:
        channels["trace"] = traces
    if drifts:
        channels["drift"] = drifts
    if flatteries:
        channels["flattery"] = flatteries
    if nudges:
        channels["nudge"] = nudges

    return channels


def _build_summary(pairwise: dict, lagged: dict, divergences: list,
                   stability: dict) -> str:
    """Build concise summary of cross-channel state."""
    parts = []

    # Report strong correlations
    strong = [(k, v) for k, v in pairwise.items() if abs(v) > 0.5]
    if strong:
        for (a, b), corr in sorted(strong, key=lambda x: -abs(x[1]))[:2]:
            sign = "+" if corr > 0 else "-"
            parts.append(f"{a}/{b}:{sign}{abs(corr):.2f}")

    # Report significant divergences
    sig_div = [d for d in divergences if d["significant"]]
    if sig_div:
        for d in sig_div[:2]:
            parts.append(f"DIVERGE:{d['pair'][0]}/{d['pair'][1]}")

    # Report lagged relationships
    for pair_key, lc in lagged.items():
        if abs(lc.get("best_corr", 0)) > 0.5 and lc.get("best_lag", 0) != 0:
            parts.append(f"lag:{pair_key[0]}→{pair_key[1]}@{lc['best_lag']}")

    if not parts:
        return "channels uncorrelated"

    return " | ".join(parts)


# ═══════════════════════════════════════════════════
# FORMAT FOR DISPLAY
# ═══════════════════════════════════════════════════

def format_channel_correlation(result: dict) -> tuple[str | None, str | None]:
    """Format correlation result for medulla/interoception display.

    Returns (summary_str, detail_str) or (None, None) if nothing notable.
    """
    if not result or result.get("summary", "").startswith("no data"):
        return None, None

    summary = result.get("summary", "")
    if summary.startswith("accumulating"):
        return None, None

    # Only surface if there's divergence or strong unexpected correlations
    has_divergence = any(d.get("significant") for d in result.get("divergences", []))

    # Strong correlation is notable
    pairwise = result.get("pairwise", {})
    has_strong = any(abs(v) > 0.7 for v in pairwise.values())

    if not has_divergence and not has_strong:
        return None, None

    parts = []
    if has_divergence:
        parts.append("DIV")
    if has_strong:
        # Find the strongest pair
        strongest = max(pairwise.items(), key=lambda x: abs(x[1]), default=(None, 0))
        if strongest[0]:
            parts.append(f"r={strongest[1]:+.2f}")

    summary_str = " ".join(parts) if parts else None
    return summary_str, summary
