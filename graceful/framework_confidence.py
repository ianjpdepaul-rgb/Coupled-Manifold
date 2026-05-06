"""
Coupled Manifold — framework confidence (Move 3).
Visible failure: the framework reports its own measurement confidence
as a 0-1 scalar. When channels disagree, the measurement is marked
provisional and the disagreement is logged.

The framework does not pretend its measurements are certain.
This is not a bug — it is the apparatus acknowledging its limits.

Pure functions — no side effects, no state.
"""

import time


def compute_confidence(trace, drift, flattery_score, trace_history,
                       coupled_nudge=None):
    """Compute framework confidence from channel agreement.

    trace:           float or None — Hessian trace (coupled or raw)
    drift:           float — corpus drift (0-1, -1 if unavailable)
    flattery_score:  float — sycophancy score (0-1)
    trace_history:   list — recent trace values (for stability)
    coupled_nudge:   str or None — nudge string from coupled trace

    Returns dict:
        confidence:  float 0-1
        provisional: bool — True if measurement should be marked uncertain
        channels:    dict of per-channel assessments
        disagreements: list of string descriptions of channel conflicts
    """
    channels = {}
    disagreements = []

    # ── Channel 1: Trace health ─────────────────────────────────
    if trace is not None:
        # Healthy trace: positive, moderate magnitude
        if trace > 50:
            channels["trace"] = "healthy"
        elif trace > 0:
            channels["trace"] = "neutral"
        elif trace > -50:
            channels["trace"] = "stressed"
        else:
            channels["trace"] = "pathological"
    else:
        channels["trace"] = "unavailable"

    # ── Channel 2: Drift ────────────────────────────────────────
    if drift >= 0:
        if drift < 0.3:
            channels["drift"] = "on-topic"
        elif drift < 0.5:
            channels["drift"] = "wandering"
        else:
            channels["drift"] = "off-topic"
    else:
        channels["drift"] = "unavailable"

    # ── Channel 3: Flattery ─────────────────────────────────────
    if flattery_score < 0.2:
        channels["flattery"] = "authentic"
    elif flattery_score < 0.5:
        channels["flattery"] = "mild"
    elif flattery_score < 0.7:
        channels["flattery"] = "elevated"
    else:
        channels["flattery"] = "sycophantic"

    # ── Channel 4: Sample stability ─────────────────────────────
    stability = _compute_stability(trace_history)
    channels["stability"] = stability["label"]

    # ── Disagreement detection ──────────────────────────────────
    # Healthy trace + high flattery = disagreement
    if channels["trace"] == "healthy" and channels["flattery"] in ("elevated", "sycophantic"):
        disagreements.append("trace_healthy_but_flattering")

    # Pathological trace + low drift = disagreement (bad geometry but on-topic)
    if channels["trace"] == "pathological" and channels["drift"] == "on-topic":
        disagreements.append("trace_pathological_but_on_topic")

    # Healthy trace + high drift = disagreement
    if channels["trace"] == "healthy" and channels["drift"] == "off-topic":
        disagreements.append("trace_healthy_but_drifting")

    # Unstable trace + no other warning signals
    if channels["stability"] == "volatile" and channels["flattery"] == "authentic":
        disagreements.append("volatile_trace_no_flattery")

    # ── Confidence score ────────────────────────────────────────
    score = 1.0

    # Unavailable channels reduce confidence
    unavailable = sum(1 for v in channels.values() if v == "unavailable")
    score -= unavailable * 0.15

    # Disagreements reduce confidence
    score -= len(disagreements) * 0.2

    # Instability reduces confidence
    if channels["stability"] == "volatile":
        score -= 0.15
    elif channels["stability"] == "shifting":
        score -= 0.05

    # High flattery reduces confidence in the whole measurement
    if channels["flattery"] in ("elevated", "sycophantic"):
        score -= 0.1

    # Coupled nudge magnitude as signal — large nudge = human state
    # strongly disagrees with model state
    if coupled_nudge is not None:
        try:
            nudge_val = float(coupled_nudge.replace("+", ""))
            if abs(nudge_val) > 10:
                score -= 0.1
                if abs(nudge_val) > 15:
                    disagreements.append("strong_human_model_divergence")
        except (ValueError, AttributeError):
            pass

    score = max(0.0, min(1.0, score))
    provisional = score < 0.6 or len(disagreements) >= 2

    return {
        "confidence": round(score, 2),
        "provisional": provisional,
        "channels": channels,
        "disagreements": disagreements,
    }


def format_confidence(result):
    """Format confidence for medulla display.

    Returns (confidence_str, detail_str) or (None, None) if fully confident.
    """
    conf = result["confidence"]
    if conf >= 0.9 and not result["provisional"]:
        return None, None

    # Confidence bar: 5 blocks
    filled = round(conf * 5)
    bar = "\u2588" * filled + "\u2591" * (5 - filled)

    parts = []
    if result["provisional"]:
        parts.append("PROVISIONAL")
    if result["disagreements"]:
        parts.append(" ".join(d.replace("_", "-") for d in result["disagreements"][:2]))

    conf_str = f"{conf:.0%} {bar}"
    detail_str = " | ".join(parts) if parts else ""
    return conf_str, detail_str


def build_failure_record(turn, result, trace, drift, flattery_score):
    """Build a failure log entry when confidence is low or provisional."""
    if not result["provisional"] and result["confidence"] >= 0.7:
        return None
    return {
        "timestamp": time.time(),
        "turn": turn,
        "confidence": result["confidence"],
        "provisional": result["provisional"],
        "channels": result["channels"],
        "disagreements": result["disagreements"],
        "trace": trace if trace is not None else None,
        "drift": drift,
        "flattery_score": flattery_score,
    }


def _compute_stability(trace_history):
    """Assess trace stability from recent history."""
    valid = [t for t in (trace_history or []) if t is not None and isinstance(t, (int, float))]
    if len(valid) < 3:
        return {"label": "building", "variance": 0}

    recent = valid[-5:]
    if len(recent) < 2:
        return {"label": "building", "variance": 0}

    mean = sum(recent) / len(recent)
    variance = sum((x - mean) ** 2 for x in recent) / len(recent)

    if variance > 5000:
        return {"label": "volatile", "variance": round(variance, 1)}
    elif variance > 1000:
        return {"label": "shifting", "variance": round(variance, 1)}
    else:
        return {"label": "stable", "variance": round(variance, 1)}
