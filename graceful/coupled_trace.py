"""
Coupled Manifold — coupled trace (Move 2).
Measurement crosses the apparatus boundary: model Hessian trace
is coupled with human state signals to produce a joint reading.

Barad's agential cut: the observer is not separable from the
measured system. The coupled trace registers both sides.

Pure function — no side effects, no state.
"""

import statistics


def couple(model_trace, human_state):
    """Couple model trace with human behavioral signals.

    model_trace: float or None — raw Hessian trace from model.
    human_state: dict — keys from frontend instrumentation:
        typing_duration_ms  — time from first keystroke to send
        keystroke_count     — total keystrokes
        edit_ratio          — backspaces / total keystrokes (0-1)
        median_pause_ms     — median inter-keystroke pause
        max_pause_ms        — longest inter-keystroke pause
        read_time_ms        — time spent reading before typing
        scroll_ups          — discrete scroll-up gestures on response

    Returns: float or None — coupled trace value.
    If model_trace is None or human_state is empty, returns model_trace unchanged.
    """
    if model_trace is None:
        return None
    if not human_state:
        return model_trace

    nudge = _compute_nudge(human_state)
    return model_trace + nudge


def _compute_nudge(hs):
    """Compute a signed nudge from human state signals.

    Positive = human appears engaged/confident → healthier trace.
    Negative = human appears struggling/uncertain → trace pushed toward pathology.
    Nudge is bounded to [-20, +20] absolute points.
    """
    signals = []

    # ── Typing engagement ────────────────────────────────────────
    ks = hs.get("keystroke_count", 0)
    if ks > 0:
        # More keystrokes → more engaged. Baseline: 30 keystrokes = neutral.
        # Scale: <10 = slight negative, >60 = slight positive.
        engagement = _clamp((ks - 30) / 50, -1.0, 1.0)
        signals.append(("engagement", engagement, 3.0))

    # ── Edit ratio — high backspace ratio signals struggle ──────
    edit_ratio = hs.get("edit_ratio", 0)
    if edit_ratio > 0:
        # 0.05 = normal editing, 0.3+ = heavy rewriting
        struggle = _clamp((edit_ratio - 0.1) / 0.3, -1.0, 1.0)
        signals.append(("edit_struggle", -struggle, 4.0))

    # ── Pause distribution — long pauses signal thinking/confusion ──
    median_pause = hs.get("median_pause_ms", 0)
    max_pause = hs.get("max_pause_ms", 0)
    if median_pause > 0:
        # Median pause: <200ms = fast/fluent, >1000ms = deliberate/stuck
        pause_signal = _clamp((median_pause - 300) / 700, -1.0, 1.0)
        signals.append(("pause_median", -pause_signal, 2.0))
    if max_pause > 0:
        # Very long max pause (>5s) = went away to think or got confused
        long_pause = _clamp((max_pause - 2000) / 8000, 0, 1.0)
        signals.append(("pause_max", -long_pause, 2.0))

    # ── Read time — long reading before responding signals density ──
    read_time = hs.get("read_time_ms", 0)
    if read_time > 0:
        # <3s = quick scan (engaged or short response), >30s = deep read or confusion
        read_signal = _clamp((read_time - 5000) / 25000, -1.0, 1.0)
        signals.append(("read_time", -read_signal, 3.0))

    # ── Scroll-ups — re-reading response signals need for review ──
    scroll_ups = hs.get("scroll_ups", 0)
    if scroll_ups > 0:
        # 1-2 = normal review, 5+ = significant re-reading
        scroll_signal = _clamp((scroll_ups - 2) / 6, 0, 1.0)
        signals.append(("scroll_review", -scroll_signal, 2.0))

    # ── Typing duration — very fast = confident, very slow = deliberate ──
    typing_ms = hs.get("typing_duration_ms", 0)
    if typing_ms > 0 and ks > 0:
        # Characters per second as proxy
        cps = (ks / typing_ms) * 1000
        # 3-5 cps = normal, <1 = slow/deliberate, >7 = fast/confident
        speed_signal = _clamp((cps - 3) / 4, -1.0, 1.0)
        signals.append(("typing_speed", speed_signal, 2.0))

    if not signals:
        return 0.0

    # Weighted average of all signals
    total_weight = sum(w for _, _, w in signals)
    weighted_sum = sum(val * w for _, val, w in signals)
    raw_nudge = weighted_sum / total_weight

    # Scale to [-20, +20] range
    return _clamp(raw_nudge * 20, -20, 20)


def summarize(model_trace, human_state, coupled_trace):
    """Return a short summary string for medulla display.

    Returns (nudge_str, detail_str) or (None, None) if no coupling.
    """
    if model_trace is None or coupled_trace is None or not human_state:
        return None, None
    nudge = coupled_trace - model_trace
    if abs(nudge) < 0.1:
        return None, None

    parts = []
    ks = human_state.get("keystroke_count", 0)
    if ks > 0:
        parts.append(f"ks:{ks}")
    er = human_state.get("edit_ratio", 0)
    if er > 0.05:
        parts.append(f"edit:{er:.0%}")
    read_ms = human_state.get("read_time_ms", 0)
    if read_ms > 1000:
        parts.append(f"read:{read_ms/1000:.1f}s")
    scroll = human_state.get("scroll_ups", 0)
    if scroll > 0:
        parts.append(f"scroll:{scroll}")

    sign = "+" if nudge > 0 else ""
    nudge_str = f"{sign}{nudge:.1f}"
    detail_str = " ".join(parts) if parts else "minimal"
    return nudge_str, detail_str


def _clamp(val, lo, hi):
    return max(lo, min(hi, val))
