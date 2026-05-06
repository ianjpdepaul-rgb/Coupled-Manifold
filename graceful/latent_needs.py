"""Latent need detection — Group G (Pass 8).

Infers what the user might need but hasn't explicitly asked for,
based on behavioral patterns, topic context, and interaction signals.
"""

import re
import time


# ═══════════════════════════════════════════════════
# NEED CATEGORIES
# ═══════════════════════════════════════════════════

NEED_TYPES = {
    "clarification": "User may need concepts explained differently",
    "break": "User may benefit from a pause",
    "encouragement": "User may need reassurance or validation",
    "structure": "User may need help organizing their thoughts",
    "depth": "User seems to want deeper exploration of a topic",
    "simplification": "User may be overwhelmed — simplify",
    "action_items": "User may need concrete next steps",
    "context_switch": "User seems ready to move to a new topic",
}


def detect_latent_needs(
    signals: dict,
    mood: dict,
    session_trajectory: str,
    turn_count: int,
    recent_modes: list[str],
) -> list[dict]:
    """Detect latent needs from combined signal state.

    Returns list of detected needs with confidence and reasoning.
    """
    needs = []

    # ── Clarification need ──
    # Multiple questions + confused mood + exploring mode
    qc = signals.get("question_count", 0)
    mood_label = mood.get("mood", "neutral")
    if qc >= 2 and mood_label in ("confused", "curious"):
        needs.append({
            "type": "clarification",
            "confidence": min(0.9, 0.4 + qc * 0.15),
            "reason": f"{qc} questions + {mood_label} mood",
        })

    # ── Break need ──
    # Long session + contracting trajectory + declining engagement
    if turn_count > 15 and session_trajectory == "contracting":
        needs.append({
            "type": "break",
            "confidence": 0.5 + (turn_count - 15) * 0.02,
            "reason": f"turn {turn_count}, messages getting shorter",
        })

    # Anxiety/frustration sustained over 3+ turns
    if mood_label in ("negative", "anxious"):
        intensity = mood.get("intensity", 0)
        if intensity > 0.6:
            needs.append({
                "type": "break",
                "confidence": 0.6,
                "reason": f"sustained {mood_label} (intensity {intensity:.1f})",
            })

    # ── Encouragement need ──
    # Low confidence in user signals + anxious mood
    urgency = signals.get("urgency", 0)
    if mood_label == "anxious" and urgency > 0.3:
        needs.append({
            "type": "encouragement",
            "confidence": 0.5,
            "reason": "anxious mood with elevated urgency",
        })

    # ── Structure need ──
    # Very long messages (teaching mode) — user may be processing aloud
    wc = signals.get("word_count", 0)
    if wc > 150 and signals.get("question_count", 0) < 2:
        needs.append({
            "type": "structure",
            "confidence": 0.5,
            "reason": f"long message ({wc} words) — may be thinking aloud",
        })

    # ── Depth need ──
    # Repeated questions on same topic + exploring mode
    mode = signals.get("mode", "working")
    if mode == "exploring" and qc >= 1:
        needs.append({
            "type": "depth",
            "confidence": 0.4 + qc * 0.1,
            "reason": "exploring mode with follow-up questions",
        })

    # ── Simplification need ──
    # Short, terse responses after previously long ones = overwhelmed
    if session_trajectory == "contracting" and wc < 20 and turn_count > 5:
        needs.append({
            "type": "simplification",
            "confidence": 0.45,
            "reason": "messages getting terse — may be overwhelmed",
        })

    # ── Action items need ──
    # Working mode sustained, long session
    if turn_count > 10 and recent_modes.count("working") >= 3:
        needs.append({
            "type": "action_items",
            "confidence": 0.35,
            "reason": "sustained working mode — may need summary/next steps",
        })

    # ── Context switch need ──
    # Session expanding with many topic shifts
    if session_trajectory == "expanding" and len(set(recent_modes)) >= 3:
        needs.append({
            "type": "context_switch",
            "confidence": 0.4,
            "reason": "expanding messages across multiple modes",
        })

    # Sort by confidence descending, limit to top 3
    needs.sort(key=lambda x: -x["confidence"])
    return needs[:3]


def needs_context_string(needs: list[dict]) -> str:
    """Compact context string for interoceptive block."""
    if not needs:
        return ""
    high_conf = [n for n in needs if n["confidence"] >= 0.5]
    if not high_conf:
        return ""
    labels = [f"{n['type']}({n['confidence']:.1f})" for n in high_conf[:2]]
    return "[LATENT] " + " | ".join(labels)
