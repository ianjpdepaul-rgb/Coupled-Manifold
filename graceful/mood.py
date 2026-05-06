"""Per-turn mood classification — Group C.

Classifies user emotional state per message into discrete categories
with a continuous intensity score. Tracks mood trajectory across
the session for drift detection.
"""

import json
import os
import time


# ═══════════════════════════════════════════════════
# MOOD CATEGORIES
# ═══════════════════════════════════════════════════

MOODS = [
    "neutral",
    "positive",      # happy, satisfied, grateful
    "negative",      # frustrated, angry, disappointed
    "anxious",       # worried, uncertain, stressed
    "curious",       # questioning, exploring, interested
    "focused",       # task-oriented, efficient, terse
    "playful",       # joking, casual, lighthearted
    "confused",      # lost, unclear, needing help
]


def classify_mood(signals: dict, text: str = "", embedder=None) -> dict:
    """Classify mood from human_profile signals and raw text.

    When embedder is provided, uses sentence embeddings for semantic
    mood scoring and profanity context disambiguation.

    Returns:
        {
            "mood": str,           # primary mood label
            "intensity": float,    # 0-1
            "secondary": str|None, # secondary mood if ambiguous
            "indicators": list,    # what drove the classification
        }
    """
    indicators = []
    mood_scores: dict[str, float] = {m: 0.0 for m in MOODS}

    # ── Embedding-based scoring (when available) ──
    _embedding_active = False
    if embedder and text:
        try:
            from graceful.embedder import score_mood_embedding, _has_embed_method
            if _has_embed_method(embedder):
                emb_scores = score_mood_embedding(text, embedder)
                for mood_cat, sim in emb_scores.items():
                    if mood_cat in mood_scores:
                        mood_scores[mood_cat] += sim * 0.5
                _embedding_active = True
                indicators.append("embedding")
        except Exception:
            pass

    # ── Signal-based scoring ──
    valence = signals.get("emotional_valence", 0.0)
    urgency = signals.get("urgency", 0.0)
    mode = signals.get("mode", "working")
    profanity = signals.get("profanity_flag", False)
    all_caps = signals.get("all_caps", False)
    emoji_count = signals.get("emoji_count", 0)
    question_count = signals.get("question_count", 0)
    exclamation_count = signals.get("exclamation_count", 0)
    word_count = signals.get("word_count", 0)
    slang_ratio = signals.get("slang_ratio", 0.0)

    # Positive signals
    if valence > 0.2:
        mood_scores["positive"] += valence * 0.6
        indicators.append("positive_valence")
    if emoji_count > 0:
        mood_scores["positive"] += 0.15
        mood_scores["playful"] += 0.1
        indicators.append("emoji")

    # Negative signals
    if valence < -0.2:
        mood_scores["negative"] += abs(valence) * 0.6
        indicators.append("negative_valence")
    if profanity:
        # Context-aware profanity scoring
        _prof_ctx = "negative_emotion"
        if embedder and text:
            try:
                from graceful.embedder import classify_profanity_context, _has_embed_method
                if _has_embed_method(embedder):
                    _prof_ctx = classify_profanity_context(text, embedder)
            except Exception:
                pass
        if _prof_ctx == "intensifier":
            mood_scores["positive"] += 0.15
            indicators.append("profanity_intensifier")
        elif _prof_ctx == "negative_emotion":
            mood_scores["negative"] += 0.3
            indicators.append("profanity_negative")
        else:
            mood_scores["negative"] += 0.15
            indicators.append("profanity_ambiguous")
    if all_caps:
        mood_scores["negative"] += 0.2
        mood_scores["anxious"] += 0.1
        indicators.append("all_caps")

    # Curiosity signals
    if question_count >= 2:
        mood_scores["curious"] += 0.4
        indicators.append("multiple_questions")
    elif question_count == 1:
        mood_scores["curious"] += 0.2

    # Focus signals
    if mode == "working" and word_count < 30 and question_count == 0:
        mood_scores["focused"] += 0.3
        indicators.append("terse_working")
    if word_count < 10 and exclamation_count == 0:
        mood_scores["focused"] += 0.2

    # Anxiety signals
    if urgency > 0.5:
        mood_scores["anxious"] += urgency * 0.4
        indicators.append("high_urgency")

    # Playful signals
    if slang_ratio > 0.2:
        mood_scores["playful"] += 0.3
        indicators.append("high_slang")
    if mode == "socializing":
        mood_scores["playful"] += 0.2
        indicators.append("socializing_mode")

    # Confusion signals
    if mode == "exploring" and word_count > 50:
        mood_scores["confused"] += 0.15
    if question_count >= 3:
        mood_scores["confused"] += 0.2
        indicators.append("many_questions")

    # Venting → negative
    if mode == "venting":
        mood_scores["negative"] += 0.4
        indicators.append("venting_mode")

    # ── Text keyword boost ──
    if text:
        text_lower = text.lower()
        _CONFUSED_WORDS = {"confused", "don't understand", "what do you mean",
                           "lost", "huh", "unclear", "explain", "wait what"}
        _ANXIOUS_WORDS = {"worried", "urgent", "asap", "deadline", "hurry",
                          "running out", "panic", "stressed", "nervous"}
        _PLAYFUL_WORDS = {"haha", "lol", "lmao", "rofl", "jk", "just kidding",
                          "😂", "funny"}

        for w in _CONFUSED_WORDS:
            if w in text_lower:
                mood_scores["confused"] += 0.2
                indicators.append(f"keyword:{w}")
                break
        for w in _ANXIOUS_WORDS:
            if w in text_lower:
                mood_scores["anxious"] += 0.2
                indicators.append(f"keyword:{w}")
                break
        for w in _PLAYFUL_WORDS:
            if w in text_lower:
                mood_scores["playful"] += 0.15
                break

    # Neutral baseline — always has some weight
    mood_scores["neutral"] = 0.15

    # ── Resolve ──
    sorted_moods = sorted(mood_scores.items(), key=lambda x: -x[1])
    primary = sorted_moods[0]
    secondary = sorted_moods[1] if sorted_moods[1][1] > 0.1 else None

    # Intensity: how strong is the primary vs baseline
    intensity = min(1.0, primary[1] / max(sum(mood_scores.values()), 0.01))

    return {
        "mood": primary[0],
        "intensity": round(intensity, 2),
        "secondary": secondary[0] if secondary else None,
        "indicators": indicators[:5],
    }


# ═══════════════════════════════════════════════════
# MOOD TRACKER — session trajectory
# ═══════════════════════════════════════════════════

class MoodTracker:
    """Tracks mood across a session and detects shifts."""

    def __init__(self, data_dir: str):
        self._data_dir = data_dir
        self._log_path = os.path.join(data_dir, "logs", "mood_trace.jsonl")
        self._trajectory: list[dict] = []

    def record(self, mood_result: dict):
        """Record a mood classification."""
        entry = {
            "ts": time.time(),
            **mood_result,
        }
        self._trajectory.append(entry)

        # Log to file
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except OSError:
            pass

    def detect_shift(self) -> str | None:
        """Detect if mood has shifted recently. Returns description or None."""
        if len(self._trajectory) < 3:
            return None

        recent = self._trajectory[-3:]
        moods = [e["mood"] for e in recent]

        # All same → no shift
        if len(set(moods)) == 1:
            return None

        # Transition to negative
        if moods[-1] in ("negative", "anxious") and moods[0] not in ("negative", "anxious"):
            return f"mood_shift:{moods[0]}→{moods[-1]}"

        # Transition from negative to positive
        if moods[-1] in ("positive", "playful") and moods[0] in ("negative", "anxious"):
            return f"mood_recovery:{moods[0]}→{moods[-1]}"

        # Any distinct change
        if moods[-1] != moods[-2] and moods[-2] != moods[-3]:
            return f"mood_unstable:{moods[-3]}→{moods[-2]}→{moods[-1]}"

        return None

    def session_summary(self) -> dict:
        """Summarize mood distribution for the session."""
        if not self._trajectory:
            return {"dominant": "neutral", "shifts": 0, "distribution": {}}

        from collections import Counter
        counts = Counter(e["mood"] for e in self._trajectory)
        dominant = counts.most_common(1)[0][0]

        shifts = 0
        for i in range(1, len(self._trajectory)):
            if self._trajectory[i]["mood"] != self._trajectory[i-1]["mood"]:
                shifts += 1

        total = len(self._trajectory)
        dist = {m: round(c / total, 2) for m, c in counts.items()}

        return {
            "dominant": dominant,
            "shifts": shifts,
            "distribution": dist,
            "total_turns": total,
        }

    def current_mood(self) -> dict | None:
        """Return the most recent mood entry, or None if no data."""
        return self._trajectory[-1] if self._trajectory else None

    def context_string(self) -> str:
        """Compact string for interoceptive block."""
        if not self._trajectory:
            return ""

        current = self._trajectory[-1]
        parts = []

        mood = current["mood"]
        if mood != "neutral":
            intensity = current.get("intensity", 0)
            parts.append(f"{mood}({intensity:.1f})")

        shift = self.detect_shift()
        if shift:
            parts.append(shift)

        if not parts:
            return ""
        return "[MOOD] " + " | ".join(parts)
