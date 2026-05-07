"""OCEAN (Big Five) personality profiles — Group B.

Dual profiles: UserOcean (inferred from user messages) and ModelOcean
(inferred from model responses). Each dimension is tracked with an
exponential moving average (EMA) so the profile adapts over time.

Dimensions:
  O — Openness:       curiosity, creativity, variety-seeking
  C — Conscientiousness: organization, precision, task-focus
  E — Extraversion:    sociability, enthusiasm, assertiveness
  A — Agreeableness:   cooperation, politeness, conflict-avoidance
  N — Neuroticism:     anxiety, volatility, negativity
"""

import json
import math
import os
import re
import time
from collections import defaultdict


# ═══════════════════════════════════════════════════
# DIMENSION DEFINITIONS
# ═══════════════════════════════════════════════════

DIMENSIONS = ("O", "C", "E", "A", "N")
DIM_NAMES = {
    "O": "Openness",
    "C": "Conscientiousness",
    "E": "Extraversion",
    "A": "Agreeableness",
    "N": "Neuroticism",
}

# Default midpoint — no information
_DEFAULT = 0.5

# EMA smoothing factor (higher = faster adaptation)
_ALPHA = 0.15


# ═══════════════════════════════════════════════════
# SIGNAL → DIMENSION SCORING
# ═══════════════════════════════════════════════════

# Lexical cues — lightweight keyword heuristics (no ML)
_OPENNESS_HIGH = {
    "imagine", "creative", "explore", "wonder", "curious", "novel",
    "experiment", "innovative", "abstract", "philosophical", "artistic",
    "alternative", "dream", "idea", "brainstorm", "possibilities",
    "unconventional", "fascinating", "intriguing", "hypothetical",
}
_OPENNESS_LOW = {
    "standard", "usual", "normal", "conventional", "traditional",
    "proven", "reliable", "safe", "practical", "straightforward",
}

_CONSCIENTIOUS_HIGH = {
    "plan", "organize", "schedule", "deadline", "priority", "checklist",
    "systematic", "precise", "thorough", "detail", "careful", "accurate",
    "structure", "methodology", "step-by-step", "review", "verify",
}
_CONSCIENTIOUS_LOW = {
    "whatever", "idk", "meh", "yolo", "wing it", "figure it out",
    "roughly", "approximately", "ish", "sorta", "kinda",
}

_EXTRAVERSION_HIGH = {
    "excited", "awesome", "amazing", "love", "great", "fantastic",
    "incredible", "brilliant", "team", "together", "share", "discuss",
    "collaborate", "everyone", "social", "community",
}
_EXTRAVERSION_LOW = {
    "alone", "quiet", "private", "solo", "independent", "prefer",
    "rather not", "by myself", "introvert",
}

_AGREEABLE_HIGH = {
    "please", "thank", "thanks", "appreciate", "kind", "helpful",
    "understand", "agree", "support", "compromise", "cooperate",
    "sorry", "apologies", "fair", "reasonable", "empathy",
}
_AGREEABLE_LOW = {
    "wrong", "disagree", "no", "refuse", "reject", "unacceptable",
    "terrible", "awful", "ridiculous", "stupid", "absurd", "demand",
}

_NEUROTIC_HIGH = {
    "worried", "anxious", "nervous", "stressed", "afraid", "fear",
    "panic", "overwhelmed", "frustrated", "angry", "annoyed", "upset",
    "confused", "lost", "stuck", "hopeless", "desperate", "failing",
}
_NEUROTIC_LOW = {
    "calm", "relaxed", "confident", "fine", "good", "stable",
    "steady", "comfortable", "secure", "peaceful", "assured",
}


def _score_dimension(words: set, high_set: set, low_set: set) -> float | None:
    """Score a dimension from word overlap. Returns None if no signal."""
    high_hits = len(words & high_set)
    low_hits = len(words & low_set)
    total = high_hits + low_hits
    if total == 0:
        return None
    return high_hits / total


def score_text(text: str) -> dict[str, float | None]:
    """Score all 5 dimensions from text using keyword heuristics. None means no signal."""
    words = set(re.findall(r'\b\w+\b', text.lower()))
    return {
        "O": _score_dimension(words, _OPENNESS_HIGH, _OPENNESS_LOW),
        "C": _score_dimension(words, _CONSCIENTIOUS_HIGH, _CONSCIENTIOUS_LOW),
        "E": _score_dimension(words, _EXTRAVERSION_HIGH, _EXTRAVERSION_LOW),
        "A": _score_dimension(words, _AGREEABLE_HIGH, _AGREEABLE_LOW),
        "N": _score_dimension(words, _NEUROTIC_HIGH, _NEUROTIC_LOW),
    }


def score_text_smart(text: str, embedder=None) -> dict[str, float | None]:
    """Score OCEAN using embeddings if available, keyword fallback otherwise."""
    if embedder is None:
        return score_text(text)
    try:
        from graceful.embedder import score_ocean_embedding, _has_embed_method
        if not _has_embed_method(embedder):
            return score_text(text)
        return score_ocean_embedding(text, embedder)
    except Exception:
        return score_text(text)


def score_from_signals(signals: dict, embedder=None, text: str = "") -> dict[str, float | None]:
    """Infer OCEAN hints from HumanProfile signal dict (Group A integration).

    When embedder + text are provided, uses profanity context disambiguation
    for the N dimension instead of blanket penalty.
    """
    scores: dict[str, float | None] = {d: None for d in DIMENSIONS}

    # Openness: higher question count + longer messages → curious
    qc = signals.get("question_count", 0)
    wc = signals.get("word_count", 0)
    if qc >= 2 or wc > 80:
        scores["O"] = min(1.0, 0.5 + qc * 0.1 + (wc - 40) * 0.002)
    elif wc > 0:
        scores["O"] = 0.4

    # Conscientiousness: punctuation density as proxy for careful writing
    pd = signals.get("punct_density", 0)
    if pd > 0:
        scores["C"] = min(1.0, 0.3 + pd * 5)

    # Extraversion: emoji + exclamation
    emoji = signals.get("emoji_count", 0)
    exc = signals.get("exclamation_count", 0)
    if emoji > 0 or exc > 0:
        scores["E"] = min(1.0, 0.4 + emoji * 0.15 + exc * 0.1)

    # Neuroticism: profanity + all_caps — with context disambiguation
    if signals.get("profanity_flag"):
        profanity_ctx = "negative_emotion"  # default assumption
        if embedder and text:
            try:
                from graceful.embedder import classify_profanity_context, _has_embed_method
                if _has_embed_method(embedder):
                    profanity_ctx = classify_profanity_context(text, embedder)
            except Exception:
                pass
        if profanity_ctx == "intensifier":
            scores["N"] = 0.4  # mild, not strong neuroticism
        else:
            scores["N"] = 0.7
    elif signals.get("all_caps"):
        scores["N"] = 0.6
    elif signals.get("exclamation_count", 0) > 2:
        scores["N"] = 0.55

    # ── Behavioral signal enrichment (keyboard-as-interface) ────────
    # These refine OCEAN from *how* you type, not just *what* you type.

    # Openness: high rhythm variance = exploratory typing, long pauses = deliberation
    rhythm_var = signals.get("typing_rhythm_variance", 0)
    if rhythm_var > 50000:  # high variance = uneven rhythm = exploring
        scores["O"] = min(1.0, (scores["O"] or 0.5) + 0.1)
    read_ms = signals.get("read_time_ms", 0)
    if read_ms > 15000:  # long read before responding = absorbing, curious
        scores["O"] = min(1.0, (scores["O"] or 0.5) + 0.08)

    # Conscientiousness: low edit ratio = clean first drafts, high = careful revision
    edit_ratio = signals.get("edit_ratio", 0)
    if edit_ratio > 0.15:  # heavy editing = conscientious self-correction
        scores["C"] = min(1.0, (scores["C"] or 0.5) + 0.12)
    elif edit_ratio == 0 and signals.get("keystroke_count", 0) > 20:
        scores["C"] = max(0.0, (scores["C"] or 0.5) - 0.05)  # no editing on long msg

    # Extraversion: fast typing speed = assertive/expressive
    ks = signals.get("keystroke_count", 0)
    td = signals.get("typing_duration_ms", 0)
    if td > 0 and ks > 0:
        cps = (ks / td) * 1000
        if cps > 5:   # fast typist = more extraverted
            scores["E"] = min(1.0, (scores["E"] or 0.5) + 0.08)
        elif cps < 1.5:  # slow, deliberate = more introverted
            scores["E"] = max(0.0, (scores["E"] or 0.5) - 0.06)

    # Agreeableness: paste ratio as proxy — low paste = original thought
    paste_ratio = signals.get("paste_ratio", 0)
    scroll_ups = signals.get("scroll_ups", 0)
    if scroll_ups >= 2:  # re-reading model's response = engaged, agreeable
        scores["A"] = min(1.0, (scores["A"] or 0.5) + 0.06)
    if paste_ratio > 0.5:  # mostly pasted content = less personal engagement
        scores["A"] = max(0.0, (scores["A"] or 0.5) - 0.08)

    # Neuroticism: high edit ratio + long pauses + rhythm variance = anxiety signal
    if edit_ratio > 0.25 and signals.get("median_pause_ms", 0) > 800:
        scores["N"] = min(1.0, (scores["N"] or 0.5) + 0.1)

    return scores


# ═══════════════════════════════════════════════════
# OCEAN PROFILE CLASS
# ═══════════════════════════════════════════════════

class OceanProfile:
    """EMA-based Big Five profile with persistence."""

    def __init__(self, profile_type: str, data_dir: str, alpha: float = _ALPHA):
        self._type = profile_type  # "user" or "model"
        self._data_dir = data_dir
        self._path = os.path.join(data_dir, f"ocean_{profile_type}.json")
        self._alpha = alpha
        self._update_count = 0

        # Current EMA values
        self.scores: dict[str, float] = {d: _DEFAULT for d in DIMENSIONS}

        # Confidence: how many updates have informed each dimension
        self.sample_counts: dict[str, int] = {d: 0 for d in DIMENSIONS}

        self._load()

    def _load(self):
        if not os.path.isfile(self._path):
            return
        try:
            with open(self._path) as f:
                data = json.load(f)
            for d in DIMENSIONS:
                if d in data.get("scores", {}):
                    self.scores[d] = data["scores"][d]
                if d in data.get("sample_counts", {}):
                    self.sample_counts[d] = data["sample_counts"][d]
            self._update_count = data.get("update_count", 0)
        except (json.JSONDecodeError, OSError):
            pass

    def save(self):
        os.makedirs(self._data_dir, exist_ok=True)
        data = {
            "type": self._type,
            "scores": self.scores,
            "sample_counts": self.sample_counts,
            "update_count": self._update_count,
            "last_updated": time.time(),
        }
        try:
            with open(self._path, "w") as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass

    def update(self, raw_scores: dict[str, float | None]):
        """Update profile with new observations. None values are skipped."""
        self._update_count += 1
        for d in DIMENSIONS:
            val = raw_scores.get(d)
            if val is None:
                continue
            val = max(0.0, min(1.0, val))
            self.sample_counts[d] += 1
            # EMA update
            self.scores[d] = self._alpha * val + (1 - self._alpha) * self.scores[d]

    def confidence(self, dim: str) -> float:
        """How confident are we in this dimension? 0-1 based on sample count."""
        n = self.sample_counts.get(dim, 0)
        # Reaches ~0.9 confidence at 30 samples
        return 1.0 - math.exp(-n / 15)

    def dominant_traits(self, threshold: float = 0.65) -> list[str]:
        """Return dimensions with score above threshold and sufficient confidence."""
        result = []
        for d in DIMENSIONS:
            if self.scores[d] >= threshold and self.confidence(d) >= 0.3:
                result.append(d)
        return result

    def recessive_traits(self, threshold: float = 0.35) -> list[str]:
        """Return dimensions with score below threshold and sufficient confidence."""
        result = []
        for d in DIMENSIONS:
            if self.scores[d] <= threshold and self.confidence(d) >= 0.3:
                result.append(d)
        return result

    def to_summary(self) -> dict:
        """Return structured summary for settings display."""
        summary = {}
        for d in DIMENSIONS:
            summary[DIM_NAMES[d]] = {
                "score": round(self.scores[d], 2),
                "confidence": round(self.confidence(d), 2),
                "samples": self.sample_counts[d],
                "label": _label(self.scores[d]),
            }
        return summary

    def context_string(self) -> str:
        """Compact string for interoceptive block — only surfaces notable traits."""
        parts = []
        for d in DIMENSIONS:
            if self.confidence(d) < 0.3:
                continue
            s = self.scores[d]
            if s >= 0.7:
                parts.append(f"{d}↑")
            elif s <= 0.3:
                parts.append(f"{d}↓")
        if not parts:
            return ""
        return f"[OCEAN-{self._type.upper()}] " + " ".join(parts)


def _label(score: float) -> str:
    if score >= 0.75:
        return "high"
    elif score >= 0.55:
        return "moderate-high"
    elif score >= 0.45:
        return "moderate"
    elif score >= 0.25:
        return "moderate-low"
    return "low"


# ═══════════════════════════════════════════════════
# DANCE COHERENCE — how well model mirrors/complements user
# ═══════════════════════════════════════════════════

def compute_dance_coherence(user: OceanProfile, model: OceanProfile) -> dict:
    """Measure how the model's personality aligns with the user's.

    Returns alignment scores and a recommendation.
    Mirror dimensions: A, E (model should roughly match user)
    Complement dimensions: C (model should be high regardless)
    Independent: O, N
    """
    result = {}
    for d in DIMENSIONS:
        uc = user.confidence(d)
        mc = model.confidence(d)
        if uc < 0.2 or mc < 0.2:
            continue
        diff = model.scores[d] - user.scores[d]
        result[d] = {
            "user": round(user.scores[d], 2),
            "model": round(model.scores[d], 2),
            "diff": round(diff, 2),
        }

    # Overall coherence: penalize large gaps on mirror dimensions
    penalties = 0.0
    for d in ("A", "E"):
        if d in result:
            penalties += abs(result[d]["diff"]) * 0.5

    # Conscientiousness: model should be high
    if "C" in result and result["C"]["model"] < 0.5:
        penalties += 0.3

    # Neuroticism: model should be low
    if "N" in result and result["N"]["model"] > 0.5:
        penalties += 0.2

    coherence = max(0.0, 1.0 - penalties)
    return {
        "dimensions": result,
        "coherence": round(coherence, 2),
    }
