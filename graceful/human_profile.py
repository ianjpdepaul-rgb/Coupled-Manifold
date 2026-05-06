"""Extended human-side trace — Group A.

Tracks per-message, per-turn-cycle, and per-session signals from user behavior.
Uses Welford's online algorithm for running statistics. Infers emotional valence,
urgency, engagement quality, and conversational mode.
"""

import json
import math
import os
import re
import time
from collections import defaultdict


# ═══════════════════════════════════════════════════
# WELFORD'S ONLINE ALGORITHM
# ═══════════════════════════════════════════════════

class WelfordStats:
    """Online mean/variance computation via Welford's algorithm."""

    __slots__ = ("n", "mean", "_m2")

    def __init__(self, n: int = 0, mean: float = 0.0, m2: float = 0.0):
        self.n = n
        self.mean = mean
        self._m2 = m2

    def update(self, value: float):
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self._m2 += delta * delta2

    @property
    def variance(self) -> float:
        return self._m2 / self.n if self.n > 1 else 0.0

    @property
    def stddev(self) -> float:
        return math.sqrt(self.variance)

    def z_score(self, value: float) -> float:
        """How many stddevs is value from mean."""
        if self.stddev < 1e-9:
            return 0.0
        return (value - self.mean) / self.stddev

    def is_anomalous(self, value: float, threshold: float = 2.0) -> bool:
        return self.n >= 5 and abs(self.z_score(value)) > threshold

    def to_dict(self) -> dict:
        return {"n": self.n, "mean": self.mean, "m2": self._m2}

    @classmethod
    def from_dict(cls, d: dict) -> "WelfordStats":
        return cls(n=d.get("n", 0), mean=d.get("mean", 0.0), m2=d.get("m2", 0.0))


# ═══════════════════════════════════════════════════
# SIGNAL EXTRACTION
# ═══════════════════════════════════════════════════

_PROFANITY_WORDS = {
    "fuck", "shit", "damn", "hell", "ass", "crap", "bitch",
    "wtf", "omg", "fml", "stfu", "lmao",
}

_SLANG_WORDS = {
    "gonna", "wanna", "gotta", "kinda", "sorta", "ya", "yep", "nah",
    "nope", "bruh", "dude", "lol", "haha", "tbh", "imo", "afaik",
    "idk", "ngl", "rn", "fr", "sus", "lit", "vibe", "chill",
}

_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U0001f900-\U0001f9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "]+", flags=re.UNICODE
)


def extract_message_signals(text: str, human_state: dict | None = None) -> dict:
    """Extract all signals from a single user message.

    text: the message text
    human_state: frontend-provided typing metrics (if available)
    Returns a dict of all extracted signals.
    """
    words = text.split()
    word_count = len(words)
    char_count = len(text)

    # Punctuation analysis
    punct_counts = defaultdict(int)
    for c in text:
        if c in ".,!?;:-…()\"'":
            punct_counts[c] += 1
    punct_density = sum(punct_counts.values()) / max(char_count, 1)

    # Capitalization
    upper_chars = sum(1 for c in text if c.isupper())
    alpha_chars = sum(1 for c in text if c.isalpha())
    caps_ratio = upper_chars / max(alpha_chars, 1)
    all_caps = text.isupper() and len(text) > 3

    # Emoji
    emojis = _EMOJI_RE.findall(text)
    emoji_count = len(emojis)

    # Profanity
    lower_words = set(w.lower().strip(".,!?;:") for w in words)
    profanity_flag = bool(lower_words & _PROFANITY_WORDS)

    # Slang/abbreviation ratio
    slang_count = len(lower_words & _SLANG_WORDS)
    slang_ratio = slang_count / max(word_count, 1)

    # Question marks and exclamation
    question_count = text.count("?")
    exclamation_count = text.count("!")

    signals = {
        "char_count": char_count,
        "word_count": word_count,
        "punct_density": round(punct_density, 4),
        "caps_ratio": round(caps_ratio, 4),
        "all_caps": all_caps,
        "emoji_count": emoji_count,
        "profanity_flag": profanity_flag,
        "slang_ratio": round(slang_ratio, 4),
        "question_count": question_count,
        "exclamation_count": exclamation_count,
        "ts": time.time(),
    }

    # Merge frontend-provided human state
    if human_state:
        signals["typing_duration_ms"] = human_state.get("typing_duration_ms", 0)
        signals["keystroke_count"] = human_state.get("keystroke_count", 0)
        signals["edit_ratio"] = human_state.get("edit_ratio", 0)
        signals["median_pause_ms"] = human_state.get("median_pause_ms", 0)
        signals["max_pause_ms"] = human_state.get("max_pause_ms", 0)
        signals["read_time_ms"] = human_state.get("read_time_ms", 0)
        signals["scroll_ups"] = human_state.get("scroll_ups", 0)
        # New signals from extended capture
        signals["typing_rhythm_variance"] = human_state.get("typing_rhythm_variance", 0)
        signals["paste_ratio"] = human_state.get("paste_ratio", 0)

    return signals


# ═══════════════════════════════════════════════════
# INFERRED STATE
# ═══════════════════════════════════════════════════

def infer_emotional_valence(signals: dict, profanity_context: str = "") -> float:
    """Rough emotional valence estimate: -1 (negative) to +1 (positive).

    Based on punctuation, emoji, caps, profanity patterns.
    profanity_context: 'intensifier', 'negative_emotion', 'ambiguous', or '' (unknown)
    """
    score = 0.0
    if signals.get("profanity_flag"):
        if profanity_context == "intensifier":
            score += 0.1  # profanity as emphasis, not negativity
        elif profanity_context == "negative_emotion":
            score -= 0.4
        else:
            score -= 0.25  # default: mild penalty when context unknown
    if signals.get("all_caps"):
        score -= 0.3
    if signals.get("exclamation_count", 0) > 2:
        score -= 0.1  # excessive exclamation can be frustration
    if signals.get("emoji_count", 0) > 0:
        score += 0.2
    if signals.get("question_count", 0) > 0:
        score += 0.1  # engagement
    # Moderate punctuation is neutral/positive
    pd = signals.get("punct_density", 0)
    if 0.02 < pd < 0.08:
        score += 0.1
    return max(-1.0, min(1.0, score))


def infer_urgency(signals: dict, stats: dict[str, WelfordStats] | None = None) -> float:
    """Urgency estimate: 0 (relaxed) to 1 (urgent).

    Based on typing speed deviation and message length deviation from baseline.
    """
    score = 0.0
    if stats:
        wc = signals.get("word_count", 0)
        wc_stat = stats.get("word_count")
        if wc_stat and wc_stat.n > 5:
            if wc_stat.z_score(wc) > 1.5:
                score += 0.3  # unusually long message
            elif wc_stat.z_score(wc) < -1.5:
                score += 0.2  # unusually short (terse = urgent?)

        td = signals.get("typing_duration_ms", 0)
        td_stat = stats.get("typing_duration_ms")
        if td_stat and td_stat.n > 5 and td > 0:
            if td_stat.z_score(td) < -1.5:
                score += 0.3  # typing faster than usual

    if signals.get("all_caps"):
        score += 0.2
    if signals.get("exclamation_count", 0) > 1:
        score += 0.1

    return min(1.0, score)


def infer_mode(signals: dict) -> str:
    """Infer conversational mode: working/exploring/venting/socializing/teaching.

    Based on message patterns.
    """
    wc = signals.get("word_count", 0)
    qc = signals.get("question_count", 0)
    ec = signals.get("exclamation_count", 0)
    profanity = signals.get("profanity_flag", False)
    slang = signals.get("slang_ratio", 0)
    emoji = signals.get("emoji_count", 0)

    if profanity and (signals.get("all_caps") or ec > 1):
        return "venting"
    if qc >= 2:
        return "exploring"
    if wc > 100:
        return "teaching"  # long messages often explain/teach
    if emoji > 1 or slang > 0.3:
        return "socializing"
    return "working"


# ═══════════════════════════════════════════════════
# HUMAN PROFILE
# ═══════════════════════════════════════════════════

class HumanProfile:
    """Extended human-side profile with running statistics and inferred state."""

    TRACKED_SIGNALS = [
        "char_count", "word_count", "typing_duration_ms", "keystroke_count",
        "edit_ratio", "median_pause_ms", "max_pause_ms", "read_time_ms",
        "scroll_ups", "punct_density", "caps_ratio", "emoji_count",
        "slang_ratio",
    ]

    def __init__(self, data_dir: str):
        self._data_dir = data_dir
        self._profile_path = os.path.join(data_dir, "human_profile.json")
        self._log_path = os.path.join(data_dir, "logs", "human_state_extended.jsonl")

        # Running stats for all tracked signals
        self.stats: dict[str, WelfordStats] = {
            s: WelfordStats() for s in self.TRACKED_SIGNALS
        }

        # Session trajectory
        self._session_message_lengths: list[int] = []
        self._session_paces: list[float] = []  # chars per second

        # Inferred state (current)
        self.emotional_valence: float = 0.0
        self.urgency: float = 0.0
        self.engagement_quality: float = 0.5
        self.mode: str = "working"

        self._load()

    def _load(self):
        if not os.path.isfile(self._profile_path):
            return
        try:
            with open(self._profile_path) as f:
                data = json.load(f)
            for name, stat_data in data.get("stats", {}).items():
                if name in self.stats:
                    self.stats[name] = WelfordStats.from_dict(stat_data)
        except (json.JSONDecodeError, OSError):
            pass

    def save(self):
        os.makedirs(self._data_dir, exist_ok=True)
        data = {
            "stats": {name: stat.to_dict() for name, stat in self.stats.items()},
            "last_updated": time.time(),
        }
        try:
            with open(self._profile_path, "w") as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass

    def update(self, text: str, human_state: dict | None = None,
               prior_response: str = "", embedder=None) -> dict:
        """Process a new user message. Returns the full signal dict."""
        signals = extract_message_signals(text, human_state)

        # Update running stats
        for name in self.TRACKED_SIGNALS:
            if name in signals and isinstance(signals[name], (int, float)):
                self.stats[name].update(signals[name])

        # Track session trajectory
        self._session_message_lengths.append(signals["word_count"])
        td = signals.get("typing_duration_ms", 0)
        cc = signals.get("char_count", 1)
        if td > 0:
            self._session_paces.append(cc / (td / 1000))

        # Compute profanity context if embedder available
        _prof_ctx = ""
        if signals.get("profanity_flag") and embedder and text:
            try:
                from graceful.embedder import classify_profanity_context, _has_embed_method
                if _has_embed_method(embedder):
                    _prof_ctx = classify_profanity_context(text, embedder)
            except Exception:
                pass

        # Compute inferred state
        self.emotional_valence = infer_emotional_valence(signals, profanity_context=_prof_ctx)
        self.urgency = infer_urgency(signals, self.stats)
        self.mode = infer_mode(signals)

        # Engagement quality: how much does user reference prior response?
        if prior_response:
            prior_words = set(prior_response.lower().split())
            user_words = set(text.lower().split())
            overlap = len(prior_words & user_words)
            self.engagement_quality = min(1.0, overlap / max(len(prior_words), 1) * 5)
        else:
            self.engagement_quality = 0.5

        # Add inferred state to signals
        signals["emotional_valence"] = round(self.emotional_valence, 3)
        signals["urgency"] = round(self.urgency, 3)
        signals["engagement_quality"] = round(self.engagement_quality, 3)
        signals["mode"] = self.mode

        # Check for anomalies
        anomalies = []
        for name in self.TRACKED_SIGNALS:
            if name in signals and isinstance(signals[name], (int, float)):
                stat = self.stats[name]
                if stat.is_anomalous(signals[name]):
                    anomalies.append(name)
        signals["anomalies"] = anomalies

        # Log
        self._log(signals)

        return signals

    def context_string(self, signals: dict | None = None) -> str:
        """Compressed string for interoceptive block.

        Only surfaces notable/anomalous signals. Most turns return empty or one line.
        """
        if not signals:
            return ""

        parts = []

        # Mood shift
        if abs(self.emotional_valence) > 0.3:
            v = "positive" if self.emotional_valence > 0 else "negative"
            parts.append(f"valence:{v}({self.emotional_valence:+.1f})")

        if self.urgency > 0.5:
            parts.append(f"urgency:{'high' if self.urgency > 0.7 else 'elevated'}")

        if self.mode != "working":
            parts.append(f"mode:{self.mode}")

        if signals.get("anomalies"):
            parts.append(f"anomalous:{','.join(signals['anomalies'][:3])}")

        if self.engagement_quality < 0.2:
            parts.append("low_engagement")

        if not parts:
            return ""
        return "[HUMAN] " + " | ".join(parts)

    def session_trajectory(self) -> str:
        """Describe how pace/length has changed across session."""
        if len(self._session_message_lengths) < 3:
            return "stable"
        first_half = self._session_message_lengths[:len(self._session_message_lengths)//2]
        second_half = self._session_message_lengths[len(self._session_message_lengths)//2:]
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        if avg_second > avg_first * 1.3:
            return "expanding"
        elif avg_second < avg_first * 0.7:
            return "contracting"
        return "stable"

    def get_profile_summary(self) -> dict:
        """Return full profile for settings panel display."""
        summary = {}
        for name, stat in self.stats.items():
            if stat.n > 0:
                summary[name] = {
                    "mean": round(stat.mean, 2),
                    "stddev": round(stat.stddev, 2),
                    "samples": stat.n,
                }
        summary["session_trajectory"] = self.session_trajectory()
        summary["current_state"] = {
            "valence": round(self.emotional_valence, 2),
            "urgency": round(self.urgency, 2),
            "engagement": round(self.engagement_quality, 2),
            "mode": self.mode,
        }
        return summary

    def _log(self, signals: dict):
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(signals, default=str) + "\n")
        except OSError:
            pass
