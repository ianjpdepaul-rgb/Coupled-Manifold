"""Dual confidence calibration — Group D.

Tracks model confidence per response and calibrates against actual
outcomes. Displays a compact confidence indicator alongside responses.

Two calibration channels:
  1. Self-reported: model's stated confidence (when it hedges, qualifies, etc.)
  2. Outcome-tracked: did the user accept/correct/reject the response?
"""

import json
import math
import os
import re
import time


# ═══════════════════════════════════════════════════
# CONFIDENCE EXTRACTION
# ═══════════════════════════════════════════════════

# Hedging language that lowers confidence
_HEDGE_PHRASES = [
    "i think", "i believe", "i'm not sure", "i'm not certain",
    "probably", "possibly", "might be", "could be", "may be",
    "it seems", "it appears", "perhaps", "likely", "unlikely",
    "if i recall", "if i remember", "iirc", "afaik",
    "not entirely sure", "don't quote me", "take this with",
    "i could be wrong", "i may be mistaken", "roughly",
    "approximately", "around", "about",
]

# High-confidence language
_CONFIDENT_PHRASES = [
    "definitely", "certainly", "absolutely", "clearly",
    "without a doubt", "for sure", "100%", "guaranteed",
    "always", "never", "must be", "is exactly",
    "the answer is", "this is because", "specifically",
]


def extract_confidence(response: str, embedder=None) -> dict:
    """Extract confidence signals from a model response.

    When embedder is provided, uses sentence embeddings with negation
    awareness for more accurate confidence scoring.

    Returns:
        {
            "self_confidence": float,  # 0-1 inferred from language
            "hedge_count": int,
            "confident_count": int,
            "indicators": list[str],
        }
    """
    # ── Embedding-based scoring (when available) ──
    _emb_confidence = None
    if embedder and response:
        try:
            from graceful.embedder import score_confidence_embedding, _has_embed_method
            if _has_embed_method(embedder):
                conf_scores = score_confidence_embedding(response, embedder)
                h_sim = conf_scores["hedge_sim"]
                c_sim = conf_scores["confident_sim"]
                total_sim = h_sim + c_sim
                if total_sim > 0.01:
                    _emb_confidence = c_sim / total_sim
        except Exception:
            pass

    text_lower = response.lower()
    indicators = []

    hedge_count = 0
    for phrase in _HEDGE_PHRASES:
        if phrase in text_lower:
            hedge_count += 1
            if len(indicators) < 3:
                indicators.append(f"hedge:{phrase}")

    confident_count = 0
    for phrase in _CONFIDENT_PHRASES:
        if phrase in text_lower:
            confident_count += 1
            if len(indicators) < 5:
                indicators.append(f"confident:{phrase}")

    # Base confidence from hedge/confident ratio (keyword path)
    total = hedge_count + confident_count
    if total == 0:
        kw_confidence = 0.6
    else:
        kw_confidence = confident_count / total

    # Blend keyword and embedding scores
    if _emb_confidence is not None:
        self_confidence = 0.4 * kw_confidence + 0.6 * _emb_confidence
        indicators.append("embedding")
    else:
        self_confidence = kw_confidence

    # Adjust by response length — very short responses on complex topics
    # may indicate low confidence (avoidance)
    word_count = len(response.split())
    if word_count < 15 and hedge_count > 0:
        self_confidence *= 0.8

    # Question marks in response (asking for clarification = lower confidence)
    q_marks = response.count("?")
    if q_marks >= 2:
        self_confidence *= 0.85
        indicators.append("asks_questions")

    return {
        "self_confidence": round(max(0.0, min(1.0, self_confidence)), 2),
        "hedge_count": hedge_count,
        "confident_count": confident_count,
        "indicators": indicators,
    }


# ═══════════════════════════════════════════════════
# OUTCOME TRACKING
# ═══════════════════════════════════════════════════

def infer_outcome(user_response: str) -> str:
    """Infer whether the user accepted or corrected the prior response.

    Returns: "accepted", "corrected", "rejected", or "neutral"
    """
    text = user_response.lower().strip()

    # Explicit acceptance
    _ACCEPT = {"yes", "yep", "yup", "yeah", "correct", "right", "exactly",
               "perfect", "great", "thanks", "thank you", "that's it",
               "that works", "got it", "makes sense", "good"}
    words = set(text.split())
    if words & _ACCEPT and len(text) < 40:
        return "accepted"

    # Explicit rejection/correction
    _REJECT = {"no", "wrong", "incorrect", "that's not", "actually",
               "not quite", "not exactly", "that's wrong", "nope"}
    if words & _REJECT:
        return "corrected"

    # "but" after short acknowledgment → partial correction
    if re.match(r'^(ok|okay|sure|right|yeah)\s*(but|however|though)', text):
        return "corrected"

    return "neutral"


# ═══════════════════════════════════════════════════
# CALIBRATION TRACKER
# ═══════════════════════════════════════════════════

class ConfidenceTracker:
    """Tracks and calibrates model confidence over time."""

    def __init__(self, data_dir: str):
        self._data_dir = data_dir
        self._path = os.path.join(data_dir, "confidence_calibration.json")
        self._log_path = os.path.join(data_dir, "logs", "confidence_trace.jsonl")

        # Running calibration bins: confidence range → accuracy
        # Bins: [0.0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0]
        self.bins: dict[str, dict] = {
            f"{i/5:.1f}-{(i+1)/5:.1f}": {"total": 0, "correct": 0}
            for i in range(5)
        }

        # Session state
        self._pending_confidence: float | None = None
        self._turn_confidences: list[dict] = []

        self._load()

    def _bin_key(self, confidence: float) -> str:
        idx = min(int(confidence * 5), 4)
        return f"{idx/5:.1f}-{(idx+1)/5:.1f}"

    def _load(self):
        if not os.path.isfile(self._path):
            return
        try:
            with open(self._path) as f:
                data = json.load(f)
            self.bins = data.get("bins", self.bins)
        except (json.JSONDecodeError, OSError):
            pass

    def save(self):
        os.makedirs(self._data_dir, exist_ok=True)
        data = {
            "bins": self.bins,
            "last_updated": time.time(),
        }
        try:
            with open(self._path, "w") as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass

    def record_response(self, response: str, embedder=None) -> dict:
        """Extract and record confidence for a model response."""
        result = extract_confidence(response, embedder=embedder)
        self._pending_confidence = result["self_confidence"]

        entry = {
            "ts": time.time(),
            **result,
        }
        self._turn_confidences.append(entry)

        # Log
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except OSError:
            pass

        return result

    def record_outcome(self, user_msg: str):
        """Record the user's response to update calibration."""
        if self._pending_confidence is None:
            return

        outcome = infer_outcome(user_msg)
        if outcome == "neutral":
            self._pending_confidence = None
            return

        bin_key = self._bin_key(self._pending_confidence)
        self.bins[bin_key]["total"] += 1
        if outcome == "accepted":
            self.bins[bin_key]["correct"] += 1

        self._pending_confidence = None

    def calibration_error(self) -> float | None:
        """Expected calibration error (ECE) across bins.

        Returns None if insufficient data.
        """
        total_samples = sum(b["total"] for b in self.bins.values())
        if total_samples < 10:
            return None

        ece = 0.0
        for key, bin_data in self.bins.items():
            if bin_data["total"] == 0:
                continue
            lo = float(key.split("-")[0])
            hi = float(key.split("-")[1])
            expected = (lo + hi) / 2
            actual = bin_data["correct"] / bin_data["total"]
            weight = bin_data["total"] / total_samples
            ece += weight * abs(actual - expected)

        return round(ece, 3)

    def get_summary(self) -> dict:
        """Summary for settings panel."""
        total = sum(b["total"] for b in self.bins.values())
        ece = self.calibration_error()

        # Session stats
        session_conf = [e["self_confidence"] for e in self._turn_confidences]
        avg_conf = sum(session_conf) / len(session_conf) if session_conf else 0.5

        return {
            "calibration_error": ece,
            "total_calibrated_turns": total,
            "session_avg_confidence": round(avg_conf, 2),
            "session_turns": len(session_conf),
            "bins": {
                k: {
                    "accuracy": round(v["correct"] / v["total"], 2) if v["total"] > 0 else None,
                    "samples": v["total"],
                }
                for k, v in self.bins.items()
            },
        }

    def context_string(self) -> str:
        """Compact string for interoceptive block."""
        if not self._turn_confidences:
            return ""

        parts = []

        # Latest confidence
        last = self._turn_confidences[-1]
        conf = last["self_confidence"]
        if conf < 0.4:
            parts.append(f"conf:low({conf:.1f})")
        elif conf > 0.8:
            parts.append(f"conf:high({conf:.1f})")

        # Calibration error
        ece = self.calibration_error()
        if ece is not None and ece > 0.15:
            parts.append(f"cal_err:{ece:.2f}")

        # Hedging trend
        if len(self._turn_confidences) >= 3:
            recent_hedges = sum(e.get("hedge_count", 0) for e in self._turn_confidences[-3:])
            if recent_hedges >= 5:
                parts.append("over_hedging")

        if not parts:
            return ""
        return "[CONFIDENCE] " + " | ".join(parts)
