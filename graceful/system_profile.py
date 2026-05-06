"""
Coupled Manifold — system profile (mirrored co-evolving profile).
Accumulates this instance's response patterns toward the user.
Mirrors the user identity model, but tracks the system side of the dance.

Uses Welford's online algorithm for running statistics — no raw storage.
"""

import json
import math
import os
import time


# Message-length buckets for conditional statistics
_BUCKETS = [
    ("tiny", 0, 50),
    ("short", 50, 150),
    ("medium", 150, 500),
    ("long", 500, float("inf")),
]


def _bucket_for(length):
    for name, lo, hi in _BUCKETS:
        if lo <= length < hi:
            return name
    return "long"


def _new_stat():
    """Welford's online accumulator."""
    return {"n": 0, "mean": 0.0, "m2": 0.0}


def _update_stat(stat, value):
    """Update a Welford accumulator with a new value."""
    stat["n"] += 1
    delta = value - stat["mean"]
    stat["mean"] += delta / stat["n"]
    delta2 = value - stat["mean"]
    stat["m2"] += delta * delta2


def _stat_stddev(stat):
    if stat["n"] < 2:
        return 0.0
    return math.sqrt(stat["m2"] / stat["n"])


def _z_score(stat, value):
    """Z-score of value against running statistics."""
    sd = _stat_stddev(stat)
    if sd < 1e-6:
        return 0.0
    return (value - stat["mean"]) / sd


class SystemProfile:
    """Accumulating profile of this instance's response patterns toward this user."""

    _TRACKED_METRICS = [
        "response_length", "search_invocation", "intervention_rate",
        "clarification_rate", "response_latency",
    ]

    def __init__(self, path):
        self.path = path
        self.data = self._default_data()
        self._load()

    def _default_data(self):
        return {
            # Bucketed response length stats (keyed by user-message-length bucket)
            "response_length": {b: _new_stat() for b, _, _ in _BUCKETS},
            # Overall response length
            "response_length_all": _new_stat(),
            # Search invocation rate (running fraction)
            "search_invocation": {"total": 0, "searched": 0},
            # Anti-LoRA intervention rate per session
            "intervention_rate": _new_stat(),
            # Clarification rate (questions asked vs direct answers)
            "clarification_rate": {"total": 0, "clarifications": 0},
            # Register match tracking (formal vs casual)
            "register": {"formal": 0, "casual": 0, "technical": 0, "colloquial": 0},
            # Mode preference (engagement duration by mode)
            "mode_preference": {},
            # Failure recovery patterns
            "failure_recovery": {"pushback": 0, "restart": 0, "ignore": 0, "leave": 0},
            # Response latency stats bucketed by response length
            "response_latency": {b: _new_stat() for b, _, _ in _BUCKETS},
            # Session-level stats
            "sessions_observed": 0,
            "total_turns_observed": 0,
            # Turn snapshots for co-evolution display (capped at 50)
            "turn_snapshots": [],
            # Dance coherence predictions
            "_predictions": [],  # recent predictions for next-turn comparison
            "dance_coherence": _new_stat(),
            # Last update timestamp
            "last_updated": 0,
        }

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    saved = json.load(f)
                # Merge saved into default (handles new fields gracefully)
                self._merge(saved)
            except (json.JSONDecodeError, OSError) as e:
                print(f"  SystemProfile: load failed ({e}), starting fresh")

    def _merge(self, saved):
        """Merge saved data into self.data, preserving structure."""
        for key in self.data:
            if key in saved:
                if isinstance(self.data[key], dict) and isinstance(saved[key], dict):
                    self.data[key].update(saved[key])
                else:
                    self.data[key] = saved[key]

    def _save(self):
        self.data["last_updated"] = time.time()
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"  SystemProfile: save failed ({e})")

    def observe(self, turn):
        """Update from one completed turn.

        turn: dict with keys:
            user_msg:       str — user's message
            response:       str — system's response
            gen_time:       float — generation time in seconds
            mode:           str — "lora" or "anti"
            searched:       bool — whether web search was used
            flattery_score: float — sycophancy score
            trace:          float or None
            coupled_nudge:  str or None
            framework_conf: float — framework confidence 0-1
        """
        user_len = len(turn.get("user_msg", ""))
        resp_len = len(turn.get("response", ""))
        bucket = _bucket_for(user_len)

        # Response length by bucket
        if bucket in self.data["response_length"]:
            _update_stat(self.data["response_length"][bucket], resp_len)
        _update_stat(self.data["response_length_all"], resp_len)

        # Search invocation
        self.data["search_invocation"]["total"] += 1
        if turn.get("searched"):
            self.data["search_invocation"]["searched"] += 1

        # Clarification detection (simple: response ends with ?)
        self.data["clarification_rate"]["total"] += 1
        resp_text = turn.get("response", "").strip()
        if resp_text.endswith("?"):
            self.data["clarification_rate"]["clarifications"] += 1

        # Register tracking
        _register = self._detect_register(resp_text)
        if _register in self.data["register"]:
            self.data["register"][_register] += 1

        # Mode preference — track current mode engagement
        _active_mode = turn.get("active_mode")
        if _active_mode:
            self.data["mode_preference"][_active_mode] = (
                self.data["mode_preference"].get(_active_mode, 0) + 1)

        # Anti-LoRA intervention
        if turn.get("mode") == "anti":
            _update_stat(self.data.get("intervention_rate", _new_stat()), 1)
        else:
            _update_stat(self.data.get("intervention_rate", _new_stat()), 0)

        self.data["total_turns_observed"] += 1

        # Dance coherence — compare prediction to actual
        self._update_dance_coherence(turn)

        # Turn snapshot (capped)
        snapshot = {
            "t": time.time(),
            "turn": self.data["total_turns_observed"],
            "resp_len": resp_len,
            "user_len": user_len,
            "searched": turn.get("searched", False),
            "mode": turn.get("mode"),
            "fw_conf": turn.get("framework_conf"),
        }
        self.data["turn_snapshots"].append(snapshot)
        if len(self.data["turn_snapshots"]) > 50:
            self.data["turn_snapshots"] = self.data["turn_snapshots"][-50:]

        self._save()

    def _detect_register(self, text):
        """Simple register classification."""
        lower = text.lower()
        formal_markers = ["however", "furthermore", "therefore", "consequently",
                          "nevertheless", "regarding", "thus"]
        casual_markers = ["hey", "yeah", "gonna", "wanna", "cool", "awesome",
                          "btw", "lol", "haha"]
        tech_markers = ["function", "class ", "def ", "import ", "module", "api",
                        "endpoint", "algorithm", "parameter", "variable"]

        f_count = sum(1 for m in formal_markers if m in lower)
        c_count = sum(1 for m in casual_markers if m in lower)
        t_count = sum(1 for m in tech_markers if m in lower)

        if t_count >= 2:
            return "technical"
        if f_count > c_count:
            return "formal"
        if c_count > f_count:
            return "casual"
        return "colloquial"

    def _update_dance_coherence(self, turn):
        """Compare recent predictions against actual user behavior."""
        if not self.data.get("_predictions"):
            return

        preds = self.data["_predictions"]
        if not preds:
            return

        last_pred = preds[-1]
        actual_len = len(turn.get("user_msg", ""))
        predicted_bucket = last_pred.get("expected_user_bucket")
        actual_bucket = _bucket_for(actual_len)

        # Simple coherence: did we predict the right bucket?
        coherence = 1.0 if predicted_bucket == actual_bucket else 0.0

        # Also check if response time matches expectation
        if "expected_response_len" in last_pred:
            resp_len = len(turn.get("response", ""))
            pred_len = last_pred["expected_response_len"]
            if pred_len > 0:
                ratio = resp_len / pred_len
                coherence = (coherence + max(0, 1.0 - abs(1.0 - ratio))) / 2

        _update_stat(self.data["dance_coherence"], coherence)
        # Clear predictions after comparison
        self.data["_predictions"] = []

    def predict_next(self, user_msg):
        """Store a prediction for the next turn, to be evaluated on next observe()."""
        user_len = len(user_msg)
        bucket = _bucket_for(user_len)

        # Predict response length based on this user-message bucket
        bucket_stat = self.data["response_length"].get(bucket, _new_stat())
        expected_resp_len = bucket_stat["mean"] if bucket_stat["n"] > 2 else 200

        self.data["_predictions"].append({
            "expected_user_bucket": bucket,
            "expected_response_len": expected_resp_len,
            "t": time.time(),
        })

    def calibrate_response(self):
        """Suggest response parameters based on accumulated patterns.

        Returns dict of suggestions (not overrides):
            suggested_length: int — target response length in chars
            suggested_register: str — "formal", "casual", "technical"
            search_tendency: float — 0-1, how often search helps this user
        """
        # Response length suggestion
        all_stat = self.data["response_length_all"]
        suggested_len = int(all_stat["mean"]) if all_stat["n"] > 5 else None

        # Register suggestion
        reg = self.data["register"]
        max_reg = max(reg, key=reg.get) if any(reg.values()) else None

        # Search tendency
        si = self.data["search_invocation"]
        search_rate = si["searched"] / max(si["total"], 1)

        return {
            "suggested_length": suggested_len,
            "suggested_register": max_reg,
            "search_tendency": round(search_rate, 2),
        }

    def z_score(self, metric, value):
        """Compute z-score for a metric value against running statistics."""
        if metric == "response_length":
            return _z_score(self.data["response_length_all"], value)
        if metric == "dance_coherence":
            return _z_score(self.data["dance_coherence"], value)
        return 0.0

    def get_dance_coherence(self):
        """Return current dance coherence value (0-1 average)."""
        dc = self.data["dance_coherence"]
        if dc["n"] < 3:
            return None
        return round(dc["mean"], 2)

    def is_non_convergent(self, window=10):
        """Check for sustained low dance coherence (Task 5)."""
        dc = self.data["dance_coherence"]
        if dc["n"] < window:
            return False, None
        coherence = dc["mean"]
        if coherence < 0.3:
            return True, (
                "the dance isn't settling. Either you're in a period of variability "
                "that the system can't yet model, or the framework's measurement choices "
                "aren't capturing what's actually varying about you.")
        return False, None

    def maturity(self):
        """Profile maturity as fraction (0-1) of metrics with sufficient data."""
        thresholds = {
            "response_length_all": 20,
            "search_invocation": 20,
            "intervention_rate": 10,
            "clarification_rate": 20,
            "dance_coherence": 10,
        }
        mature = 0
        for metric, threshold in thresholds.items():
            data = self.data.get(metric)
            if data is None:
                continue
            if isinstance(data, dict) and "n" in data:
                if data["n"] >= threshold:
                    mature += 1
            elif isinstance(data, dict) and "total" in data:
                if data["total"] >= threshold:
                    mature += 1
        return round(mature / max(len(thresholds), 1), 2)

    def to_context_string(self):
        """Brief summary for model's interoceptive block."""
        parts = []
        all_stat = self.data["response_length_all"]
        if all_stat["n"] > 5:
            sd = _stat_stddev(all_stat)
            parts.append(f"avg response: {all_stat['mean']:.0f} chars (±{sd:.0f})")

        si = self.data["search_invocation"]
        if si["total"] > 5:
            rate = si["searched"] / max(si["total"], 1)
            parts.append(f"search: {rate:.0%} of turns")

        cr = self.data["clarification_rate"]
        if cr["total"] > 5:
            rate = cr["clarifications"] / max(cr["total"], 1)
            parts.append(f"clarify: {rate:.0%}")

        dc = self.get_dance_coherence()
        if dc is not None:
            parts.append(f"dance: {dc:.0%}")

        mat = self.maturity()
        parts.append(f"maturity: {mat:.0%}")

        if not parts:
            return "System profile: building (< 5 turns)"
        return "System profile: " + " | ".join(parts)

    def to_display_dict(self):
        """Structured data for the settings panel."""
        all_stat = self.data["response_length_all"]
        si = self.data["search_invocation"]
        cr = self.data["clarification_rate"]
        dc = self.get_dance_coherence()

        return {
            "response_length_avg": round(all_stat["mean"]) if all_stat["n"] > 3 else None,
            "response_length_range": (
                f"{all_stat['mean'] - 2*_stat_stddev(all_stat):.0f}-{all_stat['mean'] + 2*_stat_stddev(all_stat):.0f}"
                if all_stat["n"] > 5 else None),
            "search_rate": (
                f"{si['searched']/max(si['total'],1):.0%}"
                if si["total"] > 3 else None),
            "clarification_rate": (
                f"{cr['clarifications']/max(cr['total'],1):.0%}"
                if cr["total"] > 3 else None),
            "dominant_register": (
                max(self.data["register"], key=self.data["register"].get)
                if any(self.data["register"].values()) else None),
            "mode_preferences": self.data["mode_preference"],
            "dance_coherence": dc,
            "maturity": self.maturity(),
            "total_turns": self.data["total_turns_observed"],
            "sessions": self.data["sessions_observed"],
            "turn_snapshots": self.data["turn_snapshots"][-20:],  # last 20 for chart
        }

    def reset(self):
        """Reset all accumulated data."""
        self.data = self._default_data()
        self._save()

    def new_session(self):
        """Mark the start of a new session."""
        self.data["sessions_observed"] += 1
        self._save()
