"""Cross-session pattern detection — Group G (Pass 7).

Detects recurring topics, unresolved threads, and behavioral patterns
across sessions. Feeds into the interoceptive block and proactive triggers.
"""

import json
import os
import re
import time
from collections import Counter, defaultdict


# ═══════════════════════════════════════════════════
# TOPIC EXTRACTION (lightweight, no ML)
# ═══════════════════════════════════════════════════

_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "don", "now", "and", "but", "or", "if", "while", "because", "until",
    "about", "that", "this", "these", "those", "what", "which", "who",
    "whom", "it", "its", "i", "me", "my", "we", "our", "you", "your",
    "he", "him", "his", "she", "her", "they", "them", "their", "up",
    "like", "get", "got", "go", "went", "make", "made", "know", "think",
    "want", "see", "look", "find", "give", "tell", "say", "said", "one",
    "also", "back", "use", "way", "even", "new", "well", "also", "thing",
}


def extract_topics(text: str, max_topics: int = 10) -> list[str]:
    """Extract key topic words from text (stop-word filtered)."""
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    filtered = [w for w in words if w not in _STOP_WORDS]
    counts = Counter(filtered)
    return [w for w, _ in counts.most_common(max_topics)]


def extract_questions(text: str) -> list[str]:
    """Extract question sentences from text."""
    sentences = re.split(r'[.!?]+', text)
    questions = []
    for s in sentences:
        s = s.strip()
        if s.endswith('?') or s.lower().startswith(('how', 'what', 'why', 'when', 'where', 'who', 'can', 'could', 'would', 'should', 'is', 'are', 'do', 'does')):
            if len(s) > 10:
                questions.append(s)
    return questions


# ═══════════════════════════════════════════════════
# CROSS-SESSION TRACKER
# ═══════════════════════════════════════════════════

class CrossSessionTracker:
    """Tracks patterns across sessions for continuity and proactive insight."""

    def __init__(self, data_dir: str):
        self._data_dir = data_dir
        self._path = os.path.join(data_dir, "cross_session.json")
        self._log_path = os.path.join(data_dir, "logs", "cross_session.jsonl")

        # Persistent state
        self.topic_frequency: dict[str, int] = {}  # topic → count across sessions
        self.unresolved_threads: list[dict] = []    # questions/tasks left open
        self.session_summaries: list[dict] = []     # compact per-session summaries
        self.recurring_patterns: list[dict] = []    # detected repeating behaviors

        self._load()

    def _load(self):
        if not os.path.isfile(self._path):
            return
        try:
            with open(self._path) as f:
                data = json.load(f)
            self.topic_frequency = data.get("topic_frequency", {})
            self.unresolved_threads = data.get("unresolved_threads", [])
            self.session_summaries = data.get("session_summaries", [])
            self.recurring_patterns = data.get("recurring_patterns", [])
        except (json.JSONDecodeError, OSError):
            pass

    def save(self):
        os.makedirs(self._data_dir, exist_ok=True)
        data = {
            "topic_frequency": self.topic_frequency,
            "unresolved_threads": self.unresolved_threads[-50:],
            "session_summaries": self.session_summaries[-100:],
            "recurring_patterns": self.recurring_patterns[-20:],
            "last_updated": time.time(),
        }
        try:
            with open(self._path, "w") as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass

    def record_session(self, session_log: list[dict]):
        """Process a completed session and update cross-session state."""
        if not session_log:
            return

        # Extract all user messages
        user_msgs = [t.get("user", "") for t in session_log if t.get("user")]
        all_text = " ".join(user_msgs)

        # Topic extraction
        topics = extract_topics(all_text, max_topics=15)
        for topic in topics:
            self.topic_frequency[topic] = self.topic_frequency.get(topic, 0) + 1

        # Question extraction — potential unresolved threads
        for msg in user_msgs:
            questions = extract_questions(msg)
            for q in questions:
                self.unresolved_threads.append({
                    "question": q,
                    "ts": time.time(),
                    "resolved": False,
                })

        # Session summary
        turn_count = len(session_log)
        avg_trace = None
        traces = [t.get("trace") for t in session_log if isinstance(t.get("trace"), (int, float))]
        if traces:
            avg_trace = round(sum(traces) / len(traces), 1)

        moods = [t.get("mode", "working") for t in session_log]
        dominant_mode = Counter(moods).most_common(1)[0][0] if moods else "working"

        self.session_summaries.append({
            "ts": time.time(),
            "turn_count": turn_count,
            "avg_trace": avg_trace,
            "dominant_mode": dominant_mode,
            "top_topics": topics[:5],
        })

        # Detect recurring patterns
        self._detect_patterns()

        # Trim old unresolved threads (>7 days)
        cutoff = time.time() - 7 * 86400
        self.unresolved_threads = [
            t for t in self.unresolved_threads
            if t.get("ts", 0) > cutoff or not t.get("resolved", False)
        ]

        self.save()
        self._log("session_recorded", {"turns": turn_count, "topics": topics[:5]})

    def resolve_thread(self, question_fragment: str):
        """Mark a thread as resolved when the topic is addressed."""
        frag_lower = question_fragment.lower()
        for thread in self.unresolved_threads:
            if frag_lower in thread.get("question", "").lower():
                thread["resolved"] = True
                thread["resolved_at"] = time.time()

    def _detect_patterns(self):
        """Detect recurring behavioral patterns from session summaries."""
        if len(self.session_summaries) < 3:
            return

        recent = self.session_summaries[-10:]

        # Pattern: consistently short sessions
        short_count = sum(1 for s in recent if s.get("turn_count", 0) < 5)
        if short_count >= 5:
            self._add_pattern("short_sessions",
                              f"{short_count}/10 recent sessions under 5 turns")

        # Pattern: recurring topics (appear in 3+ sessions)
        for topic, count in self.topic_frequency.items():
            if count >= 5:
                self._add_pattern(f"recurring_topic:{topic}",
                                  f"'{topic}' appears in {count} sessions")

        # Pattern: declining trace
        traces = [s.get("avg_trace") for s in recent if s.get("avg_trace") is not None]
        if len(traces) >= 3:
            if all(traces[i] > traces[i+1] for i in range(len(traces)-1)):
                self._add_pattern("declining_trace",
                                  "Average trace declining across recent sessions")

    def _add_pattern(self, pattern_id: str, description: str):
        """Add a pattern if not already recorded."""
        for p in self.recurring_patterns:
            if p.get("id") == pattern_id:
                p["last_seen"] = time.time()
                p["count"] = p.get("count", 1) + 1
                return
        self.recurring_patterns.append({
            "id": pattern_id,
            "description": description,
            "first_seen": time.time(),
            "last_seen": time.time(),
            "count": 1,
        })

    def get_continuity_context(self) -> str:
        """Context string for session start — what happened last time."""
        if not self.session_summaries:
            return ""

        last = self.session_summaries[-1]
        parts = []

        topics = last.get("top_topics", [])
        if topics:
            parts.append(f"last_topics:{','.join(topics[:3])}")

        # Unresolved threads
        unresolved = [t for t in self.unresolved_threads if not t.get("resolved")]
        if unresolved:
            parts.append(f"open_threads:{len(unresolved)}")

        # Recurring patterns
        active_patterns = [p for p in self.recurring_patterns
                          if time.time() - p.get("last_seen", 0) < 7 * 86400]
        if active_patterns:
            parts.append(f"patterns:{len(active_patterns)}")

        if not parts:
            return ""
        return "[CONTINUITY] " + " | ".join(parts)

    def get_summary(self) -> dict:
        """Full summary for settings panel."""
        top_topics = sorted(self.topic_frequency.items(), key=lambda x: -x[1])[:20]
        unresolved = [t for t in self.unresolved_threads if not t.get("resolved")]
        return {
            "total_sessions": len(self.session_summaries),
            "top_topics": dict(top_topics),
            "unresolved_threads": len(unresolved),
            "recurring_patterns": [
                {"id": p["id"], "description": p["description"], "count": p.get("count", 1)}
                for p in self.recurring_patterns
            ],
        }

    def _log(self, action: str, detail: dict):
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps({"ts": time.time(), "action": action, **detail}, default=str) + "\n")
        except OSError:
            pass
