"""Temporal awareness — Phase 5 (E1).

Provides TemporalAwareness class that tracks time context continuously.
Background thread updates every minute. Provides temporal context string
for the interoceptive block.
"""

import json
import os
import threading
import time
from collections import defaultdict
from datetime import datetime


_TIME_BUCKETS = {
    (5, 12): "morning",
    (12, 17): "afternoon",
    (17, 21): "evening",
    (21, 24): "night",
    (0, 5): "late_night",
}


def _time_bucket(hour: int) -> str:
    for (lo, hi), name in _TIME_BUCKETS.items():
        if lo <= hour < hi:
            return name
    return "night"


class TemporalAwareness:
    """Full temporal context for the model."""

    def __init__(self, data_dir: str):
        self._data_dir = data_dir
        self._state_path = os.path.join(data_dir, "temporal_state.json")
        self._pattern_path = os.path.join(data_dir, "temporal_patterns.json")

        # Live state
        self.session_start: float = time.time()
        self.last_message_time: float = time.time()
        self.last_session_end: float | None = None
        self.session_count_today: int = 0
        self.session_count_week: int = 0

        # Pattern data — histogram of activity across hours (0-23) and days (0-6)
        self._hour_histogram: dict[int, int] = defaultdict(int)
        self._day_histogram: dict[int, int] = defaultdict(int)

        # Load persisted state
        self._load()

        # Background thread
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._background_loop, daemon=True)
        self._thread.start()

    def _load(self):
        """Load persisted temporal state."""
        if os.path.isfile(self._state_path):
            try:
                with open(self._state_path) as f:
                    state = json.load(f)
                self.last_session_end = state.get("last_session_end")
                self.session_count_today = state.get("session_count_today", 0)
                self.session_count_week = state.get("session_count_week", 0)
                # Check if count is from today
                last_date = state.get("count_date", "")
                if last_date != time.strftime("%Y-%m-%d"):
                    self.session_count_today = 0
                # Check if week count is current
                last_week = state.get("count_week", "")
                if last_week != time.strftime("%Y-W%W"):
                    self.session_count_week = 0
            except (json.JSONDecodeError, OSError):
                pass

        if os.path.isfile(self._pattern_path):
            try:
                with open(self._pattern_path) as f:
                    patterns = json.load(f)
                self._hour_histogram = defaultdict(int, {
                    int(k): v for k, v in patterns.get("hours", {}).items()
                })
                self._day_histogram = defaultdict(int, {
                    int(k): v for k, v in patterns.get("days", {}).items()
                })
            except (json.JSONDecodeError, OSError):
                pass

    def save(self):
        """Persist temporal state to disk."""
        os.makedirs(self._data_dir, exist_ok=True)
        state = {
            "last_session_end": self.last_session_end,
            "session_count_today": self.session_count_today,
            "session_count_week": self.session_count_week,
            "count_date": time.strftime("%Y-%m-%d"),
            "count_week": time.strftime("%Y-W%W"),
        }
        try:
            with open(self._state_path, "w") as f:
                json.dump(state, f, indent=2)
        except OSError:
            pass

        patterns = {
            "hours": dict(self._hour_histogram),
            "days": dict(self._day_histogram),
        }
        try:
            with open(self._pattern_path, "w") as f:
                json.dump(patterns, f, indent=2)
        except OSError:
            pass

    def record_message(self):
        """Called each time a user message arrives."""
        now = time.time()
        self.last_message_time = now
        dt = datetime.fromtimestamp(now)
        self._hour_histogram[dt.hour] += 1
        self._day_histogram[dt.weekday()] += 1

    def record_session_start(self):
        """Called when a new session begins."""
        self.session_start = time.time()
        self.session_count_today += 1
        self.session_count_week += 1
        self.save()

    def record_session_end(self):
        """Called when session ends."""
        self.last_session_end = time.time()
        self.save()

    def stop(self):
        """Stop background thread."""
        self._stop.set()

    def _background_loop(self):
        """Background update every 60 seconds."""
        while not self._stop.wait(60):
            self.save()

    # ── Computed properties ──────────────────────────────────────────────

    @property
    def current_datetime(self) -> str:
        return datetime.now().isoformat(timespec="seconds")

    @property
    def day_of_week(self) -> str:
        return datetime.now().strftime("%A")

    @property
    def time_of_day_bucket(self) -> str:
        return _time_bucket(datetime.now().hour)

    @property
    def time_since_last_message(self) -> float:
        """Seconds since last user message this session."""
        return time.time() - self.last_message_time

    @property
    def time_since_last_session(self) -> float | None:
        """Seconds since last session ended. None if first session."""
        if self.last_session_end is None:
            return None
        return self.session_start - self.last_session_end

    @property
    def session_duration(self) -> float:
        """Minutes since session started."""
        return (time.time() - self.session_start) / 60

    @property
    def is_unusual_time(self) -> bool:
        """True if current time is outside typical usage pattern."""
        hour = datetime.now().hour
        if not self._hour_histogram:
            return False
        total = sum(self._hour_histogram.values())
        if total < 10:
            return False  # not enough data
        hour_pct = self._hour_histogram.get(hour, 0) / total
        return hour_pct < 0.02  # less than 2% of messages at this hour

    @property
    def inferred_user_state(self) -> str:
        """Rough inference: working/resting/probably_asleep."""
        hour = datetime.now().hour
        bucket = _time_bucket(hour)
        if bucket == "late_night":
            return "probably_asleep"
        if bucket in ("morning", "afternoon"):
            return "working"
        return "resting"

    @property
    def typical_hours(self) -> list[int]:
        """Return the hours when user is most active (top 5)."""
        if not self._hour_histogram:
            return []
        sorted_hours = sorted(self._hour_histogram.items(),
                              key=lambda x: x[1], reverse=True)
        return [h for h, _ in sorted_hours[:5]]

    @property
    def recent_session_frequency(self) -> float:
        """Approximate sessions per day over available data."""
        total = sum(self._day_histogram.values())
        if total == 0:
            return 0
        # Rough estimate based on day spread
        days_active = len([d for d, c in self._day_histogram.items() if c > 0])
        return total / max(days_active, 1)

    # ── Context strings ──────────────────────────────────────────────

    def context_string(self) -> str:
        """Compressed string for interoceptive block."""
        parts = [
            f"{self.day_of_week} {self.time_of_day_bucket}",
            f"session: {self.session_duration:.0f}min",
        ]

        gap = self.time_since_last_session
        if gap is not None:
            if gap < 3600:
                parts.append(f"gap: {gap/60:.0f}min")
            elif gap < 86400:
                parts.append(f"gap: {gap/3600:.1f}h")
            else:
                parts.append(f"gap: {gap/86400:.1f}d")

        if self.is_unusual_time:
            parts.append("unusual_time")

        idle = self.time_since_last_message
        if idle > 120:
            parts.append(f"idle: {idle/60:.0f}min")

        return " | ".join(parts)

    def is_continuation(self) -> bool:
        """True if current session likely continues prior work."""
        gap = self.time_since_last_session
        if gap is None:
            return False
        return gap < 1800  # less than 30 minutes

    def get_pattern_summary(self) -> str:
        """Full description of temporal patterns for tools."""
        lines = [
            f"Current: {self.current_datetime} ({self.day_of_week}, {self.time_of_day_bucket})",
            f"Session duration: {self.session_duration:.0f} minutes",
            f"Sessions today: {self.session_count_today}",
            f"Sessions this week: {self.session_count_week}",
            f"Inferred state: {self.inferred_user_state}",
        ]

        gap = self.time_since_last_session
        if gap is not None:
            if gap < 3600:
                lines.append(f"Time since last session: {gap/60:.0f} minutes")
            elif gap < 86400:
                lines.append(f"Time since last session: {gap/3600:.1f} hours")
            else:
                lines.append(f"Time since last session: {gap/86400:.1f} days")

        if self.typical_hours:
            lines.append(f"Typical active hours: {', '.join(str(h) for h in self.typical_hours)}")

        if self.is_unusual_time:
            lines.append("NOTE: Current time is unusual for this user")

        return "\n".join(lines)

    def check_proactive_triggers(self) -> list[dict]:
        """Return conditions that might warrant proactive action."""
        triggers = []

        # Long gap since last session
        gap = self.time_since_last_session
        if gap is not None and gap > 3 * 86400:
            triggers.append({
                "type": "long_gap",
                "detail": f"User hasn't messaged in {gap/86400:.1f} days",
                "priority": "low",
            })

        # Unusual time
        if self.is_unusual_time:
            triggers.append({
                "type": "unusual_time",
                "detail": f"User active at unusual hour ({datetime.now().hour}:00)",
                "priority": "info",
            })

        return triggers
