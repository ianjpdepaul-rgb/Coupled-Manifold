"""Proactive messaging — Phase 6 (E2-E7).

Background process that checks triggers and can initiate messages
when conditions warrant. Default OFF — user opts in.
"""

import json
import os
import subprocess
import threading
import time


# Quiet hours default: 11pm-8am
DEFAULT_QUIET_START = 23
DEFAULT_QUIET_END = 8
DEFAULT_CHECK_INTERVAL = 900  # 15 minutes


class ProactiveMessenger:
    """Allows model to initiate messages when conditions warrant.

    Background process runs every N minutes. Checks proactive triggers.
    When triggers fire, logs them for model decision on next invocation.
    Delivers notifications via macOS osascript.
    """

    def __init__(self, data_dir: str, temporal=None):
        self._data_dir = data_dir
        self._temporal = temporal
        self._config_path = os.path.join(data_dir, "proactive_config.json")
        self._log_path = os.path.join(data_dir, "logs", "proactive_messages.jsonl")
        self._pending_path = os.path.join(data_dir, "proactive_pending.json")
        self._dismissals_path = os.path.join(data_dir, "proactive_dismissals.json")

        # Configuration
        self.enabled: bool = False
        self.check_interval: int = DEFAULT_CHECK_INTERVAL
        self.quiet_start: int = DEFAULT_QUIET_START
        self.quiet_end: int = DEFAULT_QUIET_END
        self.trigger_types_enabled: dict[str, bool] = {
            "reminder": True,
            "long_gap": True,
            "unusual_time": False,
            "pattern_match": True,
            "open_question": True,
        }

        # Runtime state
        self._pending_messages: list[dict] = []
        self._dismissal_patterns: list[str] = []
        self._engagement_stats = {"sent": 0, "engaged": 0, "dismissed": 0}

        # Load config
        self._load_config()
        self._load_dismissals()
        self._load_pending()

        # Background thread
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        """Start background check thread (only if enabled)."""
        if not self.enabled:
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._background_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop background thread."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def enable(self):
        """Enable proactive messaging and start background thread."""
        self.enabled = True
        self._save_config()
        self.start()

    def disable(self):
        """Disable proactive messaging and stop background thread."""
        self.enabled = False
        self._save_config()
        self.stop()

    # ── Configuration persistence ────────────────────────────────────

    def _load_config(self):
        if not os.path.isfile(self._config_path):
            return
        try:
            with open(self._config_path) as f:
                cfg = json.load(f)
            self.enabled = cfg.get("enabled", False)
            self.check_interval = cfg.get("check_interval", DEFAULT_CHECK_INTERVAL)
            self.quiet_start = cfg.get("quiet_start", DEFAULT_QUIET_START)
            self.quiet_end = cfg.get("quiet_end", DEFAULT_QUIET_END)
            if "trigger_types" in cfg:
                self.trigger_types_enabled.update(cfg["trigger_types"])
            self._engagement_stats = cfg.get("engagement_stats",
                                             {"sent": 0, "engaged": 0, "dismissed": 0})
        except (json.JSONDecodeError, OSError):
            pass

    def _save_config(self):
        os.makedirs(self._data_dir, exist_ok=True)
        cfg = {
            "enabled": self.enabled,
            "check_interval": self.check_interval,
            "quiet_start": self.quiet_start,
            "quiet_end": self.quiet_end,
            "trigger_types": self.trigger_types_enabled,
            "engagement_stats": self._engagement_stats,
        }
        try:
            with open(self._config_path, "w") as f:
                json.dump(cfg, f, indent=2)
        except OSError:
            pass

    def _load_dismissals(self):
        if not os.path.isfile(self._dismissals_path):
            return
        try:
            with open(self._dismissals_path) as f:
                self._dismissal_patterns = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    def _save_dismissals(self):
        os.makedirs(self._data_dir, exist_ok=True)
        try:
            with open(self._dismissals_path, "w") as f:
                json.dump(self._dismissal_patterns, f, indent=2)
        except OSError:
            pass

    def _load_pending(self):
        if not os.path.isfile(self._pending_path):
            return
        try:
            with open(self._pending_path) as f:
                self._pending_messages = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    def _save_pending(self):
        os.makedirs(self._data_dir, exist_ok=True)
        try:
            with open(self._pending_path, "w") as f:
                json.dump(self._pending_messages, f, indent=2)
        except OSError:
            pass

    # ── Core logic ───────────────────────────────────────────────────

    def is_quiet_hours(self) -> bool:
        """Check if current time is within quiet hours."""
        import datetime
        hour = datetime.datetime.now().hour
        if self.quiet_start > self.quiet_end:
            # Wraps midnight: e.g. 23-8
            return hour >= self.quiet_start or hour < self.quiet_end
        return self.quiet_start <= hour < self.quiet_end

    def _is_dismissed(self, topic: str) -> bool:
        """Check if a topic matches a dismissal pattern."""
        topic_lower = topic.lower()
        return any(p.lower() in topic_lower for p in self._dismissal_patterns)

    def add_dismissal(self, pattern: str):
        """User says 'stop messaging me about X'."""
        if pattern not in self._dismissal_patterns:
            self._dismissal_patterns.append(pattern)
            self._save_dismissals()

    def check_triggers(self) -> list[dict]:
        """Check all enabled trigger sources and return fired triggers."""
        triggers = []

        # Check reminders
        if self.trigger_types_enabled.get("reminder", True):
            from graceful.workflow_tools import get_pending_reminders
            for r in get_pending_reminders():
                text = r.get("text", "")
                if not self._is_dismissed(text):
                    triggers.append({
                        "type": "reminder",
                        "text": text,
                        "trigger_context": r.get("trigger_context", ""),
                        "created": r.get("created_human", ""),
                    })

        # Check temporal triggers
        if self._temporal:
            for t in self._temporal.check_proactive_triggers():
                ttype = t.get("type", "")
                if self.trigger_types_enabled.get(ttype, False):
                    if not self._is_dismissed(t.get("detail", "")):
                        triggers.append(t)

        return triggers

    def _background_loop(self):
        """Periodic check loop."""
        while not self._stop.wait(self.check_interval):
            if not self.enabled:
                continue
            if self.is_quiet_hours():
                continue

            triggers = self.check_triggers()
            for trigger in triggers:
                # Queue as pending — model decides on next interaction
                self._queue_trigger(trigger)

    def _queue_trigger(self, trigger: dict):
        """Queue a trigger as a pending proactive message."""
        entry = {
            "trigger": trigger,
            "queued_at": time.time(),
            "queued_human": time.strftime("%Y-%m-%d %H:%M"),
            "status": "pending",  # pending → sent/deferred/dismissed
        }
        self._pending_messages.append(entry)
        self._save_pending()
        self._log("queued", trigger)

    def send_notification(self, text: str, title: str = "Graceful"):
        """Send a macOS notification via osascript."""
        # Escape for AppleScript
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        title_escaped = title.replace("\\", "\\\\").replace('"', '\\"')
        script = (
            f'display notification "{escaped}" '
            f'with title "{title_escaped}" sound name "Glass"'
        )
        try:
            subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, timeout=5,
            )
        except Exception:
            pass

    def record_action(self, trigger_index: int, action: str, reason: str = ""):
        """Record model's decision on a pending trigger.

        action: 'message', 'defer', 'dismiss'
        """
        if trigger_index < 0 or trigger_index >= len(self._pending_messages):
            return

        entry = self._pending_messages[trigger_index]
        entry["status"] = action
        entry["reason"] = reason
        entry["decided_at"] = time.time()

        if action == "message":
            self._engagement_stats["sent"] += 1
        elif action == "dismiss":
            topic = entry.get("trigger", {}).get("text", "")
            if topic:
                self.add_dismissal(topic)

        self._save_pending()
        self._save_config()
        self._log(action, entry.get("trigger", {}), reason)

    def record_user_response(self, engaged: bool):
        """Track whether user engaged with or dismissed a proactive message."""
        if engaged:
            self._engagement_stats["engaged"] += 1
        else:
            self._engagement_stats["dismissed"] += 1
        self._save_config()

        # Auto-increase discretion if dismissal rate is high
        total = self._engagement_stats["sent"]
        if total >= 10:
            dismissed = self._engagement_stats["dismissed"]
            if dismissed / total > 0.7:
                # High dismissal rate — increase check interval
                self.check_interval = min(self.check_interval * 2, 3600)
                self._save_config()

    def get_pending(self) -> list[dict]:
        """Return pending messages for display when user opens app."""
        return [m for m in self._pending_messages if m.get("status") == "pending"]

    def clear_pending(self):
        """Dismiss all pending messages."""
        for m in self._pending_messages:
            if m.get("status") == "pending":
                m["status"] = "dismissed"
                m["reason"] = "user cleared all"
        self._save_pending()

    def get_config_for_ui(self) -> dict:
        """Return config for settings panel display."""
        return {
            "enabled": self.enabled,
            "check_interval_minutes": self.check_interval // 60,
            "quiet_hours": f"{self.quiet_start}:00 - {self.quiet_end}:00",
            "trigger_types": self.trigger_types_enabled,
            "stats": self._engagement_stats,
            "pending_count": len(self.get_pending()),
            "dismissal_patterns": self._dismissal_patterns,
        }

    def _log(self, action: str, trigger: dict, reason: str = ""):
        """Log proactive action."""
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
        entry = {
            "ts": time.time(),
            "action": action,
            "trigger": trigger,
            "reason": reason,
        }
        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except OSError:
            pass

    # ── Decision prompt for model ────────────────────────────────────

    def decision_prompt(self, trigger: dict) -> str:
        """Generate the prompt given to the model for proactive decisions."""
        return (
            f"A proactive trigger has fired:\n"
            f"Type: {trigger.get('type', 'unknown')}\n"
            f"Detail: {trigger.get('detail', trigger.get('text', ''))}\n\n"
            f"Should you message the user about this? "
            f"Most triggers should result in 'defer' or 'dismiss', not 'message'. "
            f"Message only when the user would genuinely benefit from the interruption. "
            f"Consider: is this actually time-sensitive? Could it wait until they message you next? "
            f"If unsure, defer.\n\n"
            f"Reply with action: message|defer|dismiss and reason."
        )
