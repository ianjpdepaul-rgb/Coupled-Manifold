"""Modification tracker — reversible self-modification infrastructure (Stage 2)."""

import json
import os
import time
from typing import Any


class ModificationTracker:
    """Tracks model-initiated config changes for reversibility."""

    def __init__(self, data_dir: str):
        self._log_path = os.path.join(data_dir, "logs", "modifications.jsonl")
        self._session_mods: list[dict] = []
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)

    def log_modification(self, tool_name: str, before_state: Any,
                         after_state: Any, reason: str, turn_id: int,
                         *, scope: str = "session") -> dict:
        """Record a modification. Returns the entry for immediate use."""
        entry = {
            "ts": time.time(),
            "turn_id": turn_id,
            "tool": tool_name,
            "before": before_state,
            "after": after_state,
            "reason": reason,
            "scope": scope,
            "reverted": False,
        }
        self._session_mods.append(entry)
        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            pass
        return entry

    def undo_last(self) -> dict | None:
        """Revert the most recent unreverted modification. Returns it or None."""
        for mod in reversed(self._session_mods):
            if not mod["reverted"]:
                mod["reverted"] = True
                self._persist_revert(mod)
                return mod
        return None

    def undo_session(self) -> list[dict]:
        """Revert all session modifications (newest first). Returns list of reverted."""
        reverted = []
        for mod in reversed(self._session_mods):
            if not mod["reverted"]:
                mod["reverted"] = True
                self._persist_revert(mod)
                reverted.append(mod)
        return reverted

    def list_modifications(self, scope: str = "session") -> list[dict]:
        """Return modifications. 'session' = current session, 'all' = from disk."""
        if scope == "session":
            return [m for m in self._session_mods if not m["reverted"]]
        # Read all from disk
        if not os.path.isfile(self._log_path):
            return []
        mods = []
        try:
            with open(self._log_path) as f:
                for line in f:
                    try:
                        mods.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass
        return mods

    def pending_count(self) -> int:
        """Number of active (unreverted) modifications in this session."""
        return sum(1 for m in self._session_mods if not m["reverted"])

    def _persist_revert(self, mod: dict):
        """Append a revert record to the log."""
        revert_entry = {
            "ts": time.time(),
            "turn_id": mod["turn_id"],
            "tool": mod["tool"],
            "action": "revert",
            "reverted_to": mod["before"],
        }
        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(revert_entry, default=str) + "\n")
        except Exception:
            pass
