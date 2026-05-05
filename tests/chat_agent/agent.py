from __future__ import annotations

"""
GracefulAgent — drives a running Graceful instance via HTTP.

Pure black-box: no imports from app.py.  Communicates exclusively through
the public API surface (/api/init, /api/chat, /api/new, /api/settings,
/api/sessions/{id}, /api/export_json, /api/stats_json).

SSE parsing is manual (no sseclient-py dependency).
"""

import json
import re
import time
from dataclasses import dataclass, field

import requests

# ── Medulla status regex ──────────────────────────────────
# Format: t{turn} · {trace} · {mode} · {prompt}↓ {output}↑ · {chunks}c · {archive}a[|{drift}]
_MEDULLA_RE = re.compile(
    r"t(\d+)\s*·\s*([^\s·]+)\s*·\s*(\w+)\s*·\s*"
    r"([^\s]+)↓\s*([^\s]+)↑\s*·\s*(\d+)c\s*·\s*(\d+)a"
    r"(?:\|([0-9.eE\-+]+))?"
)

_RATE_LIMIT_PAUSE = 1.0   # seconds to wait on 429
_INTER_MSG_PAUSE  = 0.6   # seconds between messages (> _MIN_GEN_INTERVAL)


@dataclass
class MedullaState:
    """Parsed medulla status string."""
    turn: int = 0
    trace: str = "?"
    mode: str = "?"
    prompt_tokens: str = "?"
    output_tokens: str = "?"
    corpus_chunks: int = 0
    archive_turns: int = 0
    drift: float | None = None
    raw: str = ""

    @classmethod
    def parse(cls, raw: str) -> "MedullaState":
        if not raw:
            return cls(raw=raw)
        m = _MEDULLA_RE.search(raw)
        if not m:
            return cls(raw=raw)
        drift = float(m.group(8)) if m.group(8) else None
        return cls(
            turn=int(m.group(1)),
            trace=m.group(2),
            mode=m.group(3),
            prompt_tokens=m.group(4),
            output_tokens=m.group(5),
            corpus_chunks=int(m.group(6)),
            archive_turns=int(m.group(7)),
            drift=drift,
            raw=raw,
        )


@dataclass
class TurnRecord:
    """One send_message round-trip."""
    user_message: str
    response: str
    status: MedullaState
    elapsed: float
    error: str | None = None
    events_raw: list = field(default_factory=list)


class GracefulAgent:
    """Drives a Graceful instance through its HTTP API."""

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self.active_sid: str | None = None
        self.history: list[dict] = []
        self.medulla: MedullaState = MedullaState()
        self.errors: list[str] = []
        self.turn_log: list[TurnRecord] = []
        self.settings: dict = {}

    # ── Session lifecycle ─────────────────────────────────

    def init_session(self) -> dict:
        """GET /api/init — load current server state."""
        r = requests.get(f"{self.base_url}/api/init", timeout=30)
        r.raise_for_status()
        d = r.json()
        self.active_sid = d.get("active_sid")
        self.history = d.get("history", [])
        self.settings = {
            k: d.get(k) for k in (
                "think_mode", "temp", "temperature", "online_learning",
                "trace_sync_mode", "show_medulla", "think_budget",
                "user_name", "assistant_name", "system_prompt",
                "vision_tokens", "model",
            ) if d.get(k) is not None
        }
        status_raw = d.get("status", "")
        if status_raw:
            self.medulla = MedullaState.parse(status_raw)
        return d

    def new_session(self) -> dict:
        """POST /api/new — start a fresh session."""
        r = requests.post(f"{self.base_url}/api/new", timeout=15)
        r.raise_for_status()
        d = r.json()
        self.active_sid = d.get("active_sid")
        self.history = d.get("history", [])
        self.medulla = MedullaState()
        self.turn_log = []
        self.errors = []
        return d

    def load_session(self, session_id: str) -> dict:
        """GET /api/sessions/{id} — switch to an existing session."""
        r = requests.get(f"{self.base_url}/api/sessions/{session_id}", timeout=15)
        r.raise_for_status()
        d = r.json()
        self.active_sid = session_id
        self.history = d.get("history", [])
        status_raw = d.get("status", "")
        if status_raw:
            self.medulla = MedullaState.parse(status_raw)
        return d

    # ── Chat ──────────────────────────────────────────────

    def send_message(self, text: str, retries: int = 2) -> TurnRecord:
        """POST /api/chat — send a message and parse the SSE stream.

        Returns a TurnRecord with the full response, parsed medulla state,
        timing, and any error.  Handles 429 rate limiting automatically.
        """
        time.sleep(_INTER_MSG_PAUSE)

        for attempt in range(retries + 1):
            r = requests.post(
                f"{self.base_url}/api/chat",
                json={"message": text},
                stream=True,
                timeout=120,
            )
            if r.status_code == 429:
                if attempt < retries:
                    time.sleep(_RATE_LIMIT_PAUSE)
                    continue
                rec = TurnRecord(
                    user_message=text, response="", status=self.medulla,
                    elapsed=0, error="rate_limited",
                )
                self.errors.append("rate_limited")
                self.turn_log.append(rec)
                return rec
            break

        t0 = time.time()
        response_chunks: list[str] = []
        events_raw: list[dict] = []
        error: str | None = None
        status_raw = ""
        last_history_msg: dict | None = None

        if r.status_code != 200:
            error = f"http_{r.status_code}"
            self.errors.append(error)
            rec = TurnRecord(
                user_message=text, response="", status=self.medulla,
                elapsed=time.time() - t0, error=error,
            )
            self.turn_log.append(rec)
            return rec

        # Parse SSE stream line by line
        for raw_line in r.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data: "):
                continue
            payload = raw_line[6:]  # strip "data: "
            try:
                ev = json.loads(payload)
            except json.JSONDecodeError:
                continue
            events_raw.append(ev)
            t = ev.get("t")
            if t == "d":
                response_chunks.append(ev.get("v", ""))
            elif t == "done":
                status_raw = ev.get("status", "")
                last_history_msg = (ev.get("history") or [None])[-1]
            elif t == "err":
                error = ev.get("v", "unknown error")
                status_raw = ev.get("status", "")

        elapsed = time.time() - t0
        full_response = "".join(response_chunks)

        # Update agent state
        if status_raw:
            self.medulla = MedullaState.parse(status_raw)
        if last_history_msg:
            self.history.append({"role": "user", "content": text})
            self.history.append(last_history_msg)
        if error:
            self.errors.append(error)

        rec = TurnRecord(
            user_message=text,
            response=full_response,
            status=self.medulla,
            elapsed=elapsed,
            error=error,
            events_raw=events_raw,
        )
        self.turn_log.append(rec)
        return rec

    def send_command(self, command: str) -> TurnRecord:
        """Send a slash command (just calls send_message)."""
        if not command.startswith("/"):
            command = "/" + command
        return self.send_message(command)

    # ── Settings ──────────────────────────────────────────

    def get_settings(self) -> dict:
        """GET /api/init and return settings subset."""
        d = self.init_session()
        return self.settings

    def update_settings(self, **kwargs) -> dict:
        """POST /api/settings with arbitrary key-value pairs."""
        r = requests.post(
            f"{self.base_url}/api/settings",
            json=kwargs,
            timeout=10,
        )
        r.raise_for_status()
        self.settings.update(kwargs)
        return r.json()

    # ── Introspection ─────────────────────────────────────

    def get_session_log(self) -> list[TurnRecord]:
        """Return the in-memory turn log for this agent run."""
        return list(self.turn_log)

    def export_session(self) -> list[dict]:
        """GET /api/export_json — return server-side turn-by-turn data."""
        r = requests.get(f"{self.base_url}/api/export_json", timeout=15)
        r.raise_for_status()
        return r.json()

    def get_trace_history(self) -> dict:
        """GET /api/stats_json — return structured trace analytics."""
        r = requests.get(f"{self.base_url}/api/stats_json", timeout=15)
        r.raise_for_status()
        return r.json()

    # ── Helpers ───────────────────────────────────────────

    def wait_ready(self, timeout: float = 60) -> bool:
        """Poll /api/health until the server is up."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                r = requests.get(f"{self.base_url}/api/health", timeout=5)
                if r.status_code == 200:
                    return True
            except requests.ConnectionError:
                pass
            time.sleep(1)
        return False

    def to_jsonl_record(self) -> dict:
        """Serialize this agent run as a single JSONL-compatible dict."""
        return {
            "active_sid": self.active_sid,
            "turns": len(self.turn_log),
            "errors": list(self.errors),
            "medulla_final": {
                "turn": self.medulla.turn,
                "trace": self.medulla.trace,
                "mode": self.medulla.mode,
                "drift": self.medulla.drift,
                "raw": self.medulla.raw,
            },
            "turn_log": [
                {
                    "user": tr.user_message,
                    "response_len": len(tr.response),
                    "response_preview": tr.response[:200],
                    "elapsed": round(tr.elapsed, 3),
                    "error": tr.error,
                    "medulla": tr.status.raw,
                }
                for tr in self.turn_log
            ],
        }
