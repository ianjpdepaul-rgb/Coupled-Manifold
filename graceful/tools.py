"""Agentic tools — Stage 1 complete toolset."""

import json
import os
import time

from graceful.python_executor import run_calc as _run_calc


def _data_dir() -> str:
    """Resolve the active data directory."""
    return os.environ.get("GRACEFUL_DATA", "manifold_data")


# ── audit_recent ─────────────────────────────────────────────────────────────
def audit_recent(arg: str) -> tuple[str, str]:
    """Return a summary of recent tool usage from tool_calls.jsonl.

    arg: optional integer N (default 10) — how many recent calls to show.
    """
    n = 10
    if arg.strip().isdigit():
        n = min(int(arg.strip()), 50)

    log_path = os.path.join(_data_dir(), "logs", "tool_calls.jsonl")
    if not os.path.isfile(log_path):
        return ("No tool call log found yet.", "")

    lines = []
    with open(log_path) as f:
        for line in f:
            lines.append(line)
    recent = lines[-n:]

    entries = []
    for line in recent:
        try:
            e = json.loads(line)
            entries.append(
                f"turn {e.get('turn','?')}: {e.get('tool','?')}({e.get('arg','')[:60]}) "
                f"→ {e.get('result','')[:80]} [{e.get('elapsed_s',0):.1f}s]"
            )
        except json.JSONDecodeError:
            continue

    if not entries:
        return ("Log exists but no parseable entries.", "")
    return ("\n".join(entries), "")


# ── read_session ─────────────────────────────────────────────────────────────
def read_session(arg: str) -> tuple[str, str]:
    """Return session metadata — turn count, uptime, active model, thresholds.

    arg: ignored.
    """
    # These will be injected at registration time via closure or pulled from globals.
    # For now, return what we can discover from disk.
    data_dir = _data_dir()
    session_file = os.path.join(data_dir, "session_state.json")
    if os.path.isfile(session_file):
        with open(session_file) as f:
            state = json.load(f)
        summary = (
            f"turns: {state.get('turn_count', '?')}\n"
            f"model: {state.get('active_model', '?')}\n"
            f"thresholds: low={state.get('threshold_low', '?')} high={state.get('threshold_high', '?')}\n"
            f"uptime: {state.get('uptime_minutes', '?')} min"
        )
        return (summary, "")
    return ("Session state file not found — session may not have persisted yet.", "")


# ── take_note ────────────────────────────────────────────────────────────────
def take_note(arg: str) -> tuple[str, str]:
    """Persist a model-authored note for cross-turn memory.

    arg: the note text to save.
    """
    if not arg.strip():
        return ("No note content provided.", "")

    notes_path = os.path.join(_data_dir(), "logs", "model_notes.jsonl")
    os.makedirs(os.path.dirname(notes_path), exist_ok=True)
    entry = {
        "ts": time.time(),
        "note": arg.strip()[:1000],
    }
    with open(notes_path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return ("Note saved.", "")


# ── check_disagreement ───────────────────────────────────────────────────────
def check_disagreement(arg: str) -> tuple[str, str]:
    """Return current framework disagreement signals and confidence.

    arg: ignored. Reads from the live framework confidence state.
    """
    data_dir = _data_dir()
    fc_path = os.path.join(data_dir, "framework_confidence.json")
    if not os.path.isfile(fc_path):
        return ("No framework confidence data on disk.", "")

    with open(fc_path) as f:
        fc = json.load(f)

    conf = fc.get("confidence", "?")
    prov = fc.get("provisional", False)
    disagree = fc.get("disagreements", [])
    channels = fc.get("channels", {})

    lines = [
        f"confidence: {conf}",
        f"provisional: {prov}",
        f"disagreements: {', '.join(disagree) if disagree else 'none'}",
        f"channels: {json.dumps(channels)}",
    ]
    return ("\n".join(lines), "")


# ── calculate ────────────────────────────────────────────────────────────────
def calculate(arg: str) -> tuple[str, str]:
    """Evaluate a math expression symbolically + numerically via SymPy.

    arg: math expression (e.g. "integrate(x**2, x)", "factorial(20)")
    """
    if not arg.strip():
        return ("No expression provided.", "")
    return _run_calc(arg.strip())


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — SELF-MODIFICATION TOOLS
# ══════════════════════════════════════════════════════════════════════════════
# These tools are registered with `explicit` permission by default.
# Each accepts a JSON arg with a required `reason` field.
# The actual state mutation is performed by callbacks injected at registration.
# ══════════════════════════════════════════════════════════════════════════════


def _parse_json_arg(arg: str, required_fields: list[str]) -> tuple[dict | None, str]:
    """Parse JSON arg, validate required fields. Returns (parsed, error_msg)."""
    if not arg.strip():
        return None, "No argument provided."
    try:
        parsed = json.loads(arg)
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"
    if "reason" not in parsed:
        return None, "Missing required 'reason' field."
    for field in required_fields:
        if field not in parsed:
            return None, f"Missing required '{field}' field."
    return parsed, ""


def make_adjust_sampling(get_state, set_state, tracker):
    """Factory: returns adjust_sampling handler with injected dependencies.

    get_state() -> dict with keys: temperature, top_p, top_k
    set_state(temperature, top_p, top_k) -> None
    tracker: ModificationTracker instance
    """
    def adjust_sampling(arg: str) -> tuple[str, str]:
        """Adjust sampling parameters (temperature, top_p, top_k).

        arg: JSON with optional temperature, top_p, top_k, and required reason.
        """
        parsed, err = _parse_json_arg(arg, [])
        if err:
            return (err, "")

        before = get_state()
        temp = float(parsed.get("temperature", before["temperature"]))
        top_p = float(parsed.get("top_p", before["top_p"]))
        top_k = int(parsed.get("top_k", before.get("top_k", 0)))
        reason = parsed["reason"]

        # Clamp to safe ranges
        temp = max(0.01, min(2.0, temp))
        top_p = max(0.01, min(1.0, top_p))
        top_k = max(0, min(200, top_k))

        set_state(temp, top_p, top_k)
        after = {"temperature": temp, "top_p": top_p, "top_k": top_k}
        tracker.log_modification("adjust_sampling", before, after, reason,
                                 turn_id=0, scope="session")
        return (f"Sampling adjusted: t={temp} p={top_p} k={top_k} — {reason}", "")
    return adjust_sampling


def make_pin_context(get_pins, add_pin, tracker):
    """Factory: returns pin_context handler.

    get_pins() -> list[str]
    add_pin(text: str) -> None
    tracker: ModificationTracker instance
    """
    def pin_context(arg: str) -> tuple[str, str]:
        """Pin text to persistent context for this session.

        arg: JSON with 'text' and 'reason'.
        """
        parsed, err = _parse_json_arg(arg, ["text"])
        if err:
            return (err, "")

        text = parsed["text"][:500]  # cap length
        reason = parsed["reason"]
        before = list(get_pins())

        add_pin(text)
        after = list(get_pins())
        tracker.log_modification("pin_context", before, after, reason,
                                 turn_id=0, scope="session")
        return (f"Pinned: \"{text[:80]}{'…' if len(text) > 80 else ''}\" — {reason}", "")
    return pin_context


def make_update_system_prompt(get_additions, add_addition, tracker):
    """Factory: returns update_system_prompt handler.

    get_additions() -> list[str]
    add_addition(text: str) -> None
    tracker: ModificationTracker instance
    """
    def update_system_prompt(arg: str) -> tuple[str, str]:
        """Append to system prompt for current session.

        arg: JSON with 'addition' and 'reason'.
        """
        parsed, err = _parse_json_arg(arg, ["addition"])
        if err:
            return (err, "")

        addition = parsed["addition"][:300]
        reason = parsed["reason"]
        before = list(get_additions())

        add_addition(addition)
        after = list(get_additions())
        tracker.log_modification("update_system_prompt", before, after, reason,
                                 turn_id=0, scope="session")
        return (f"System prompt updated: +\"{addition[:60]}…\" — {reason}", "")
    return update_system_prompt


def make_mark_disagreement(get_disagreements, add_disagreement, tracker):
    """Factory: returns mark_disagreement handler.

    get_disagreements() -> list[dict]
    add_disagreement(entry: dict) -> None
    tracker: ModificationTracker instance
    """
    def mark_disagreement(arg: str) -> tuple[str, str]:
        """Record a topic where model holds a different position than user.

        arg: JSON with 'topic', 'user_position', 'prior_position', 'reason'.
        """
        parsed, err = _parse_json_arg(arg, ["topic", "user_position", "prior_position"])
        if err:
            return (err, "")

        entry = {
            "topic": parsed["topic"],
            "user_position": parsed["user_position"],
            "prior_position": parsed["prior_position"],
            "reason": parsed["reason"],
            "ts": time.time(),
        }
        before = list(get_disagreements())
        add_disagreement(entry)
        after = list(get_disagreements())

        tracker.log_modification("mark_disagreement", before, after,
                                 parsed["reason"], turn_id=0, scope="persistent")
        return (f"Disagreement noted: {parsed['topic']} — {parsed['reason']}", "")
    return mark_disagreement


def make_save_to_long_term(save_fn, tracker):
    """Factory: returns save_to_long_term handler.

    save_fn(text: str, category: str) -> None
    tracker: ModificationTracker instance
    """
    def save_to_long_term(arg: str) -> tuple[str, str]:
        """Promote information to persistent long-term memory.

        arg: JSON with 'text', 'category', 'reason'.
        """
        parsed, err = _parse_json_arg(arg, ["text", "category"])
        if err:
            return (err, "")

        text = parsed["text"][:1000]
        category = parsed["category"][:50]
        reason = parsed["reason"]

        save_fn(text, category)
        tracker.log_modification("save_to_long_term",
                                 None, {"text": text, "category": category},
                                 reason, turn_id=0, scope="persistent")
        return (f"Saved to long-term ({category}): \"{text[:60]}…\" — {reason}", "")
    return save_to_long_term
