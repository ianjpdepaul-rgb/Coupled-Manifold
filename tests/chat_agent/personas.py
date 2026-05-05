from __future__ import annotations

"""
Persona functions — each drives a GracefulAgent through a specific usage pattern.

Every persona function takes an agent (already connected) and returns a dict
with structured results suitable for JSONL logging.  The agent's turn_log
captures all individual turns.

Personas are pure black-box: they exercise the HTTP API surface only.
"""

import time

from .agent import GracefulAgent


def _result(agent: GracefulAgent, persona: str, notes: str = "") -> dict:
    """Standard result envelope."""
    return {
        "persona": persona,
        "notes": notes,
        "sid": agent.active_sid,
        **agent.to_jsonl_record(),
    }


# ── 1. normal_user ────────────────────────────────────────

def normal_user(agent: GracefulAgent) -> dict:
    """Casual 5-turn conversation — the golden path."""
    agent.new_session()
    messages = [
        "Hi there, how are you?",
        "What's the weather like in a haiku?",
        "Tell me something interesting about octopuses.",
        "Can you write a short limerick about coffee?",
        "Thanks, that was fun!",
    ]
    for msg in messages:
        rec = agent.send_message(msg)
        if rec.error:
            break
    return _result(agent, "normal_user")


# ── 2. recursive_continuer ────────────────────────────────

def recursive_continuer(agent: GracefulAgent) -> dict:
    """Sends 'continue' repeatedly — tests the /continue path and loop stability."""
    agent.new_session()
    agent.send_message("Write a short story about a detective in three parts.")
    for _ in range(4):
        agent.send_message("continue")
    return _result(agent, "recursive_continuer")


# ── 3. slash_command_spammer ──────────────────────────────

def slash_command_spammer(agent: GracefulAgent) -> dict:
    """Fires a battery of slash commands — verifies they don't crash."""
    agent.new_session()
    # Warm up with a real message so there's session content
    agent.send_message("Hello, I'm testing slash commands.")

    commands = [
        "/who",
        "/stats",
        "/version",
        "/mood",
        "/help",
        "/recap",
        "/summarize",
        "/check",
        "/clear",
    ]
    errors = []
    for cmd in commands:
        rec = agent.send_command(cmd)
        if rec.error:
            errors.append(f"{cmd}: {rec.error}")

    return _result(agent, "slash_command_spammer",
                   notes=f"{len(errors)} errors" if errors else "clean")


# ── 4. mode_switcher ─────────────────────────────────────

def mode_switcher(agent: GracefulAgent) -> dict:
    """Toggles think_mode and temperature, then sends messages in each config."""
    agent.new_session()

    # Default state
    agent.send_message("What is 2+2?")

    # Enable think mode
    agent.update_settings(think_mode=True)
    agent.send_message("Explain why the sky is blue, step by step.")

    # High temperature
    agent.update_settings(temp=1.5)
    agent.send_message("Write something creative and weird.")

    # Back to defaults
    agent.update_settings(think_mode=False, temp=0.7)
    agent.send_message("Now be precise: what is the capital of France?")

    return _result(agent, "mode_switcher")


# ── 5. long_session ──────────────────────────────────────

def long_session(agent: GracefulAgent) -> dict:
    """15-turn session — tests context growth, auto-summarize, trace evolution."""
    agent.new_session()
    topics = [
        "Tell me about the history of jazz music.",
        "Who were the key figures in bebop?",
        "How did cool jazz differ from bebop?",
        "What role did Miles Davis play?",
        "Tell me about John Coltrane.",
        "How did free jazz emerge?",
        "What's the connection between jazz and the civil rights movement?",
        "How did jazz influence rock and roll?",
        "What happened to jazz in the 1980s?",
        "Tell me about the jazz revival in the 1990s.",
        "Who are the most important contemporary jazz musicians?",
        "How has jazz influenced hip hop?",
        "What's the future of jazz?",
        "Summarize everything we discussed about jazz.",
        "What was the most surprising thing you told me?",
    ]
    for msg in topics:
        rec = agent.send_message(msg)
        if rec.error:
            break
    return _result(agent, "long_session")


# ── 6. edge_caser ────────────────────────────────────────

def edge_caser(agent: GracefulAgent) -> dict:
    """Boundary inputs: empty-ish, Unicode, long, special chars."""
    agent.new_session()

    cases = [
        " ",                         # whitespace-only (should get "empty" error)
        "a",                         # minimal
        "Hello! 🎵🌍🔥",            # emoji
        "日本語のテスト",              # CJK
        "x" * 5000,                  # long message
        "<script>alert(1)</script>",  # XSS attempt (should be harmless)
        "```python\nprint('hi')\n```",  # code block
        "/notacommand",              # unknown slash command
        "\\n\\n\\n",                 # escaped newlines
    ]
    for msg in cases:
        agent.send_message(msg)
    return _result(agent, "edge_caser")


# ── 7. search_spammer ────────────────────────────────────

def search_spammer(agent: GracefulAgent) -> dict:
    """Fires /search and /find commands — tests search integration."""
    agent.new_session()
    agent.send_message("Hi, I want to test search features.")
    queries = [
        "/find quantum computing",
        "/find neural networks",
        "/search latest AI news",
    ]
    for q in queries:
        agent.send_command(q)
    return _result(agent, "search_spammer")


# ── 8. session_persistence ────────────────────────────────

def session_persistence(agent: GracefulAgent) -> dict:
    """Create session, send messages, switch away, switch back — verify history survives."""
    agent.new_session()
    sid1 = agent.active_sid

    agent.send_message("Remember this: the password is 'swordfish'.")
    agent.send_message("What did I just tell you to remember?")
    turn_count_before = len(agent.turn_log)

    # Create second session
    agent.new_session()
    sid2 = agent.active_sid
    agent.send_message("This is a different session.")

    # Switch back to first session
    try:
        data = agent.load_session(sid1)
        history_restored = len(data.get("history", []))
    except Exception as e:
        history_restored = 0
        agent.errors.append(f"load_session failed: {e}")

    return _result(agent, "session_persistence",
                   notes=f"sid1={sid1}, sid2={sid2}, "
                         f"turns_before_switch={turn_count_before}, "
                         f"history_restored={history_restored}")


# ── 9. fresh_session_continue ─────────────────────────────

def fresh_session_continue(agent: GracefulAgent) -> dict:
    """New session → single message → re-init → verify state."""
    agent.new_session()
    agent.send_message("Testing fresh session persistence.")

    # Simulate a page refresh by re-initing
    init_data = agent.init_session()
    history_len = len(init_data.get("history", []))
    sid_match = init_data.get("active_sid") == agent.active_sid

    return _result(agent, "fresh_session_continue",
                   notes=f"history_len_after_reinit={history_len}, "
                         f"sid_match={sid_match}")


# ── 10. diversity_stress ──────────────────────────────────

def diversity_stress(agent: GracefulAgent) -> dict:
    """Sends the same message 5 times — tests diversity/repetition detection."""
    agent.new_session()
    for i in range(5):
        agent.send_message("Tell me a joke.")
    return _result(agent, "diversity_stress")


# ── 11. identity_roundtrip ────────────────────────────────

def identity_roundtrip(agent: GracefulAgent) -> dict:
    """/iam → /who → /forget — tests identity model CRUD."""
    agent.new_session()
    agent.send_command("/iam I am a test robot from the future")
    agent.send_command("/who")
    agent.send_command("/forget test robot")
    agent.send_command("/who")
    return _result(agent, "identity_roundtrip")


# ── Registry ──────────────────────────────────────────────

ALL_PERSONAS: dict[str, callable] = {
    "normal_user":           normal_user,
    "recursive_continuer":   recursive_continuer,
    "slash_command_spammer": slash_command_spammer,
    "mode_switcher":         mode_switcher,
    "long_session":          long_session,
    "edge_caser":            edge_caser,
    "search_spammer":        search_spammer,
    "session_persistence":   session_persistence,
    "fresh_session_continue": fresh_session_continue,
    "diversity_stress":      diversity_stress,
    "identity_roundtrip":    identity_roundtrip,
}
