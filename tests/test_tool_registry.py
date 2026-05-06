"""Tests for graceful.tool_registry — Agentic Stage 1."""

import json
import os
import tempfile

import pytest

from graceful.tool_registry import ToolRegistry, TOOL_PENDING, save_permission_overrides
from graceful.tools import audit_recent, take_note, check_disagreement, calculate


class TestToolRegistry:
    """Core registry behavior."""

    def _make_registry(self, tmp_path=None):
        return ToolRegistry(log_dir=tmp_path)

    def test_register_and_dispatch(self):
        reg = self._make_registry()
        reg.register("echo", lambda arg: (f"echo: {arg}", ""),
                     description="echoes input")
        text, html = reg.dispatch("echo", "hello")
        assert text == "echo: hello"
        assert html == ""

    def test_unknown_tool(self):
        reg = self._make_registry()
        text, _ = reg.dispatch("nonexistent", "arg")
        assert "Unknown tool" in text

    def test_handler_exception_caught(self):
        reg = self._make_registry()
        def bad_handler(arg):
            raise ValueError("boom")
        reg.register("bad", bad_handler, description="always fails")
        text, _ = reg.dispatch("bad", "x")
        assert "Tool error" in text
        assert "boom" in text

    def test_names_property(self):
        reg = self._make_registry()
        reg.register("a", lambda x: ("", ""), description="tool a")
        reg.register("b", lambda x: ("", ""), description="tool b")
        assert reg.names == ["a", "b"]

    def test_tool_prompt_contains_all_tools(self):
        reg = self._make_registry()
        reg.register("search", lambda x: ("", ""), description="web search",
                     usage_example="query")
        reg.register("run", lambda x: ("", ""), description="run python",
                     usage_example="code")
        prompt = reg.tool_prompt()
        assert "<tool:search>query</tool:search>" in prompt
        assert "<tool:run>code</tool:run>" in prompt
        assert "web search" in prompt
        assert "run python" in prompt

    def test_logging_to_jsonl(self):
        with tempfile.TemporaryDirectory() as td:
            reg = ToolRegistry(log_dir=td)
            reg.register("echo", lambda arg: (f"ok: {arg}", ""),
                         description="echo")
            reg.dispatch("echo", "test_arg", turn_id=7)

            log_path = os.path.join(td, "tool_calls.jsonl")
            assert os.path.isfile(log_path)
            with open(log_path) as f:
                entry = json.loads(f.readline())
            assert entry["tool"] == "echo"
            assert entry["arg"] == "test_arg"
            assert entry["turn"] == 7
            assert "elapsed_s" in entry

    def test_no_logging_when_no_dir(self):
        reg = ToolRegistry(log_dir=None)
        reg.register("echo", lambda arg: ("ok", ""), description="echo")
        # Should not raise
        reg.dispatch("echo", "x")


class TestPermissionLevels:
    """Verify permission gating works correctly."""

    def test_auto_executes_immediately(self):
        reg = ToolRegistry()
        reg.register("echo", lambda arg: (f"ok: {arg}", ""),
                     description="echo", permission="auto")
        text, _ = reg.dispatch("echo", "hi")
        assert text == "ok: hi"

    def test_prompt_returns_pending_without_confirmation(self):
        reg = ToolRegistry()
        reg.register("risky", lambda arg: ("executed", ""),
                     description="risky tool", permission="prompt")
        text, html = reg.dispatch("risky", "arg")
        assert text == TOOL_PENDING
        # html contains payload with tool info
        payload = json.loads(html)
        assert payload["tool"] == "risky"
        assert payload["arg"] == "arg"

    def test_prompt_executes_with_confirmation(self):
        reg = ToolRegistry()
        reg.register("risky", lambda arg: ("executed", ""),
                     description="risky tool", permission="prompt")
        text, _ = reg.dispatch("risky", "arg", user_confirmed=True)
        assert text == "executed"

    def test_prompt_cancelled_by_user(self):
        reg = ToolRegistry()
        reg.register("risky", lambda arg: ("executed", ""),
                     description="risky tool", permission="prompt")
        text, _ = reg.dispatch("risky", "arg", user_confirmed=False)
        assert "cancelled" in text

    def test_explicit_refuses_without_user_request(self):
        reg = ToolRegistry()
        reg.register("danger", lambda arg: ("done", ""),
                     description="dangerous", permission="explicit")
        text, _ = reg.dispatch("danger", "arg")
        assert "requires explicit user request" in text

    def test_explicit_executes_with_user_requested(self):
        reg = ToolRegistry()
        reg.register("danger", lambda arg: ("done", ""),
                     description="dangerous", permission="explicit")
        text, _ = reg.dispatch("danger", "arg", user_requested=True)
        assert text == "done"

    def test_explicit_executes_with_user_confirmed(self):
        reg = ToolRegistry()
        reg.register("danger", lambda arg: ("done", ""),
                     description="dangerous", permission="explicit")
        text, _ = reg.dispatch("danger", "arg", user_confirmed=True)
        assert text == "done"

    def test_explicit_tools_hidden_from_prompt(self):
        reg = ToolRegistry()
        reg.register("safe", lambda x: ("", ""), description="safe tool",
                     permission="auto")
        reg.register("hidden", lambda x: ("", ""), description="hidden tool",
                     permission="explicit")
        prompt = reg.tool_prompt()
        assert "safe tool" in prompt
        assert "hidden" not in prompt

    def test_invalid_permission_raises(self):
        reg = ToolRegistry()
        try:
            reg.register("x", lambda a: ("", ""), description="x",
                         permission="invalid")
            assert False, "Should have raised"
        except ValueError as e:
            assert "invalid" in str(e).lower()


class TestPermissionOverrides:
    """User can override default permissions via settings file."""

    def test_override_elevates_permission(self):
        with tempfile.TemporaryDirectory() as td:
            # Save override: make "search" require prompt
            save_permission_overrides({"search": "prompt"}, td)
            reg = ToolRegistry(data_dir=td)
            reg.register("search", lambda a: ("results", ""),
                         description="search", permission="auto")
            # Now search should require confirmation
            text, _ = reg.dispatch("search", "query")
            assert text == TOOL_PENDING

    def test_override_lowers_permission(self):
        with tempfile.TemporaryDirectory() as td:
            # Save override: make "run" auto
            save_permission_overrides({"run": "auto"}, td)
            reg = ToolRegistry(data_dir=td)
            reg.register("run", lambda a: ("executed", ""),
                         description="run code", permission="prompt")
            # Now run should execute freely
            text, _ = reg.dispatch("run", "print(1)")
            assert text == "executed"

    def test_reload_overrides(self):
        with tempfile.TemporaryDirectory() as td:
            reg = ToolRegistry(data_dir=td)
            reg.register("tool", lambda a: ("ok", ""),
                         description="tool", permission="auto")
            # Initially auto
            assert reg.get_permission("tool") == "auto"
            # Write override
            save_permission_overrides({"tool": "prompt"}, td)
            reg.reload_overrides()
            assert reg.get_permission("tool") == "prompt"

    def test_invalid_override_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            # Write invalid override
            path = os.path.join(td, "tool_permissions.json")
            with open(path, "w") as f:
                json.dump({"tool": "bogus_level"}, f)
            reg = ToolRegistry(data_dir=td)
            reg.register("tool", lambda a: ("ok", ""),
                         description="tool", permission="auto")
            # Should use default, not the bogus override
            assert reg.get_permission("tool") == "auto"


class TestLoggingCompleteness:
    """Verify all required fields appear in tool_calls.jsonl."""

    def test_auto_tool_log_fields(self):
        with tempfile.TemporaryDirectory() as td:
            reg = ToolRegistry(log_dir=td)
            reg.register("echo", lambda a: ("ok", ""),
                         description="echo", permission="auto")
            reg.dispatch("echo", "test", turn_id=5)

            log_path = os.path.join(td, "tool_calls.jsonl")
            with open(log_path) as f:
                entry = json.loads(f.readline())
            assert entry["tool"] == "echo"
            assert entry["arg"] == "test"
            assert entry["turn"] == 5
            assert entry["permission"] == "auto"
            assert entry["user_confirmed"] is None  # auto tools don't need confirmation
            assert entry["error"] is None
            assert "elapsed_s" in entry
            assert "ts" in entry
            assert "result" in entry

    def test_prompt_pending_log_fields(self):
        with tempfile.TemporaryDirectory() as td:
            reg = ToolRegistry(log_dir=td)
            reg.register("risky", lambda a: ("ok", ""),
                         description="risky", permission="prompt")
            reg.dispatch("risky", "arg", turn_id=3)

            log_path = os.path.join(td, "tool_calls.jsonl")
            with open(log_path) as f:
                entry = json.loads(f.readline())
            assert entry["permission"] == "prompt"
            assert entry["user_confirmed"] is None  # pending state
            assert entry["result"] == TOOL_PENDING

    def test_prompt_confirmed_log_fields(self):
        with tempfile.TemporaryDirectory() as td:
            reg = ToolRegistry(log_dir=td)
            reg.register("risky", lambda a: ("ok", ""),
                         description="risky", permission="prompt")
            reg.dispatch("risky", "arg", turn_id=3, user_confirmed=True)

            log_path = os.path.join(td, "tool_calls.jsonl")
            with open(log_path) as f:
                entry = json.loads(f.readline())
            assert entry["permission"] == "prompt"
            assert entry["user_confirmed"] is True
            assert entry["result"] == "ok"

    def test_error_logged(self):
        with tempfile.TemporaryDirectory() as td:
            reg = ToolRegistry(log_dir=td)
            reg.register("bad", lambda a: (_ for _ in ()).throw(RuntimeError("fail")),
                         description="bad", permission="auto")
            reg.dispatch("bad", "x", turn_id=1)

            log_path = os.path.join(td, "tool_calls.jsonl")
            with open(log_path) as f:
                entry = json.loads(f.readline())
            assert entry["error"] is not None
            assert "fail" in entry["error"]

    def test_permission_denied_logged(self):
        with tempfile.TemporaryDirectory() as td:
            reg = ToolRegistry(log_dir=td)
            reg.register("locked", lambda a: ("ok", ""),
                         description="locked", permission="explicit")
            reg.dispatch("locked", "x", turn_id=2)

            log_path = os.path.join(td, "tool_calls.jsonl")
            with open(log_path) as f:
                entry = json.loads(f.readline())
            assert entry["error"] == "permission_denied"
            assert entry["user_confirmed"] is False


class TestInfoAndMetadata:
    """Test the info() method for UI display."""

    def test_info_returns_all_tools(self):
        reg = ToolRegistry()
        reg.register("a", lambda x: ("", ""), description="tool a",
                     permission="auto")
        reg.register("b", lambda x: ("", ""), description="tool b",
                     permission="prompt")
        info = reg.info()
        assert len(info) == 2
        assert info[0]["name"] == "a"
        assert info[0]["permission"] == "auto"
        assert info[1]["name"] == "b"
        assert info[1]["permission"] == "prompt"

    def test_info_reflects_override(self):
        with tempfile.TemporaryDirectory() as td:
            save_permission_overrides({"a": "explicit"}, td)
            reg = ToolRegistry(data_dir=td)
            reg.register("a", lambda x: ("", ""), description="tool a",
                         permission="auto")
            info = reg.info()
            assert info[0]["permission"] == "explicit"
            assert info[0]["default_permission"] == "auto"


class TestNewTools:
    """Test the agentic tool implementations."""

    def test_take_note_creates_file(self):
        orig = os.environ.get("GRACEFUL_DATA")
        with tempfile.TemporaryDirectory() as td:
            os.environ["GRACEFUL_DATA"] = td
            try:
                text, _ = take_note("user likes brevity")
                assert text == "Note saved."
                notes_path = os.path.join(td, "logs", "model_notes.jsonl")
                assert os.path.isfile(notes_path)
                with open(notes_path) as f:
                    entry = json.loads(f.readline())
                assert entry["note"] == "user likes brevity"
            finally:
                if orig:
                    os.environ["GRACEFUL_DATA"] = orig
                else:
                    del os.environ["GRACEFUL_DATA"]

    def test_take_note_empty_arg(self):
        text, _ = take_note("")
        assert "No note" in text

    def test_audit_recent_no_log(self):
        orig = os.environ.get("GRACEFUL_DATA")
        with tempfile.TemporaryDirectory() as td:
            os.environ["GRACEFUL_DATA"] = td
            try:
                text, _ = audit_recent("5")
                assert "No tool call log found" in text
            finally:
                if orig:
                    os.environ["GRACEFUL_DATA"] = orig
                else:
                    del os.environ["GRACEFUL_DATA"]

    def test_audit_recent_with_log(self):
        orig = os.environ.get("GRACEFUL_DATA")
        with tempfile.TemporaryDirectory() as td:
            os.environ["GRACEFUL_DATA"] = td
            log_dir = os.path.join(td, "logs")
            os.makedirs(log_dir)
            log_path = os.path.join(log_dir, "tool_calls.jsonl")
            entry = {"turn": 3, "tool": "search", "arg": "test query",
                     "result": "some result", "elapsed_s": 0.5}
            with open(log_path, "w") as f:
                f.write(json.dumps(entry) + "\n")
            try:
                text, _ = audit_recent("5")
                assert "search" in text
                assert "test query" in text
            finally:
                if orig:
                    os.environ["GRACEFUL_DATA"] = orig
                else:
                    del os.environ["GRACEFUL_DATA"]

    def test_check_disagreement_no_file(self):
        orig = os.environ.get("GRACEFUL_DATA")
        with tempfile.TemporaryDirectory() as td:
            os.environ["GRACEFUL_DATA"] = td
            try:
                text, _ = check_disagreement("")
                assert "No framework confidence data" in text
            finally:
                if orig:
                    os.environ["GRACEFUL_DATA"] = orig
                else:
                    del os.environ["GRACEFUL_DATA"]

    def test_check_disagreement_with_file(self):
        orig = os.environ.get("GRACEFUL_DATA")
        with tempfile.TemporaryDirectory() as td:
            os.environ["GRACEFUL_DATA"] = td
            fc_path = os.path.join(td, "framework_confidence.json")
            fc_data = {"confidence": 0.72, "provisional": False,
                       "disagreements": ["drift_detected"],
                       "channels": {"trace": "healthy"}}
            with open(fc_path, "w") as f:
                json.dump(fc_data, f)
            try:
                text, _ = check_disagreement("")
                assert "0.72" in text
                assert "drift_detected" in text
            finally:
                if orig:
                    os.environ["GRACEFUL_DATA"] = orig
                else:
                    del os.environ["GRACEFUL_DATA"]

    @pytest.mark.skipif(
        not bool(__import__("importlib").util.find_spec("sympy")),
        reason="sympy not installed")
    def test_calculate_basic(self):
        text, _ = calculate("factorial(10)")
        assert "3628800" in text

    def test_calculate_empty(self):
        text, _ = calculate("")
        assert "No expression" in text


class TestToolDiscretion:
    """Verify discretion framing and per-turn call cap."""

    def test_tool_prompt_has_discretion_framing(self):
        reg = ToolRegistry()
        reg.register("echo", lambda x: ("", ""), description="echo",
                     use_when="user asks to echo", skip_when="user is greeting")
        prompt = reg.tool_prompt()
        assert "default is to NOT use them" in prompt
        assert "warranted ONLY when" in prompt
        assert "NOT warranted when" in prompt

    def test_use_when_appears_in_prompt(self):
        reg = ToolRegistry()
        reg.register("search", lambda x: ("", ""), description="web search",
                     use_when="user asks about current events",
                     skip_when="topic is in training data")
        prompt = reg.tool_prompt()
        assert "USE WHEN: user asks about current events" in prompt
        assert "SKIP WHEN: topic is in training data" in prompt

    def test_empty_use_skip_not_shown(self):
        reg = ToolRegistry()
        reg.register("echo", lambda x: ("", ""), description="echo")
        prompt = reg.tool_prompt()
        assert "USE WHEN:" not in prompt
        assert "SKIP WHEN:" not in prompt

    def test_per_turn_cap_enforced(self):
        reg = ToolRegistry()
        for i in range(5):
            reg.register(f"t{i}", lambda a: ("ok", ""), description=f"tool {i}")
        # First 3 calls succeed
        for i in range(3):
            text, _ = reg.dispatch(f"t{i}", "x", turn_id=42)
            assert text == "ok"
        # 4th call is refused
        text, _ = reg.dispatch("t3", "x", turn_id=42)
        assert "cap reached" in text.lower()

    def test_cap_resets_on_new_turn(self):
        reg = ToolRegistry()
        for i in range(5):
            reg.register(f"t{i}", lambda a: ("ok", ""), description=f"tool {i}")
        # Exhaust cap on turn 1
        for i in range(3):
            reg.dispatch(f"t{i}", "x", turn_id=1)
        text, _ = reg.dispatch("t3", "x", turn_id=1)
        assert "cap reached" in text.lower()
        # New turn resets
        text, _ = reg.dispatch("t0", "x", turn_id=2)
        assert text == "ok"

    def test_multi_call_warning_logged(self):
        with tempfile.TemporaryDirectory() as td:
            reg = ToolRegistry(log_dir=td)
            reg.register("a", lambda x: ("ok", ""), description="a")
            reg.register("b", lambda x: ("ok", ""), description="b")
            reg.dispatch("a", "x", turn_id=10)
            reg.dispatch("b", "x", turn_id=10)  # 2nd call in turn → warning

            log_path = os.path.join(td, "tool_calls.jsonl")
            with open(log_path) as f:
                entries = [json.loads(line) for line in f]
            warnings = [e for e in entries if e["tool"] == "_warning"]
            assert len(warnings) == 1
            assert "Multiple tool calls" in warnings[0]["result"]

    def test_info_includes_use_skip(self):
        reg = ToolRegistry()
        reg.register("echo", lambda x: ("", ""), description="echo",
                     use_when="when needed", skip_when="when not needed")
        info = reg.info()
        assert info[0]["use_when"] == "when needed"
        assert info[0]["skip_when"] == "when not needed"

    def test_info_defaults_empty_strings(self):
        reg = ToolRegistry()
        reg.register("echo", lambda x: ("", ""), description="echo")
        info = reg.info()
        assert info[0]["use_when"] == ""
        assert info[0]["skip_when"] == ""
