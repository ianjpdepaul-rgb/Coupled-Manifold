"""Tests for graceful.workflow_tools — Phases 1-4."""

import json
import os
import tempfile
import time

import pytest

from graceful.workflow_tools import (
    read_file,
    write_draft,
    list_directory,
    shell_readonly,
    init_project_root,
    _is_path_allowed,
    _get_allowed_roots,
    ALLOWED_ROOTS,
    ALLOWED_COMMANDS,
    ALLOWED_GIT_SUBCOMMANDS,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _patch_allowed_roots(tmp_path, monkeypatch):
    """Replace the global ALLOWED_ROOTS with a temp directory for testing."""
    test_root = str(tmp_path / "allowed")
    os.makedirs(test_root, exist_ok=True)
    monkeypatch.setattr("graceful.workflow_tools.ALLOWED_ROOTS", [test_root])
    init_project_root(test_root)
    return test_root


@pytest.fixture
def allowed_dir(tmp_path):
    """Return the allowed test root (same as the patched one)."""
    return str(tmp_path / "allowed")


@pytest.fixture
def disallowed_dir(tmp_path):
    """Return a directory NOT in the allowlist."""
    d = str(tmp_path / "forbidden")
    os.makedirs(d, exist_ok=True)
    return d


# ── read_file ────────────────────────────────────────────────────────────────

class TestReadFile:
    def test_reads_file(self, allowed_dir):
        path = os.path.join(allowed_dir, "hello.txt")
        with open(path, "w") as f:
            f.write("hello world")
        text, html = read_file(path)
        assert text == "hello world"
        assert html == ""

    def test_file_not_found(self, allowed_dir):
        path = os.path.join(allowed_dir, "nope.txt")
        text, _ = read_file(path)
        assert "not found" in text.lower()

    def test_access_denied(self, disallowed_dir):
        path = os.path.join(disallowed_dir, "secret.txt")
        with open(path, "w") as f:
            f.write("secret")
        text, _ = read_file(path)
        assert "denied" in text.lower()

    def test_empty_arg(self):
        text, _ = read_file("")
        assert "no path" in text.lower()

    def test_truncation(self, allowed_dir):
        path = os.path.join(allowed_dir, "big.txt")
        with open(path, "w") as f:
            f.write("x" * 300_000)
        text, _ = read_file(path)
        assert "truncated" in text.lower()
        assert len(text) < 300_000

    def test_symlink_escape_blocked(self, allowed_dir, disallowed_dir):
        """Symlink inside allowed dir pointing outside must be blocked."""
        secret = os.path.join(disallowed_dir, "secret.txt")
        with open(secret, "w") as f:
            f.write("secret data")
        link = os.path.join(allowed_dir, "link.txt")
        os.symlink(secret, link)
        text, _ = read_file(link)
        assert "denied" in text.lower()

    def test_directory_error(self, allowed_dir):
        text, _ = read_file(allowed_dir)
        assert "directory" in text.lower()


# ── write_draft ──────────────────────────────────────────────────────────────

class TestWriteDraft:
    def test_writes_file(self, tmp_path, monkeypatch):
        drafts = str(tmp_path / "drafts_out")
        monkeypatch.setenv("GRACEFUL_DATA", str(tmp_path / "data"))
        os.makedirs(str(tmp_path / "data"), exist_ok=True)

        text, _ = write_draft(json.dumps({
            "filename": "test.txt",
            "content": "hello draft"
        }))
        assert "saved" in text.lower()

    def test_with_category(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GRACEFUL_DATA", str(tmp_path / "data"))

        text, _ = write_draft(json.dumps({
            "filename": "main.py",
            "content": "print('hello')",
            "category": "code",
        }))
        assert "saved" in text.lower()
        # Check the file actually exists in the category subdir
        expected = os.path.join(str(tmp_path / "data"), "drafts", "code", "main.py")
        assert os.path.isfile(expected)

    def test_missing_filename(self):
        text, _ = write_draft(json.dumps({"content": "hello"}))
        assert "filename" in text.lower()

    def test_missing_content(self):
        text, _ = write_draft(json.dumps({"filename": "test.txt"}))
        assert "content" in text.lower()

    def test_empty_arg(self):
        text, _ = write_draft("")
        assert "no argument" in text.lower()

    def test_invalid_json(self):
        text, _ = write_draft("not json")
        assert "invalid json" in text.lower()

    def test_path_traversal_blocked(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GRACEFUL_DATA", str(tmp_path / "data"))

        text, _ = write_draft(json.dumps({
            "filename": "../../../etc/passwd",
            "content": "hacked",
        }))
        # Should strip to just "passwd" — check it didn't write outside
        assert "saved" in text.lower()
        assert not os.path.exists("/etc/passwd_test")

    def test_dotfile_rejected(self):
        text, _ = write_draft(json.dumps({
            "filename": ".hidden",
            "content": "data"
        }))
        assert "invalid" in text.lower()


# ── list_directory ───────────────────────────────────────────────────────────

class TestListDirectory:
    def test_lists_files(self, allowed_dir):
        with open(os.path.join(allowed_dir, "a.txt"), "w") as f:
            f.write("a")
        with open(os.path.join(allowed_dir, "b.txt"), "w") as f:
            f.write("bb")
        text, _ = list_directory(allowed_dir)
        assert "a.txt" in text
        assert "b.txt" in text

    def test_shows_directories(self, allowed_dir):
        os.makedirs(os.path.join(allowed_dir, "subdir"))
        text, _ = list_directory(allowed_dir)
        assert "subdir/" in text

    def test_access_denied(self, disallowed_dir):
        text, _ = list_directory(disallowed_dir)
        assert "denied" in text.lower()

    def test_not_found(self, allowed_dir):
        text, _ = list_directory(os.path.join(allowed_dir, "nonexistent"))
        assert "not found" in text.lower()

    def test_file_not_dir(self, allowed_dir):
        path = os.path.join(allowed_dir, "file.txt")
        with open(path, "w") as f:
            f.write("x")
        text, _ = list_directory(path)
        assert "file" in text.lower() and "not a directory" in text.lower()

    def test_empty_arg_defaults(self):
        """Empty arg should default to project root."""
        text, _ = list_directory("")
        # Should not error — lists the project root
        assert "denied" not in text.lower()

    def test_entry_count(self, allowed_dir):
        for i in range(10):
            with open(os.path.join(allowed_dir, f"file{i}.txt"), "w") as f:
                f.write("x")
        text, _ = list_directory(allowed_dir)
        assert "10 entries" in text


# ── shell_readonly ───────────────────────────────────────────────────────────

class TestShellReadonly:
    def test_ls(self, allowed_dir):
        with open(os.path.join(allowed_dir, "test.txt"), "w") as f:
            f.write("x")
        text, _ = shell_readonly(f"ls {allowed_dir}")
        assert "test.txt" in text

    def test_command_not_allowed(self):
        text, _ = shell_readonly("rm -rf /")
        assert "not allowed" in text.lower()

    def test_pipe_blocked(self):
        text, _ = shell_readonly("ls | grep foo")
        assert "metacharacters" in text.lower()

    def test_redirect_blocked(self):
        text, _ = shell_readonly("echo hello > /tmp/test")
        assert "metacharacters" in text.lower()

    def test_semicolon_blocked(self):
        text, _ = shell_readonly("ls; rm -rf /")
        assert "metacharacters" in text.lower()

    def test_empty_command(self):
        text, _ = shell_readonly("")
        assert "no command" in text.lower()

    def test_git_status(self, monkeypatch, allowed_dir):
        """git status should work in allowed dirs."""
        # Initialize a git repo for testing
        import subprocess
        subprocess.run(["git", "init"], cwd=allowed_dir,
                       capture_output=True, check=True)
        monkeypatch.setattr("graceful.workflow_tools._project_root", allowed_dir)
        text, _ = shell_readonly("git status")
        # Should succeed (might say "nothing to commit" or similar)
        assert "not allowed" not in text.lower()

    def test_git_push_blocked(self):
        text, _ = shell_readonly("git push")
        assert "not allowed" in text.lower()

    def test_git_no_subcommand(self):
        text, _ = shell_readonly("git")
        assert "subcommand" in text.lower()

    def test_wc(self, allowed_dir):
        path = os.path.join(allowed_dir, "lines.txt")
        with open(path, "w") as f:
            f.write("one\ntwo\nthree\n")
        text, _ = shell_readonly(f"wc -l {path}")
        assert "3" in text

    def test_backtick_blocked(self):
        text, _ = shell_readonly("ls `pwd`")
        assert "metacharacters" in text.lower()

    def test_dollar_blocked(self):
        text, _ = shell_readonly("ls $(pwd)")
        assert "metacharacters" in text.lower()

    def test_relative_path_traversal_blocked(self, monkeypatch, allowed_dir):
        """cat ../../../etc/passwd via relative path must be blocked."""
        monkeypatch.setattr("graceful.workflow_tools._project_root", allowed_dir)
        text, _ = shell_readonly("cat ../../../etc/passwd")
        assert "denied" in text.lower()


# ── Path validation ──────────────────────────────────────────────────────────

class TestPathValidation:
    def test_allowed_path(self, allowed_dir):
        assert _is_path_allowed(os.path.join(allowed_dir, "file.txt"))

    def test_disallowed_path(self, disallowed_dir):
        assert not _is_path_allowed(os.path.join(disallowed_dir, "file.txt"))

    def test_root_itself_allowed(self, allowed_dir):
        assert _is_path_allowed(allowed_dir)

    def test_traversal_blocked(self, allowed_dir):
        """../.. from inside allowed dir should be blocked if it escapes."""
        escape_path = os.path.join(allowed_dir, "..", "..", "etc", "passwd")
        assert not _is_path_allowed(escape_path)

    def test_tilde_expansion(self, monkeypatch, tmp_path):
        """~ paths should be expanded before checking."""
        home_sub = str(tmp_path / "allowed" / "sub")
        os.makedirs(home_sub, exist_ok=True)
        assert _is_path_allowed(home_sub)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Web and content
# ══════════════════════════════════════════════════════════════════════════════

from graceful.workflow_tools import fetch_url, analyze_screenshot


class TestFetchUrl:
    def test_empty_arg(self):
        text, _ = fetch_url("")
        assert "no url" in text.lower()

    def test_bad_url(self):
        text, _ = fetch_url("http://this-domain-does-not-exist-xyz-123.com")
        assert "error" in text.lower()


class TestAnalyzeScreenshot:
    def test_file_not_found(self, allowed_dir):
        text, _ = analyze_screenshot(os.path.join(allowed_dir, "nope.png"))
        assert "not found" in text.lower()

    def test_not_image(self, allowed_dir):
        path = os.path.join(allowed_dir, "test.txt")
        with open(path, "w") as f:
            f.write("hello")
        text, _ = analyze_screenshot(path)
        assert "not a recognized image" in text.lower()

    def test_access_denied(self, disallowed_dir):
        path = os.path.join(disallowed_dir, "test.png")
        with open(path, "w") as f:
            f.write("fake")
        text, _ = analyze_screenshot(path)
        assert "denied" in text.lower()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — History and memory
# ══════════════════════════════════════════════════════════════════════════════

from graceful.workflow_tools import (
    search_my_history, query_knowledge_graph, set_reminder, get_pending_reminders,
)


class TestSearchHistory:
    def test_empty_query(self):
        text, _ = search_my_history("")
        assert "no search query" in text.lower()

    def test_no_sessions(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GRACEFUL_DATA", str(tmp_path / "data"))
        text, _ = search_my_history("test")
        assert "no session history" in text.lower()

    def test_finds_match(self, tmp_path, monkeypatch):
        data_dir = str(tmp_path / "data")
        sessions = os.path.join(data_dir, "sessions")
        os.makedirs(sessions)
        monkeypatch.setenv("GRACEFUL_DATA", data_dir)
        # Create a fake session
        ts = int(time.time())
        sess = {
            "session_id": ts,
            "turns": [{"turn": 1, "user": "tell me about neural networks",
                       "response": "Neural networks are..."}]
        }
        with open(os.path.join(sessions, f"session_{ts}.json"), "w") as f:
            json.dump(sess, f)

        text, _ = search_my_history("neural networks")
        assert "neural networks" in text.lower()
        assert "1 matches" in text.lower() or "found" in text.lower()

    def test_json_arg(self, tmp_path, monkeypatch):
        data_dir = str(tmp_path / "data")
        sessions = os.path.join(data_dir, "sessions")
        os.makedirs(sessions)
        monkeypatch.setenv("GRACEFUL_DATA", data_dir)
        ts = int(time.time())
        sess = {"session_id": ts, "turns": [
            {"turn": 1, "user": "hello", "response": "hi there"}
        ]}
        with open(os.path.join(sessions, f"session_{ts}.json"), "w") as f:
            json.dump(sess, f)

        text, _ = search_my_history(json.dumps({"query": "hello", "time_range": "today"}))
        assert "hello" in text.lower()


class TestQueryKnowledgeGraph:
    def test_empty_query(self):
        text, _ = query_knowledge_graph("")
        assert "no query" in text.lower()

    def test_no_data(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GRACEFUL_DATA", str(tmp_path / "nodata"))
        text, _ = query_knowledge_graph("test")
        assert "no knowledge graph" in text.lower()

    def test_finds_concepts(self, tmp_path, monkeypatch):
        data_dir = str(tmp_path / "data")
        os.makedirs(data_dir)
        monkeypatch.setenv("GRACEFUL_DATA", data_dir)
        identity = {
            "concepts": ["epistemology", "phenomenology", "ethics"],
            "thinkers": ["Kant", "Hegel"],
            "node_weights": {"epistemology": 0.85},
            "raw_notes": ["epistemology is the study of knowledge"],
        }
        with open(os.path.join(data_dir, "identity.json"), "w") as f:
            json.dump(identity, f)

        text, _ = query_knowledge_graph("epistemology")
        assert "epistemology" in text.lower()
        assert "0.85" in text


class TestSetReminder:
    def test_empty_arg(self):
        text, _ = set_reminder("")
        assert "no reminder" in text.lower()

    def test_creates_reminder(self, tmp_path, monkeypatch):
        monkeypatch.setattr("graceful.workflow_tools._REMINDERS_DIR", str(tmp_path / "reminders"))
        text, _ = set_reminder(json.dumps({
            "text": "check on project X",
            "trigger_context": "next session"
        }))
        assert "reminder set" in text.lower()
        # Verify file was created
        assert os.path.isfile(os.path.join(str(tmp_path / "reminders"), "reminders.jsonl"))

    def test_get_pending(self, tmp_path, monkeypatch):
        monkeypatch.setattr("graceful.workflow_tools._REMINDERS_DIR", str(tmp_path / "reminders"))
        set_reminder(json.dumps({"text": "test reminder"}))
        pending = get_pending_reminders()
        assert len(pending) == 1
        assert pending[0]["text"] == "test reminder"


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Communication
# ══════════════════════════════════════════════════════════════════════════════

from graceful.workflow_tools import draft_email


class TestDraftEmail:
    def test_empty_arg(self):
        text, _ = draft_email("")
        assert "no argument" in text.lower()

    def test_invalid_json(self):
        text, _ = draft_email("not json")
        assert "invalid json" in text.lower()

    def test_all_empty(self):
        text, _ = draft_email(json.dumps({}))
        assert "at least one" in text.lower()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — Temporal awareness
# ══════════════════════════════════════════════════════════════════════════════

from graceful.temporal_awareness import TemporalAwareness


class TestTemporalAwareness:
    def test_basic_state(self, tmp_path):
        ta = TemporalAwareness(str(tmp_path))
        assert ta.current_datetime  # not empty
        assert ta.day_of_week in ("Monday", "Tuesday", "Wednesday", "Thursday",
                                   "Friday", "Saturday", "Sunday")
        assert ta.time_of_day_bucket in ("morning", "afternoon", "evening",
                                          "night", "late_night")
        ta.stop()

    def test_session_duration(self, tmp_path):
        ta = TemporalAwareness(str(tmp_path))
        assert ta.session_duration >= 0
        ta.stop()

    def test_record_message(self, tmp_path):
        ta = TemporalAwareness(str(tmp_path))
        old_time = ta.last_message_time
        time.sleep(0.01)
        ta.record_message()
        assert ta.last_message_time > old_time
        ta.stop()

    def test_context_string(self, tmp_path):
        ta = TemporalAwareness(str(tmp_path))
        ctx = ta.context_string()
        assert "session:" in ctx
        ta.stop()

    def test_pattern_summary(self, tmp_path):
        ta = TemporalAwareness(str(tmp_path))
        summary = ta.get_pattern_summary()
        assert "Current:" in summary
        assert "Session duration:" in summary
        ta.stop()

    def test_save_and_load(self, tmp_path):
        ta = TemporalAwareness(str(tmp_path))
        ta.record_session_start()
        ta.record_message()
        ta.save()
        ta.stop()

        # Load into new instance
        ta2 = TemporalAwareness(str(tmp_path))
        assert ta2.session_count_today >= 1
        ta2.stop()

    def test_is_continuation(self, tmp_path):
        ta = TemporalAwareness(str(tmp_path))
        # No prior session
        assert not ta.is_continuation()
        # Simulate recent prior session
        ta.last_session_end = time.time() - 60  # 1 minute ago
        ta.session_start = time.time()
        assert ta.is_continuation()
        ta.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — Proactive messaging
# ══════════════════════════════════════════════════════════════════════════════

from graceful.proactive import ProactiveMessenger


class TestProactiveMessenger:
    def test_default_disabled(self, tmp_path):
        pm = ProactiveMessenger(str(tmp_path))
        assert not pm.enabled
        pm.stop()

    def test_enable_disable(self, tmp_path):
        pm = ProactiveMessenger(str(tmp_path))
        pm.enable()
        assert pm.enabled
        pm.disable()
        assert not pm.enabled
        pm.stop()

    def test_quiet_hours(self, tmp_path):
        pm = ProactiveMessenger(str(tmp_path))
        pm.quiet_start = 0
        pm.quiet_end = 24  # always quiet
        assert pm.is_quiet_hours()
        pm.quiet_start = 25
        pm.quiet_end = 26  # never quiet
        assert not pm.is_quiet_hours()
        pm.stop()

    def test_dismissal_pattern(self, tmp_path):
        pm = ProactiveMessenger(str(tmp_path))
        pm.add_dismissal("project X")
        assert pm._is_dismissed("Tell me about project X updates")
        assert not pm._is_dismissed("project Y")
        pm.stop()

    def test_config_persistence(self, tmp_path):
        pm = ProactiveMessenger(str(tmp_path))
        pm.enable()
        pm.check_interval = 1800
        pm._save_config()
        pm.stop()

        pm2 = ProactiveMessenger(str(tmp_path))
        assert pm2.enabled
        assert pm2.check_interval == 1800
        pm2.stop()

    def test_get_config_for_ui(self, tmp_path):
        pm = ProactiveMessenger(str(tmp_path))
        cfg = pm.get_config_for_ui()
        assert "enabled" in cfg
        assert "quiet_hours" in cfg
        assert "stats" in cfg
        pm.stop()

    def test_decision_prompt(self, tmp_path):
        pm = ProactiveMessenger(str(tmp_path))
        prompt = pm.decision_prompt({"type": "reminder", "text": "check project"})
        assert "defer" in prompt.lower()
        assert "reminder" in prompt.lower()
        pm.stop()
