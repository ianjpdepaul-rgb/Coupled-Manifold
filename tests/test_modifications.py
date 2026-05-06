"""Tests for graceful.modifications — Stage 2 self-modification reversibility."""

import json
import os
import tempfile

from graceful.modifications import ModificationTracker
from graceful.tools import (
    make_adjust_sampling, make_pin_context, make_update_system_prompt,
    make_mark_disagreement, make_save_to_long_term,
)


class TestModificationTracker:
    """Core tracker behavior."""

    def test_log_modification(self):
        with tempfile.TemporaryDirectory() as td:
            mt = ModificationTracker(td)
            entry = mt.log_modification("adjust_sampling",
                                        {"temperature": 0.7},
                                        {"temperature": 0.5},
                                        "user wants focused", turn_id=3)
            assert entry["tool"] == "adjust_sampling"
            assert entry["before"] == {"temperature": 0.7}
            assert entry["after"] == {"temperature": 0.5}
            assert entry["reason"] == "user wants focused"
            assert entry["reverted"] is False

    def test_undo_last(self):
        with tempfile.TemporaryDirectory() as td:
            mt = ModificationTracker(td)
            mt.log_modification("tool_a", "before_a", "after_a", "reason_a", turn_id=1)
            mt.log_modification("tool_b", "before_b", "after_b", "reason_b", turn_id=2)

            reverted = mt.undo_last()
            assert reverted["tool"] == "tool_b"
            assert reverted["reverted"] is True

            # Second undo gets tool_a
            reverted2 = mt.undo_last()
            assert reverted2["tool"] == "tool_a"

            # Nothing left
            assert mt.undo_last() is None

    def test_undo_session(self):
        with tempfile.TemporaryDirectory() as td:
            mt = ModificationTracker(td)
            mt.log_modification("a", 1, 2, "r1", turn_id=1)
            mt.log_modification("b", 3, 4, "r2", turn_id=2)
            mt.log_modification("c", 5, 6, "r3", turn_id=3)

            reverted = mt.undo_session()
            assert len(reverted) == 3
            # Newest first
            assert reverted[0]["tool"] == "c"
            assert reverted[1]["tool"] == "b"
            assert reverted[2]["tool"] == "a"

            # All marked reverted
            assert mt.undo_session() == []

    def test_list_modifications_session(self):
        with tempfile.TemporaryDirectory() as td:
            mt = ModificationTracker(td)
            mt.log_modification("a", 1, 2, "r1", turn_id=1)
            mt.log_modification("b", 3, 4, "r2", turn_id=2)

            mods = mt.list_modifications("session")
            assert len(mods) == 2

            # Undo one
            mt.undo_last()
            mods = mt.list_modifications("session")
            assert len(mods) == 1
            assert mods[0]["tool"] == "a"

    def test_list_modifications_all_from_disk(self):
        with tempfile.TemporaryDirectory() as td:
            mt = ModificationTracker(td)
            mt.log_modification("x", 0, 1, "reason", turn_id=5)

            # New tracker reads from disk
            mt2 = ModificationTracker(td)
            all_mods = mt2.list_modifications("all")
            assert len(all_mods) >= 1
            assert all_mods[0]["tool"] == "x"

    def test_pending_count(self):
        with tempfile.TemporaryDirectory() as td:
            mt = ModificationTracker(td)
            assert mt.pending_count() == 0
            mt.log_modification("a", 1, 2, "r", turn_id=1)
            mt.log_modification("b", 3, 4, "r", turn_id=2)
            assert mt.pending_count() == 2
            mt.undo_last()
            assert mt.pending_count() == 1

    def test_persists_to_jsonl(self):
        with tempfile.TemporaryDirectory() as td:
            mt = ModificationTracker(td)
            mt.log_modification("test", "old", "new", "because", turn_id=1)

            log_path = os.path.join(td, "logs", "modifications.jsonl")
            assert os.path.isfile(log_path)
            with open(log_path) as f:
                entry = json.loads(f.readline())
            assert entry["tool"] == "test"
            assert entry["before"] == "old"
            assert entry["after"] == "new"

    def test_session_scope(self):
        with tempfile.TemporaryDirectory() as td:
            mt = ModificationTracker(td)
            mt.log_modification("a", 1, 2, "r", turn_id=1, scope="session")
            mods = mt.list_modifications("session")
            assert mods[0]["scope"] == "session"

    def test_persistent_scope(self):
        with tempfile.TemporaryDirectory() as td:
            mt = ModificationTracker(td)
            mt.log_modification("a", 1, 2, "r", turn_id=1, scope="persistent")
            mods = mt.list_modifications("session")
            assert mods[0]["scope"] == "persistent"


class TestSelfModificationTools:
    """Test the Stage 2 factory-produced tool handlers."""

    def test_adjust_sampling(self):
        state = {"temperature": 0.7, "top_p": 0.95, "top_k": 0}
        def get_state(): return dict(state)
        def set_state(t, p, k):
            state["temperature"] = t
            state["top_p"] = p
            state["top_k"] = k

        with tempfile.TemporaryDirectory() as td:
            mt = ModificationTracker(td)
            handler = make_adjust_sampling(get_state, set_state, mt)

            text, _ = handler('{"temperature": 0.4, "reason": "focus mode"}')
            assert "0.4" in text
            assert state["temperature"] == 0.4
            assert mt.pending_count() == 1

    def test_adjust_sampling_clamps(self):
        state = {"temperature": 0.7, "top_p": 0.95, "top_k": 0}
        def get_state(): return dict(state)
        def set_state(t, p, k):
            state["temperature"] = t
            state["top_p"] = p
            state["top_k"] = k

        with tempfile.TemporaryDirectory() as td:
            mt = ModificationTracker(td)
            handler = make_adjust_sampling(get_state, set_state, mt)

            handler('{"temperature": 99.0, "top_p": -5, "reason": "extreme"}')
            assert state["temperature"] == 2.0  # clamped
            assert state["top_p"] == 0.01  # clamped

    def test_adjust_sampling_missing_reason(self):
        state = {"temperature": 0.7, "top_p": 0.95, "top_k": 0}
        with tempfile.TemporaryDirectory() as td:
            mt = ModificationTracker(td)
            handler = make_adjust_sampling(lambda: dict(state), lambda t,p,k: None, mt)
            text, _ = handler('{"temperature": 0.5}')
            assert "reason" in text.lower()

    def test_pin_context(self):
        pins = []
        def get_pins(): return list(pins)
        def add_pin(text): pins.append(text)

        with tempfile.TemporaryDirectory() as td:
            mt = ModificationTracker(td)
            handler = make_pin_context(get_pins, add_pin, mt)

            text, _ = handler('{"text": "remember this", "reason": "important context"}')
            assert "Pinned" in text
            assert pins == ["remember this"]
            assert mt.pending_count() == 1

    def test_update_system_prompt(self):
        additions = []
        def get_additions(): return list(additions)
        def add_addition(text): additions.append(text)

        with tempfile.TemporaryDirectory() as td:
            mt = ModificationTracker(td)
            handler = make_update_system_prompt(get_additions, add_addition, mt)

            text, _ = handler('{"addition": "be concise", "reason": "user preference"}')
            assert "updated" in text.lower()
            assert additions == ["be concise"]

    def test_mark_disagreement(self):
        disagreements = []
        def get_d(): return list(disagreements)
        def add_d(entry): disagreements.append(entry)

        with tempfile.TemporaryDirectory() as td:
            mt = ModificationTracker(td)
            handler = make_mark_disagreement(get_d, add_d, mt)

            text, _ = handler(json.dumps({
                "topic": "AI consciousness",
                "user_position": "not possible",
                "prior_position": "uncertain",
                "reason": "user stated firm view"
            }))
            assert "noted" in text.lower()
            assert len(disagreements) == 1
            assert disagreements[0]["topic"] == "AI consciousness"
            # Persistent scope
            mods = mt.list_modifications("session")
            assert mods[0]["scope"] == "persistent"

    def test_save_to_long_term(self):
        saved = []
        def save_fn(text, category): saved.append((text, category))

        with tempfile.TemporaryDirectory() as td:
            mt = ModificationTracker(td)
            handler = make_save_to_long_term(save_fn, mt)

            text, _ = handler('{"text": "user is a physicist", "category": "identity", "reason": "stated in conversation"}')
            assert "Saved" in text
            assert saved == [("user is a physicist", "identity")]
            mods = mt.list_modifications("session")
            assert mods[0]["scope"] == "persistent"

    def test_invalid_json_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            mt = ModificationTracker(td)
            handler = make_adjust_sampling(lambda: {}, lambda t,p,k: None, mt)
            text, _ = handler("not json at all")
            assert "Invalid JSON" in text

    def test_empty_arg_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            mt = ModificationTracker(td)
            handler = make_adjust_sampling(lambda: {}, lambda t,p,k: None, mt)
            text, _ = handler("")
            assert "No argument" in text
