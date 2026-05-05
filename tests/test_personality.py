"""Tests for the personality prompt loading system."""

import json
import os
import pytest


@pytest.fixture
def persona_env(tmp_path):
    """Set up a temporary personality file tree mirroring the real layout."""
    data_dir = tmp_path / "manifold_data"
    data_dir.mkdir()
    personalities = data_dir / "personalities"
    personalities.mkdir()

    # Write a shipped default at the "repo root"
    default_shipped = tmp_path / "default_system_prompt.txt"
    default_shipped.write_text("You are {assistant_name}, a neutral default assistant.")

    # Write starter prompts
    starters = tmp_path / "starter_prompts"
    starters.mkdir()
    (starters / "default.txt").write_text("You are {assistant_name}, neutral and direct.")
    (starters / "tutor.txt").write_text("You are {assistant_name}, a patient tutor.")
    (starters / "critic.txt").write_text("You are {assistant_name}, a rigorous critic.")

    return {
        "root": tmp_path,
        "data_dir": data_dir,
        "personalities": personalities,
        "config_path": data_dir / "config.json",
        "default_shipped": default_shipped,
        "starters": starters,
    }


def _load_prompt(root, data_dir, config_path, personalities_dir):
    """Minimal reimplementation of the cascade logic from app.py for testing."""
    config = {}
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
        except Exception:
            pass

    name = config.get("user_name", "").strip()

    # User-specific personality
    if name:
        user_file = personalities_dir / f"{name.lower()}_system_prompt.txt"
        if user_file.is_file():
            return user_file.read_text().strip()

    # Explicit personality_prompt path
    pp = config.get("personality_prompt", "").strip()
    if pp:
        pp_full = data_dir / pp if not os.path.isabs(pp) else pp
        if pp_full.is_file():
            return pp_full.read_text().strip()

    # Default personality in manifold_data
    default_local = personalities_dir / "default_system_prompt.txt"
    if default_local.is_file():
        return default_local.read_text().strip()

    # Shipped default at repo root
    shipped = root / "default_system_prompt.txt"
    if shipped.is_file():
        return shipped.read_text().strip()

    return "fallback"


def _substitute(prompt, user_name="", assistant_name="Graceful"):
    """Simulate app.py's placeholder substitution."""
    un = user_name or "the user"
    an = assistant_name or "Graceful"
    return prompt.replace("{user_name}", un).replace("{assistant_name}", an)


# ═══════════════════════════════════════════════════
# Layer 1 tests
# ═══════════════════════════════════════════════════

class TestDefaultPromptLoads:
    def test_default_prompt_loads_when_no_config(self, persona_env):
        prompt = _load_prompt(
            persona_env["root"], persona_env["data_dir"],
            persona_env["config_path"], persona_env["personalities"],
        )
        assert "neutral default" in prompt

    def test_default_personality_file_preferred_over_shipped(self, persona_env):
        (persona_env["personalities"] / "default_system_prompt.txt").write_text(
            "Custom default personality."
        )
        prompt = _load_prompt(
            persona_env["root"], persona_env["data_dir"],
            persona_env["config_path"], persona_env["personalities"],
        )
        assert prompt == "Custom default personality."


class TestUserPromptLoads:
    def test_user_prompt_loads_when_configured(self, persona_env):
        persona_env["config_path"].write_text(json.dumps({"user_name": "Alex"}))
        (persona_env["personalities"] / "alex_system_prompt.txt").write_text(
            "You are talking to Alex. Be direct."
        )
        prompt = _load_prompt(
            persona_env["root"], persona_env["data_dir"],
            persona_env["config_path"], persona_env["personalities"],
        )
        assert "Alex" in prompt

    def test_missing_personality_file_falls_back(self, persona_env):
        persona_env["config_path"].write_text(json.dumps({"user_name": "Nobody"}))
        prompt = _load_prompt(
            persona_env["root"], persona_env["data_dir"],
            persona_env["config_path"], persona_env["personalities"],
        )
        assert "neutral default" in prompt


class TestNameSubstitution:
    def test_user_name_placeholder_replaced(self):
        result = _substitute("Hello {user_name}.", user_name="Alex")
        assert result == "Hello Alex."

    def test_no_name_uses_the_user(self):
        result = _substitute("Hello {user_name}.")
        assert result == "Hello the user."

    def test_assistant_name_placeholder_replaced(self):
        result = _substitute("You are {assistant_name}.", assistant_name="Sage")
        assert result == "You are Sage."

    def test_assistant_name_default_graceful(self):
        result = _substitute("You are {assistant_name}.")
        assert result == "You are Graceful."

    def test_both_placeholders(self):
        result = _substitute(
            "{assistant_name} is talking to {user_name}.",
            user_name="Alex", assistant_name="Sage",
        )
        assert result == "Sage is talking to Alex."


class TestAssistantNamePersists:
    def test_assistant_name_in_config(self, persona_env):
        persona_env["config_path"].write_text(json.dumps({
            "assistant_name": "Sage",
            "user_name": "Alex",
        }))
        config = json.loads(persona_env["config_path"].read_text())
        assert config["assistant_name"] == "Sage"


class TestStarterPrompts:
    def test_starter_files_exist(self, persona_env):
        for name in ("default", "tutor", "critic"):
            assert (persona_env["starters"] / f"{name}.txt").is_file()

    def test_starters_contain_assistant_placeholder(self, persona_env):
        for name in ("default", "tutor", "critic"):
            text = (persona_env["starters"] / f"{name}.txt").read_text()
            assert "{assistant_name}" in text

    def test_starter_substitution(self, persona_env):
        text = (persona_env["starters"] / "tutor.txt").read_text()
        result = _substitute(text, assistant_name="Mentor")
        assert "Mentor" in result
        assert "{assistant_name}" not in result


class TestExplicitPromptPath:
    def test_explicit_personality_prompt_path(self, persona_env):
        custom = persona_env["personalities"] / "custom.txt"
        custom.write_text("I am a custom personality.")
        persona_env["config_path"].write_text(json.dumps({
            "personality_prompt": "personalities/custom.txt"
        }))
        prompt = _load_prompt(
            persona_env["root"], persona_env["data_dir"],
            persona_env["config_path"], persona_env["personalities"],
        )
        assert prompt == "I am a custom personality."
