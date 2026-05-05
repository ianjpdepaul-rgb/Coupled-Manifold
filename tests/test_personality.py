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
    default_shipped.write_text("You are Graceful, a neutral default assistant.")

    return {
        "root": tmp_path,
        "data_dir": data_dir,
        "personalities": personalities,
        "config_path": data_dir / "config.json",
        "default_shipped": default_shipped,
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


class TestDefaultPromptLoads:
    def test_default_prompt_loads_when_no_config(self, persona_env):
        """No config.json, no personality files — falls back to shipped default."""
        prompt = _load_prompt(
            persona_env["root"], persona_env["data_dir"],
            persona_env["config_path"], persona_env["personalities"],
        )
        assert "neutral default" in prompt

    def test_default_personality_file_preferred_over_shipped(self, persona_env):
        """A default_system_prompt.txt in personalities/ takes precedence over the shipped one."""
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
        """config.json with user_name → loads matching personality file."""
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
        """config.json points to a user, but no file exists — falls back to shipped default."""
        persona_env["config_path"].write_text(json.dumps({"user_name": "Nobody"}))
        prompt = _load_prompt(
            persona_env["root"], persona_env["data_dir"],
            persona_env["config_path"], persona_env["personalities"],
        )
        assert "neutral default" in prompt


class TestUserNameSubstitution:
    def test_user_name_placeholder_replaced(self, persona_env):
        """System prompt with {user_name} placeholder gets substituted."""
        (persona_env["personalities"] / "default_system_prompt.txt").write_text(
            "You are talking to {user_name}. Be helpful."
        )
        prompt = _load_prompt(
            persona_env["root"], persona_env["data_dir"],
            persona_env["config_path"], persona_env["personalities"],
        )
        # The _load_prompt function returns raw text; substitution happens in app.py
        assert "{user_name}" in prompt
        # Simulate app.py's substitution
        substituted = prompt.replace("{user_name}", "Alex")
        assert "Alex" in substituted
        assert "{user_name}" not in substituted

    def test_no_name_uses_the_user(self):
        """When no user_name is set, placeholder becomes 'the user'."""
        prompt = "You are talking to {user_name}."
        result = prompt.replace("{user_name}", "the user")
        assert result == "You are talking to the user."


class TestExplicitPromptPath:
    def test_explicit_personality_prompt_path(self, persona_env):
        """config.json with personality_prompt path → loads that file."""
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
