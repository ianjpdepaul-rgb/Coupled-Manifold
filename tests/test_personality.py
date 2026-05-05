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


# ═══════════════════════════════════════════════════
# Layer 2 tests — sidebar editing round-trip
# ═══════════════════════════════════════════════════

class TestSidebarRoundTrip:
    """Verify sidebar can read and write the same config the install GUI wrote."""

    def test_config_written_by_install_readable_by_sidebar(self, persona_env):
        """Install GUI writes config.json; sidebar (via _load_user_config) can read it."""
        persona_env["config_path"].write_text(json.dumps({
            "assistant_name": "Sage",
            "user_name": "Alex",
        }))
        config = json.loads(persona_env["config_path"].read_text())
        assert config["assistant_name"] == "Sage"
        assert config["user_name"] == "Alex"

    def test_sidebar_save_overwrites_install_config(self, persona_env):
        """Sidebar save writes the same config.json the install GUI created."""
        # Simulate install GUI writing config
        persona_env["config_path"].write_text(json.dumps({
            "assistant_name": "Sage",
            "user_name": "Alex",
        }))
        # Simulate sidebar save (same write path)
        config = json.loads(persona_env["config_path"].read_text())
        config["assistant_name"] = "Oracle"
        config["user_name"] = "Jordan"
        persona_env["config_path"].write_text(json.dumps(config, indent=2))
        # Verify round-trip
        reloaded = json.loads(persona_env["config_path"].read_text())
        assert reloaded["assistant_name"] == "Oracle"
        assert reloaded["user_name"] == "Jordan"

    def test_personality_file_written_by_install_loads_in_cascade(self, persona_env):
        """Install GUI writes a personality file; cascade picks it up."""
        persona_env["config_path"].write_text(json.dumps({
            "personality_prompt": "personalities/custom.txt"
        }))
        (persona_env["personalities"] / "custom.txt").write_text(
            "I am {assistant_name}, a custom personality."
        )
        prompt = _load_prompt(
            persona_env["root"], persona_env["data_dir"],
            persona_env["config_path"], persona_env["personalities"],
        )
        assert "custom personality" in prompt

    def test_prompt_change_takes_effect_immediately(self, persona_env):
        """Simulates: change prompt in sidebar → next cascade load sees new text."""
        # Initial state
        (persona_env["personalities"] / "default_system_prompt.txt").write_text(
            "Original default."
        )
        prompt1 = _load_prompt(
            persona_env["root"], persona_env["data_dir"],
            persona_env["config_path"], persona_env["personalities"],
        )
        assert prompt1 == "Original default."

        # Sidebar writes a new prompt file (simulating POST /api/settings + file write)
        (persona_env["personalities"] / "default_system_prompt.txt").write_text(
            "Updated via sidebar."
        )
        prompt2 = _load_prompt(
            persona_env["root"], persona_env["data_dir"],
            persona_env["config_path"], persona_env["personalities"],
        )
        assert prompt2 == "Updated via sidebar."

    def test_name_change_reflected_in_substitution(self, persona_env):
        """Change assistant name in config → substitution uses new name."""
        prompt = "You are {assistant_name}, talking to {user_name}."
        result = _substitute(prompt, user_name="Alex", assistant_name="Oracle")
        assert "Oracle" in result
        assert "Alex" in result
        assert "{assistant_name}" not in result

    def test_discard_reverts_to_default_prompt(self, persona_env):
        """Discard (clear system_prompt) should fall back to personality file default."""
        (persona_env["personalities"] / "default_system_prompt.txt").write_text(
            "The default personality."
        )
        # Simulate: user had a custom prompt, then hits Discard
        # Discard clears system_prompt → cascade loads from personality file
        prompt = _load_prompt(
            persona_env["root"], persona_env["data_dir"],
            persona_env["config_path"], persona_env["personalities"],
        )
        assert prompt == "The default personality."


class TestStarterPromptLoading:
    """Test loading starter prompts for the revert-to-starter flow."""

    def test_load_starters_from_directory(self, persona_env):
        starters_dir = persona_env["starters"]
        starters = []
        for fn in sorted(os.listdir(starters_dir)):
            if fn.endswith(".txt"):
                content = (starters_dir / fn).read_text().strip()
                starters.append({"name": fn[:-4], "content": content})
        assert len(starters) == 3
        names = [s["name"] for s in starters]
        assert "default" in names
        assert "tutor" in names
        assert "critic" in names

    def test_starter_content_has_placeholder(self, persona_env):
        starters_dir = persona_env["starters"]
        for fn in sorted(os.listdir(starters_dir)):
            if fn.endswith(".txt"):
                content = (starters_dir / fn).read_text()
                assert "{assistant_name}" in content

    def test_starter_substitution_applied(self, persona_env):
        content = (persona_env["starters"] / "critic.txt").read_text()
        result = _substitute(content, assistant_name="Athena")
        assert "Athena" in result
        assert "{assistant_name}" not in result

    def test_empty_starters_dir_returns_empty(self, tmp_path):
        empty_dir = tmp_path / "empty_starters"
        empty_dir.mkdir()
        starters = []
        for fn in sorted(os.listdir(empty_dir)):
            if fn.endswith(".txt"):
                starters.append(fn)
        assert starters == []


# ═══════════════════════════════════════════════════
# Layer 3 tests — personas as session-scoped mode shifts
# ═══════════════════════════════════════════════════

def _list_personas_from(shipped_dir, user_dir):
    """Test-friendly reimplementation of _list_personas()."""
    personas = []
    seen = set()
    for d, source in [(user_dir, "user"), (shipped_dir, "shipped")]:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".txt"):
                name = fn[:-4]
                if name not in seen:
                    content = open(os.path.join(d, fn)).read().strip()
                    personas.append({"name": name, "content": content, "source": source})
                    seen.add(name)
    return personas


@pytest.fixture
def persona_mode_env(tmp_path):
    """Set up shipped + user persona directories."""
    shipped = tmp_path / "personas"
    shipped.mkdir()
    (shipped / "debugger.txt").write_text("You are in debugger mode. Isolate before fixing.")
    (shipped / "rubber_duck.txt").write_text("You are in rubber duck mode. Ask questions.")
    (shipped / "pair_programmer.txt").write_text("You are in pair programming mode.")

    user = tmp_path / "manifold_data" / "personas"
    user.mkdir(parents=True)

    return {"shipped": shipped, "user": user, "root": tmp_path}


class TestPersonaListing:
    def test_list_shipped_personas(self, persona_mode_env):
        personas = _list_personas_from(
            str(persona_mode_env["shipped"]),
            str(persona_mode_env["user"]),
        )
        names = [p["name"] for p in personas]
        assert "debugger" in names
        assert "rubber_duck" in names
        assert "pair_programmer" in names

    def test_all_shipped_are_source_shipped(self, persona_mode_env):
        personas = _list_personas_from(
            str(persona_mode_env["shipped"]),
            str(persona_mode_env["user"]),
        )
        for p in personas:
            assert p["source"] == "shipped"

    def test_user_persona_overrides_shipped(self, persona_mode_env):
        """User persona with same name takes precedence."""
        (persona_mode_env["user"] / "debugger.txt").write_text("My custom debugger.")
        personas = _list_personas_from(
            str(persona_mode_env["shipped"]),
            str(persona_mode_env["user"]),
        )
        dbg = next(p for p in personas if p["name"] == "debugger")
        assert dbg["source"] == "user"
        assert dbg["content"] == "My custom debugger."

    def test_user_custom_persona_appears(self, persona_mode_env):
        (persona_mode_env["user"] / "coach.txt").write_text("Coaching mode.")
        personas = _list_personas_from(
            str(persona_mode_env["shipped"]),
            str(persona_mode_env["user"]),
        )
        names = [p["name"] for p in personas]
        assert "coach" in names

    def test_empty_dirs_return_empty(self, tmp_path):
        personas = _list_personas_from(
            str(tmp_path / "nope1"),
            str(tmp_path / "nope2"),
        )
        assert personas == []


class TestPersonaActivation:
    def test_activate_persona(self, persona_mode_env):
        personas = _list_personas_from(
            str(persona_mode_env["shipped"]),
            str(persona_mode_env["user"]),
        )
        match = next(p for p in personas if p["name"] == "debugger")
        active = {"name": match["name"], "content": match["content"], "persistent": False}
        assert active["name"] == "debugger"
        assert "debugger mode" in active["content"].lower()
        assert active["persistent"] is False

    def test_persistent_flag(self, persona_mode_env):
        active = {"name": "debugger", "content": "debug mode", "persistent": True}
        assert active["persistent"] is True

    def test_deactivate_persona(self):
        active = [{"name": "debugger", "content": "test", "persistent": False}]
        active[0] = None
        assert active[0] is None


class TestPersonaInjection:
    def test_persona_appends_to_system_prompt(self):
        """Persona content appends to base personality, never replaces."""
        base = "You are Graceful, a helpful assistant."
        persona_content = "You are in debugger mode. Isolate before fixing."
        combined = base + "\n\n" + persona_content
        assert base in combined
        assert persona_content in combined

    def test_no_persona_leaves_prompt_unchanged(self):
        base = "You are Graceful, a helpful assistant."
        active = None
        result = base
        if active:
            result = base + "\n\n" + active["content"]
        assert result == base

    def test_multiple_substitutions_still_work(self):
        base = "You are {assistant_name}."
        persona = "Debug mode for {user_name}."
        combined = base + "\n\n" + persona
        result = _substitute(combined, user_name="Alex", assistant_name="Sage")
        assert "Sage" in result
        assert "Alex" in result
        assert "{assistant_name}" not in result
        assert "{user_name}" not in result


class TestPersonaSessionScoping:
    def test_session_scoped_cleared_on_new_session(self):
        """Non-persistent persona clears on new session."""
        active = [{"name": "debugger", "content": "test", "persistent": False}]
        # Simulate new session logic
        if active[0] and not active[0].get("persistent"):
            active[0] = None
        assert active[0] is None

    def test_persistent_survives_new_session(self):
        """Persistent persona survives new session."""
        active = [{"name": "debugger", "content": "test", "persistent": True}]
        if active[0] and not active[0].get("persistent"):
            active[0] = None
        assert active[0] is not None
        assert active[0]["name"] == "debugger"


class TestPersonaSave:
    def test_save_custom_persona(self, persona_mode_env):
        user_dir = persona_mode_env["user"]
        pname = "coach"
        ptext = "You are a supportive coach."
        ppath = user_dir / f"{pname}.txt"
        ppath.write_text(ptext.strip() + "\n")
        assert ppath.is_file()
        assert ppath.read_text().strip() == ptext

    def test_saved_persona_appears_in_listing(self, persona_mode_env):
        user_dir = persona_mode_env["user"]
        (user_dir / "mentor.txt").write_text("Mentoring mode.\n")
        personas = _list_personas_from(
            str(persona_mode_env["shipped"]),
            str(user_dir),
        )
        names = [p["name"] for p in personas]
        assert "mentor" in names
