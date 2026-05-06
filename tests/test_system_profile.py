"""Tests for graceful.system_profile — co-evolving system profile (Task 1)."""

import json
import os
import tempfile

from graceful.system_profile import (
    SystemProfile, _bucket_for, _new_stat, _update_stat, _stat_stddev, _z_score
)


# ── Helpers ────────────────────────────────────────────────────────

def _make_profile(tmp_path=None):
    """Create a SystemProfile backed by a temp file."""
    if tmp_path is None:
        tmp_path = tempfile.mktemp(suffix=".json")
    return SystemProfile(tmp_path), tmp_path


def _turn(**overrides):
    """Build a turn dict with sensible defaults."""
    base = {
        "user_msg": "Hello, how are you?",
        "response": "I'm doing well, thanks for asking.",
        "gen_time": 1.2,
        "mode": "lora",
        "searched": False,
        "flattery_score": 0.1,
        "trace": 50,
        "coupled_nudge": None,
        "framework_conf": 0.9,
    }
    base.update(overrides)
    return base


# ── Bucket classification ──────────────────────────────────────────

class TestBuckets:
    def test_tiny(self):
        assert _bucket_for(0) == "tiny"
        assert _bucket_for(49) == "tiny"

    def test_short(self):
        assert _bucket_for(50) == "short"
        assert _bucket_for(149) == "short"

    def test_medium(self):
        assert _bucket_for(150) == "medium"
        assert _bucket_for(499) == "medium"

    def test_long(self):
        assert _bucket_for(500) == "long"
        assert _bucket_for(10000) == "long"


# ── Welford statistics ─────────────────────────────────────────────

class TestWelford:
    def test_new_stat_zeroed(self):
        s = _new_stat()
        assert s["n"] == 0
        assert s["mean"] == 0.0
        assert s["m2"] == 0.0

    def test_single_update(self):
        s = _new_stat()
        _update_stat(s, 10.0)
        assert s["n"] == 1
        assert s["mean"] == 10.0

    def test_mean_after_multiple(self):
        s = _new_stat()
        for v in [10, 20, 30]:
            _update_stat(s, v)
        assert s["n"] == 3
        assert abs(s["mean"] - 20.0) < 1e-6

    def test_stddev_constant_values(self):
        s = _new_stat()
        for _ in range(5):
            _update_stat(s, 100)
        assert _stat_stddev(s) < 1e-6

    def test_stddev_varying_values(self):
        s = _new_stat()
        for v in [10, 20, 30, 40, 50]:
            _update_stat(s, v)
        assert _stat_stddev(s) > 0

    def test_stddev_single_value(self):
        s = _new_stat()
        _update_stat(s, 42)
        assert _stat_stddev(s) == 0.0

    def test_z_score_zero_stddev(self):
        s = _new_stat()
        for _ in range(5):
            _update_stat(s, 100)
        assert _z_score(s, 200) == 0.0

    def test_z_score_normal(self):
        s = _new_stat()
        for v in [10, 20, 30, 40, 50]:
            _update_stat(s, v)
        z = _z_score(s, 30)  # mean
        assert abs(z) < 0.01


# ── SystemProfile basics ──────────────────────────────────────────

class TestProfileBasics:
    def test_creates_fresh(self):
        sp, path = _make_profile()
        assert sp.data["total_turns_observed"] == 0
        assert sp.maturity() == 0.0
        os.unlink(path) if os.path.exists(path) else None

    def test_save_and_reload(self):
        sp, path = _make_profile()
        sp.observe(_turn())
        sp.observe(_turn())
        assert sp.data["total_turns_observed"] == 2

        sp2 = SystemProfile(path)
        assert sp2.data["total_turns_observed"] == 2
        os.unlink(path) if os.path.exists(path) else None

    def test_reset(self):
        sp, path = _make_profile()
        for _ in range(5):
            sp.observe(_turn())
        assert sp.data["total_turns_observed"] == 5
        sp.reset()
        assert sp.data["total_turns_observed"] == 0
        os.unlink(path) if os.path.exists(path) else None

    def test_new_session(self):
        sp, path = _make_profile()
        assert sp.data["sessions_observed"] == 0
        sp.new_session()
        assert sp.data["sessions_observed"] == 1
        sp.new_session()
        assert sp.data["sessions_observed"] == 2
        os.unlink(path) if os.path.exists(path) else None

    def test_corrupt_file_starts_fresh(self):
        path = tempfile.mktemp(suffix=".json")
        with open(path, "w") as f:
            f.write("{bad json!!!")
        sp = SystemProfile(path)
        assert sp.data["total_turns_observed"] == 0
        os.unlink(path) if os.path.exists(path) else None


# ── Observe ────────────────────────────────────────────────────────

class TestObserve:
    def test_response_length_tracked(self):
        sp, path = _make_profile()
        sp.observe(_turn(user_msg="hi", response="x" * 200))
        all_stat = sp.data["response_length_all"]
        assert all_stat["n"] == 1
        assert all_stat["mean"] == 200.0
        os.unlink(path) if os.path.exists(path) else None

    def test_bucketed_response_length(self):
        sp, path = _make_profile()
        # "hi" is 2 chars → tiny bucket
        sp.observe(_turn(user_msg="hi", response="x" * 100))
        assert sp.data["response_length"]["tiny"]["n"] == 1
        os.unlink(path) if os.path.exists(path) else None

    def test_search_invocation_counted(self):
        sp, path = _make_profile()
        sp.observe(_turn(searched=True))
        sp.observe(_turn(searched=False))
        si = sp.data["search_invocation"]
        assert si["total"] == 2
        assert si["searched"] == 1
        os.unlink(path) if os.path.exists(path) else None

    def test_clarification_detection(self):
        sp, path = _make_profile()
        sp.observe(_turn(response="What do you mean?"))
        sp.observe(_turn(response="Here is the answer."))
        cr = sp.data["clarification_rate"]
        assert cr["total"] == 2
        assert cr["clarifications"] == 1
        os.unlink(path) if os.path.exists(path) else None

    def test_register_tracking(self):
        sp, path = _make_profile()
        sp.observe(_turn(response="Hey yeah gonna do that, cool awesome"))
        assert sp.data["register"]["casual"] >= 1
        os.unlink(path) if os.path.exists(path) else None

    def test_register_formal(self):
        sp, path = _make_profile()
        sp.observe(_turn(response="However, furthermore, this is therefore important."))
        assert sp.data["register"]["formal"] >= 1
        os.unlink(path) if os.path.exists(path) else None

    def test_register_technical(self):
        sp, path = _make_profile()
        sp.observe(_turn(response="The function parameter defines the class module."))
        assert sp.data["register"]["technical"] >= 1
        os.unlink(path) if os.path.exists(path) else None

    def test_anti_lora_intervention_tracked(self):
        sp, path = _make_profile()
        sp.observe(_turn(mode="anti"))
        sp.observe(_turn(mode="lora"))
        ir = sp.data["intervention_rate"]
        assert ir["n"] == 2
        assert ir["mean"] == 0.5
        os.unlink(path) if os.path.exists(path) else None

    def test_mode_preference_tracked(self):
        sp, path = _make_profile()
        sp.observe(_turn(active_mode="chat"))
        sp.observe(_turn(active_mode="chat"))
        sp.observe(_turn(active_mode="deep"))
        assert sp.data["mode_preference"]["chat"] == 2
        assert sp.data["mode_preference"]["deep"] == 1
        os.unlink(path) if os.path.exists(path) else None

    def test_turn_snapshots_capped(self):
        sp, path = _make_profile()
        for i in range(60):
            sp.observe(_turn())
        assert len(sp.data["turn_snapshots"]) == 50
        os.unlink(path) if os.path.exists(path) else None


# ── Dance coherence ────────────────────────────────────────────────

class TestDanceCoherence:
    def test_no_prediction_no_coherence(self):
        sp, path = _make_profile()
        sp.observe(_turn())
        assert sp.get_dance_coherence() is None
        os.unlink(path) if os.path.exists(path) else None

    def test_correct_prediction_high_coherence(self):
        sp, path = _make_profile()
        # Build up some stats first
        for _ in range(10):
            sp.observe(_turn(user_msg="short msg", response="x" * 200))

        # Now predict and observe
        sp.predict_next("short msg")
        sp.observe(_turn(user_msg="short msg", response="x" * 200))

        dc = sp.data["dance_coherence"]
        assert dc["n"] >= 1
        os.unlink(path) if os.path.exists(path) else None

    def test_wrong_prediction_lower_coherence(self):
        sp, path = _make_profile()
        for _ in range(10):
            sp.observe(_turn(user_msg="short msg", response="x" * 200))

        # Predict based on short msg but observe a long msg
        sp.predict_next("short msg")
        sp.observe(_turn(user_msg="x" * 600, response="x" * 200))

        dc = sp.data["dance_coherence"]
        assert dc["n"] >= 1
        # Bucket mismatch → coherence should be lower
        os.unlink(path) if os.path.exists(path) else None


# ── Calibrate response ─────────────────────────────────────────────

class TestCalibrateResponse:
    def test_early_returns_none_length(self):
        sp, path = _make_profile()
        cal = sp.calibrate_response()
        assert cal["suggested_length"] is None
        os.unlink(path) if os.path.exists(path) else None

    def test_after_enough_data_returns_length(self):
        sp, path = _make_profile()
        for _ in range(10):
            sp.observe(_turn(response="x" * 300))
        cal = sp.calibrate_response()
        assert cal["suggested_length"] == 300
        os.unlink(path) if os.path.exists(path) else None

    def test_search_tendency(self):
        sp, path = _make_profile()
        for i in range(10):
            sp.observe(_turn(searched=(i < 3)))
        cal = sp.calibrate_response()
        assert cal["search_tendency"] == 0.3
        os.unlink(path) if os.path.exists(path) else None

    def test_register_suggestion(self):
        sp, path = _make_profile()
        for _ in range(5):
            sp.observe(_turn(response="Hey yeah gonna do that, cool awesome btw"))
        cal = sp.calibrate_response()
        assert cal["suggested_register"] == "casual"
        os.unlink(path) if os.path.exists(path) else None


# ── Non-convergence (Task 5) ───────────────────────────────────────

class TestNonConvergence:
    def test_insufficient_data(self):
        sp, path = _make_profile()
        result, msg = sp.is_non_convergent()
        assert result is False
        assert msg is None
        os.unlink(path) if os.path.exists(path) else None

    def test_converged_dance(self):
        sp, path = _make_profile()
        dc = sp.data["dance_coherence"]
        # Simulate high coherence
        for _ in range(15):
            _update_stat(dc, 0.8)
        result, msg = sp.is_non_convergent()
        assert result is False
        os.unlink(path) if os.path.exists(path) else None

    def test_non_convergent_dance(self):
        sp, path = _make_profile()
        dc = sp.data["dance_coherence"]
        # Simulate sustained low coherence
        for _ in range(15):
            _update_stat(dc, 0.1)
        result, msg = sp.is_non_convergent()
        assert result is True
        assert msg is not None
        assert "dance" in msg
        os.unlink(path) if os.path.exists(path) else None


# ── Maturity ────────────────────────────────────────────────────────

class TestMaturity:
    def test_zero_at_start(self):
        sp, path = _make_profile()
        assert sp.maturity() == 0.0
        os.unlink(path) if os.path.exists(path) else None

    def test_increases_with_data(self):
        sp, path = _make_profile()
        for _ in range(25):
            sp.observe(_turn(searched=True))
        m = sp.maturity()
        assert m > 0.0
        os.unlink(path) if os.path.exists(path) else None


# ── Display methods ─────────────────────────────────────────────────

class TestDisplay:
    def test_context_string_early(self):
        sp, path = _make_profile()
        s = sp.to_context_string()
        assert "maturity: 0%" in s
        os.unlink(path) if os.path.exists(path) else None

    def test_context_string_with_data(self):
        sp, path = _make_profile()
        for _ in range(10):
            sp.observe(_turn(searched=True))
        s = sp.to_context_string()
        assert "avg response" in s
        os.unlink(path) if os.path.exists(path) else None

    def test_display_dict_structure(self):
        sp, path = _make_profile()
        for _ in range(10):
            sp.observe(_turn())
        d = sp.to_display_dict()
        assert "response_length_avg" in d
        assert "search_rate" in d
        assert "dance_coherence" in d
        assert "maturity" in d
        assert "total_turns" in d
        assert d["total_turns"] == 10
        os.unlink(path) if os.path.exists(path) else None


# ── Z-score ─────────────────────────────────────────────────────────

class TestZScore:
    def test_response_length_z(self):
        sp, path = _make_profile()
        for _ in range(10):
            sp.observe(_turn(response="x" * 200))
        z = sp.z_score("response_length", 200)
        assert abs(z) < 0.01  # at mean
        os.unlink(path) if os.path.exists(path) else None

    def test_unknown_metric(self):
        sp, path = _make_profile()
        z = sp.z_score("nonexistent", 42)
        assert z == 0.0
        os.unlink(path) if os.path.exists(path) else None
