"""Characterization tests for model_router.py."""

from model_router import (
    detect_memory_budget, route_query, route_explain,
    classify_tone, tone_instruction, auto_tag_session, compute_repetition,
)


class TestRouteQuery:
    def test_complex_query_routes_large(self):
        assert route_query("write a philosophical essay comparing Deleuze and Baudrillard") == "large"

    def test_greeting_routes_small(self):
        assert route_query("hi") == "small"

    def test_command_routes_small(self):
        assert route_query("/stats") == "small"

    def test_hi_not_matched_inside_philosophical(self):
        expl = route_explain("write a philosophical essay on consciousness")
        assert "simple: 'hi'" not in expl

    def test_long_message_routes_large(self):
        msg = " ".join(["word"] * 90)
        assert route_query(msg) == "large"

    def test_multiple_questions_with_simple_signals_stays_small(self):
        # "what is" and "why" are simple signals — they outweigh the ? count boost
        assert route_query("What is entropy? How does it relate to information? Why does it matter?") == "small"


class TestClassifyTone:
    def test_casual(self):
        assert classify_tone("lol yeah that's kinda wild tbh") == "casual"

    def test_technical(self):
        assert classify_tone("def train(model, optimizer): import torch; debug the gradient loss function") == "technical"

    def test_emotional(self):
        assert classify_tone("I feel really overwhelmed and I don't know what to do!! help me") == "emotional"

    def test_terse(self):
        assert classify_tone("ok cool") == "terse"


class TestToneInstruction:
    def test_returns_nonempty(self):
        for tone in ("casual", "formal", "technical", "emotional", "creative", "terse"):
            assert len(tone_instruction(tone)) > 0


class TestAutoTagSession:
    def test_philosophy_detected(self):
        hist = [
            {"role": "user", "content": "let's discuss baudrillard and deleuze and phenomenology and ontology"},
            {"role": "user", "content": "more philosophy and metaphysics"},
        ]
        assert "philosophy" in auto_tag_session(hist)

    def test_empty_history_returns_general(self):
        assert auto_tag_session([]) == ["general"]


class TestComputeRepetition:
    def test_repetitive_scores_higher(self):
        rep = compute_repetition("the cat sat on the mat the cat sat on the mat")
        norep = compute_repetition("the quick brown fox jumps over the lazy sleeping dog")
        assert rep > norep

    def test_short_text_returns_zero(self):
        assert compute_repetition("hi") == 0.0


class TestDetectMemoryBudget:
    def test_returns_valid_tier(self):
        assert detect_memory_budget() in ("full", "tight", "single_3b", "small")
