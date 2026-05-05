"""Characterization tests for graceful.flattery."""

from graceful.flattery import compute_flattery_score, _is_greeting_msg, _lexical_match


class TestComputeFlattery:
    def test_short_response_zero(self):
        assert compute_flattery_score("yes ok") == 0.0

    def test_pure_agreement_scores_high(self):
        text = ("I absolutely agree with you. That's exactly right. "
                "What a brilliant and fascinating observation. Definitely correct. "
                "You make a good point, and I think you are wonderful.")
        score = compute_flattery_score(text)
        assert score > 0.5

    def test_friction_lowers_score(self):
        text = ("I agree that's interesting, however I disagree with the conclusion. "
                "Actually, that's not quite right. Consider the alternative. "
                "There's a problematic tension in this argument.")
        score = compute_flattery_score(text)
        assert score < 0.5

    def test_neutral_text_low(self):
        text = ("The function takes an integer argument and returns a string. "
                "It loops through the array, checking each element against "
                "the threshold value before appending to the result list.")
        score = compute_flattery_score(text)
        assert score == 0.0


class TestIsGreeting:
    def test_hi(self):
        assert _is_greeting_msg("hi") is True

    def test_hello_exclaim(self):
        assert _is_greeting_msg("hello!") is True

    def test_sentence_not_greeting(self):
        assert _is_greeting_msg("explain the theory of relativity") is False

    def test_thanks(self):
        assert _is_greeting_msg("thanks") is True


class TestLexicalMatch:
    def test_agree_not_in_disagree(self):
        assert _lexical_match("agree", "I disagree with that") is False

    def test_agree_standalone(self):
        assert _lexical_match("agree", "I agree with that") is True

    def test_multiword_phrase(self):
        assert _lexical_match("great point", "that's a great point") is True
