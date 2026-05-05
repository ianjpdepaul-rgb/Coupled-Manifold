"""Characterization tests for context_budget.py."""

from context_budget import ContextBudget


class TestAllocate:
    def test_sum_within_budget(self):
        cb = ContextBudget(12000)
        b = cb.allocate("what is entropy", False, False, 5)
        assert sum(b.values()) <= 12000

    def test_required_keys_present(self):
        cb = ContextBudget(12000)
        b = cb.allocate("hello", False, False, 5)
        assert "history" in b
        assert "system" in b

    def test_identity_query_boosts_identity(self):
        cb = ContextBudget(12000)
        b = cb.allocate("tell me about my identity", False, False, 5)
        assert b["identity"] >= b.get("search", 0)

    def test_early_turn_boosts_identity(self):
        cb = ContextBudget(12000)
        b = cb.allocate("anything", False, False, 1)
        assert b["identity"] >= 500

    def test_search_query_allocates_search(self):
        cb = ContextBudget(12000)
        b = cb.allocate("hello", True, False, 5)
        assert b["search"] > 0

    def test_academic_query_allocates_corpus(self):
        cb = ContextBudget(12000)
        b = cb.allocate("what does the research paper say about methodology", False, True, 5)
        assert b["corpus"] > 0

    def test_proportional_scaling_under_tight_budget(self):
        cb = ContextBudget(200)
        b = cb.allocate("hello", False, False, 5)
        assert sum(b.values()) <= 200


class TestTruncateToBudget:
    def test_respects_budget(self):
        cb = ContextBudget(12000)
        long_text = "word " * 1000
        trunc = cb.truncate_to_budget(long_text, 100)
        assert len(trunc) <= 150  # budget + truncation marker

    def test_short_text_unchanged(self):
        cb = ContextBudget(12000)
        assert cb.truncate_to_budget("short", 1000) == "short"

    def test_empty_text_returns_empty(self):
        cb = ContextBudget(12000)
        assert cb.truncate_to_budget("", 100) == ""

    def test_zero_budget_returns_empty(self):
        cb = ContextBudget(12000)
        assert cb.truncate_to_budget("some text", 0) == ""


class TestFormatBarHtml:
    def test_returns_html(self):
        html = ContextBudget(12000).format_bar_html({"history": 2000, "corpus": 500, "system": 400})
        assert "<span" in html
