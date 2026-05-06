"""
COUPLED MANIFOLD — Context Budget Manager
==========================================
Dynamic context window allocation based on query type.
Keeps expensive context types (search, corpus) from squeezing out
conversation history on short-context queries, and vice versa.

Usage:
    from context_budget import ContextBudget
    cb     = ContextBudget(MAX_PROMPT_CHARS)
    budgets = cb.allocate(user_msg, has_search, has_corpus, turn_count[0])
    # Then truncate each component:
    corpus_text = cb.truncate_to_budget(corpus_text, budgets["corpus"])
"""


def _is_academic(query: str) -> bool:
    signals = [
        "paper", "study", "research", "journal", "doi", "arxiv", "published",
        "abstract", "citation", "authors", "findings", "results", "methodology",
        "coupled manifold", "framework", "analysis", "model", "theory",
    ]
    return any(s in query.lower() for s in signals)


class ContextBudget:
    """
    Allocate MAX_PROMPT_CHARS across context components dynamically.

    Components:
        identity      — identity / persona block
        interoception — interoceptive state block
        corpus        — retrieved corpus chunks
        search        — web search results
        history       — conversation history
        system        — system prompt (always reserved)
    """

    def __init__(self, max_chars: int = 12000):
        self.max_chars = max_chars

    def allocate(self, query: str, has_search: bool, has_corpus: bool,
                 turn_count: int) -> dict:
        """
        Return a dict of {component: char_budget}.
        Budgets are scaled down proportionally if the total exceeds max_chars.
        """
        q = query.lower()

        if _is_academic(q):
            budget = {
                "corpus":        1200,
                "search":        1500 if has_search else 0,
                "history":       1200,
                "identity":      200,
                "interoception": 350,
            }
        elif any(w in q for w in ["you know", "remember", "told you",
                                   "my ", "i am", "about me"]):
            budget = {
                "identity":      600,
                "history":       3000,
                "corpus":        600,
                "search":        0,
                "interoception": 350,
            }
        elif has_search:
            budget = {
                "search":        1800,
                "history":       1200,
                "corpus":        0,
                "identity":      200,
                "interoception": 350,
            }
        else:
            budget = {
                "history":       2500,
                "corpus":        800 if has_corpus else 0,
                "identity":      300,
                "interoception": 450,
                "search":        0,
            }

        # Early turns: boost identity so the model knows who Ian is
        if turn_count < 3:
            budget["identity"] = max(budget.get("identity", 0), 500)
            budget["history"]  = min(budget.get("history",  0), 800)

        # Always reserve space for the system prompt
        budget["system"] = 400

        # Scale proportionally if over budget
        total = sum(budget.values())
        if total > self.max_chars:
            scale  = self.max_chars / total
            budget = {k: int(v * scale) for k, v in budget.items()}

        return budget

    def truncate_to_budget(self, text: str, budget: int) -> str:
        """
        Truncate text to budget chars, keeping the beginning and the end
        (middle-out truncation preserves opening context + recent content).
        """
        if not text or budget <= 0:
            return ""
        if len(text) <= budget:
            return text
        _marker = "\n[... truncated ...]\n"
        half = max(0, (budget - len(_marker)) // 2)
        return text[:half] + _marker + text[-half:]

    def format_bar_html(self, budgets: dict) -> str:
        """
        Render a compact visual bar for the medulla showing context allocation.
        Returns an HTML snippet; colours match the app CSS variable system.
        """
        colors = {
            "system":        "#555",
            "identity":      "#a855f7",
            "history":       "#4ade80",
            "corpus":        "#ffa94d",
            "search":        "#ff3366",
            "interoception": "#38bdf8",
        }
        total = max(sum(budgets.values()), 1)
        parts = []
        for key, chars in budgets.items():
            if chars <= 0:
                continue
            pct   = (chars / total) * 100
            color = colors.get(key, "#444")
            parts.append(
                f'<span style="display:inline-block;height:4px;width:{pct:.1f}%;'
                f'background:{color};border-radius:1px" title="{key}:{chars}ch"></span>'
            )
        bar    = "".join(parts)
        legend = " ".join(
            f'<span style="color:{colors.get(k,"#444")};font-size:.58em">'
            f'{k[0].upper()}:{v}</span>'
            for k, v in budgets.items() if v > 0
        )
        return (
            f'<div style="display:flex;gap:1px;margin:2px 0 1px;'
            f'height:4px;background:#111;border-radius:2px;overflow:hidden">'
            f'{bar}</div>'
            f'<div style="line-height:1.4">{legend}</div>'
        )
