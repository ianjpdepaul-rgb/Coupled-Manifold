"""
COUPLED MANIFOLD — Intent Router
==================================
Two-tier query classifier for search routing.

Tier 1 (fast): regex-based, handles obvious cases in <1ms.
Tier 2 (slow): model-based, handles ambiguous messages via JSON classification.

Intent classes:
  factual        → search web + corpus
  recommendation → search corpus/memory only (relationship patterns)
  analysis       → no search, work from context
  fact_check     → search authoritative/academic sources only
  current_event  → search news sources only
  personal       → search session memory only
  meta           → about the conversation itself, no search
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

# ── Intent definitions ────────────────────────────────────────────────────────

INTENTS = ("factual", "recommendation", "analysis", "fact_check",
           "current_event", "personal", "meta")

# Maps intent → which search sources to use (None = no search)
INTENT_SEARCH_STRATEGY: dict[str, list[str] | None] = {
    "factual":       ["brave", "wikipedia", "ddg"],
    "recommendation": None,       # corpus/memory only — no web search
    "analysis":      None,        # no search, work from context
    "fact_check":    ["openalex", "arxiv", "semantic_scholar", "crossref", "pubmed", "wikipedia"],
    "current_event": ["brave_dated", "hackernews", "ddg"],
    "personal":      None,        # session memory only — no web search
    "meta":          None,        # about conversation, no search
}


@dataclass
class IntentDecision:
    """Result of classifying a user message."""
    intent: str
    confidence: str            # "high" (fast path) or "model" (slow path)
    reason: str                # human-readable explanation
    should_search: bool
    search_sources: list[str] | None
    timestamp: float = field(default_factory=time.time)
    geometric_override: bool = False   # True if geometric state influenced decision

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "reason": self.reason,
            "should_search": self.should_search,
            "search_sources": self.search_sources,
            "timestamp": self.timestamp,
            "geometric_override": self.geometric_override,
        }


# ── Fast path (regex) ─────────────────────────────────────────────────────────

def _classify_fast(msg: str) -> IntentDecision | None:
    """Tier 1: regex-based classification for obvious cases.

    Returns an IntentDecision if confident, None if ambiguous (→ slow path).
    """
    m = msg.strip()
    ml = m.lower()

    # Commands → meta
    if ml.startswith("/"):
        # /search and /find are explicit search intents
        if ml.startswith("/search ") or ml.startswith("/find "):
            return IntentDecision(
                intent="factual", confidence="high",
                reason="explicit search command",
                should_search=True,
                search_sources=INTENT_SEARCH_STRATEGY["factual"],
            )
        return IntentDecision(
            intent="meta", confidence="high",
            reason="slash command",
            should_search=False, search_sources=None,
        )

    # Explicit search intent
    _explicit = [
        r'^search\s+(for\s+)?',
        r'\b(look up|find me|search for|google)\b',
    ]
    for pattern in _explicit:
        if re.search(pattern, ml):
            return IntentDecision(
                intent="factual", confidence="high",
                reason="explicit search intent",
                should_search=True,
                search_sources=INTENT_SEARCH_STRATEGY["factual"],
            )

    # Social media / video → factual (web search)  — before greetings so
    # "instagram @grace" doesn't get caught by the short-message gate
    # But only if the user is ASKING about something, not just narrating what they're doing
    if re.search(r'\b(instagram|tiktok|twitter|linkedin|youtube|reddit)\b', ml):
        _is_narrating = re.search(r'\b(watching|scrolling|saw|seen|looking at|browsing|been on)\b', ml)
        if not _is_narrating:
            return IntentDecision(
                intent="factual", confidence="high",
                reason="social/video platform query",
                should_search=True,
                search_sources=INTENT_SEARCH_STRATEGY["factual"],
            )

    # Greetings / social → meta
    _greetings = {"hi", "hello", "hey", "sup", "thanks", "thank you",
                  "ok", "cool", "nice", "got it", "yes", "no", "sure",
                  "bye", "goodbye", "good morning", "good night"}
    _follow_words = {"why", "how", "what", "explain", "continue", "go", "more"}
    if ml in _greetings or (len(ml.split()) <= 2 and not any(
            c in ml for c in "?") and not any(
            w in _follow_words for w in ml.split())):
        return IntentDecision(
            intent="meta", confidence="high",
            reason="social/greeting",
            should_search=False, search_sources=None,
        )

    # Personal / self-referential → personal
    _personal = [
        r'\b(about me|you know me|remember .*(told|said|asked)|who am i)\b',
        r'\b(my (name|job|work|project|life|feeling|mood|goal))\b',
        r'\b(i (told|said|asked|mentioned) you)\b',
    ]
    for pattern in _personal:
        if re.search(pattern, ml):
            return IntentDecision(
                intent="personal", confidence="high",
                reason="self-referential query",
                should_search=False, search_sources=None,
            )

    # Self-referential / about-the-AI → meta (must come before factual patterns)
    _self_ref = [
        r'\b(how|why|what) do you (know|think|feel|decide|determine|assess|tell|judge)\b',
        r'\bwhat do you (think|mean|know|feel|want|believe)\b',
        r'\bdo you (think|know|believe|feel|agree|remember|understand)\b',
        r'\bwhat are you\b',
        r'\bhow are you\b',
        r'\b(tell me about|describe|report on) (yourself|your\b.*(state|status|operation))',
    ]
    for pattern in _self_ref:
        if re.search(pattern, ml):
            return IntentDecision(
                intent="meta", confidence="high",
                reason="self-referential question about AI",
                should_search=False, search_sources=None,
            )

    # Meta / conversation-about → meta
    _meta = [
        r'\b(you said|you mentioned|earlier|above|last (response|message|answer))\b',
        r'\b(this conversation|our (chat|discussion|conversation))\b',
        r'\b(what did (you|we)|rephrase|say that again|repeat)\b',
    ]
    for pattern in _meta:
        if re.search(pattern, ml):
            return IntentDecision(
                intent="meta", confidence="high",
                reason="conversation-referential",
                should_search=False, search_sources=None,
            )

    # Follow-up / continuation → analysis (no search)
    _follow_up = [
        r'^(why\b|how come|what about|can you explain|could you|tell me more|go on|continue)',
        r'^explain\s*(more|further|again|that|it)?$',
        r'^(yes|no|ok|sure|right|agreed|exactly|not quite|almost)$',
    ]
    for pattern in _follow_up:
        if re.search(pattern, ml):
            return IntentDecision(
                intent="analysis", confidence="high",
                reason="follow-up/continuation",
                should_search=False, search_sources=None,
            )

    # Current events → current_event
    _current = [
        r'\b(today|yesterday|this week|this month|right now|currently)\b',
        r'\b(latest|recent|breaking|just (happened|released|launched))\b',
        r'\b(news|update|announcement)\b',
        r'\b(price|stock|weather|score)\b',
    ]
    for pattern in _current:
        if re.search(pattern, ml):
            return IntentDecision(
                intent="current_event", confidence="high",
                reason="current events signal",
                should_search=True,
                search_sources=INTENT_SEARCH_STRATEGY["current_event"],
            )

    # Fact-check / academic → fact_check
    _academic = [
        r'\b(paper|stud(y|ies)|research|journal|doi|arxiv|published)\b',
        r'\b(citation|peer.?review|methodology|findings|results)\b',
        r'\b(is it true|fact.?check|verify|accurate|according to)\b',
    ]
    _academic_hits = sum(1 for p in _academic if re.search(p, ml))
    # Single academic keyword is enough when paired with explicit search intent
    _has_explicit_search = bool(re.search(r'\b(search|look up|find)\b', ml))
    if _academic_hits >= 2 or (_academic_hits >= 1 and _has_explicit_search):
        return IntentDecision(
            intent="fact_check", confidence="high",
            reason="academic/fact-check signals",
            should_search=True,
            search_sources=INTENT_SEARCH_STRATEGY["fact_check"],
        )

    # User's domain expertise (theory/philosophy) → analysis (use corpus, no web)
    _theory = [
        r'\b(baudrillard|deleuze|foucault|derrida|heidegger|lacan|bourdieu)\b',
        r'\b(simulacra|rhizome|deterritorializ|schizoanalysis|phenomenolog)\b',
        r'\b(hessian|manifold|curvature|eigenvalue|loss landscape)\b',
    ]
    for pattern in _theory:
        if re.search(pattern, ml):
            return IntentDecision(
                intent="analysis", confidence="high",
                reason="user domain expertise — use corpus",
                should_search=False, search_sources=None,
            )

    # Analytical / reasoning → analysis
    _analytical = [
        r'\b(explain|define|what does .* mean|how does .* work)\b',
        r'\b(compare|contrast|analyze|evaluate|synthesize)\b',
        r'\b(difference between|versus|vs\.?|relationship between)\b',
        r'\b(in general|theoretically|conceptually|philosophically)\b',
    ]
    for pattern in _analytical:
        if re.search(pattern, ml):
            return IntentDecision(
                intent="analysis", confidence="high",
                reason="analytical/reasoning question",
                should_search=False, search_sources=None,
            )

    # Code requests → analysis (no search unless looking for libraries)
    _code = [
        r'\b(write|code|implement|debug|fix|refactor)\b.*\b(python|javascript|code|script|function|class)\b',
        r'```',
    ]
    for pattern in _code:
        if re.search(pattern, ml):
            return IntentDecision(
                intent="analysis", confidence="high",
                reason="code request",
                should_search=False, search_sources=None,
            )

    # Code resource search (libraries, docs) → factual
    _code_resource = [
        r'\b(latest version|changelog|release notes|github|npm|pypi)\b',
        r'\b(how to install|documentation|docs for|best (library|tool|package))\b',
    ]
    for pattern in _code_resource:
        if re.search(pattern, ml):
            return IntentDecision(
                intent="factual", confidence="high",
                reason="code resource search",
                should_search=True,
                search_sources=INTENT_SEARCH_STRATEGY["factual"],
            )

    # Creative requests → analysis
    _creative = [
        r'\b(write|compose|draft|create) (a |an |me )?(story|poem|essay|song|letter|email)\b',
        r'\b(brainstorm|imagine|come up with|invent)\b',
    ]
    for pattern in _creative:
        if re.search(pattern, ml):
            return IntentDecision(
                intent="analysis", confidence="high",
                reason="creative request",
                should_search=False, search_sources=None,
            )

    # Math / computation → analysis (no search needed)
    _math = [
        r'^\s*\d+\s*[\+\-\*\/\%\^]\s*\d+',               # "2+2", "10 * 5"
        r'^what\s+is\s+\d+\s*[\+\-\*\/\%\^]',             # "what is 2+2"
        r'\b(calculate|compute|solve|simplify|evaluate)\b',
        r'\b(math|equation|formula)\b',
        r'\b\d+\s*(times|plus|minus|divided|multiplied)\b',
    ]
    for pattern in _math:
        if re.search(pattern, ml):
            return IntentDecision(
                intent="analysis", confidence="high",
                reason="math/computation — no search needed",
                should_search=False, search_sources=None,
            )

    # Direct factual questions → factual
    _factual = [
        r'^(what|who|when|where|how)\s+(is|are|was|were|did|does|do)\b',
        r'\b(tell me about|what happened)\b',
    ]
    for pattern in _factual:
        if re.search(pattern, ml):
            return IntentDecision(
                intent="factual", confidence="high",
                reason="factual question",
                should_search=True,
                search_sources=INTENT_SEARCH_STRATEGY["factual"],
            )

    # Short messages without triggers → analysis (default, no search)
    if len(ml.split()) <= 8:
        return IntentDecision(
            intent="analysis", confidence="high",
            reason="short message, no search triggers",
            should_search=False, search_sources=None,
        )

    # Ambiguous — fall through to slow path
    return None


# ── Slow path (model) ─────────────────────────────────────────────────────────

_CLASSIFY_PROMPT = """Classify this user message into exactly one intent. Reply with ONLY a JSON object, no other text.

Intents:
- factual: asking for facts, information, "what is X", lookup
- recommendation: asking for suggestions, "what should I", best practices
- analysis: reasoning, explaining, comparing, creating, coding
- fact_check: verifying a claim, asking for sources/evidence
- current_event: asking about recent news, today's events, prices
- personal: asking about themselves, their history, their data
- meta: about this conversation, rephrasing, social pleasantries

Message: {message}

JSON (intent field only):"""


def _classify_model(msg: str, generate_fn) -> IntentDecision:
    """Tier 2: model-based classification for ambiguous messages.

    generate_fn: callable(prompt, max_tokens) → str
    """
    prompt = _CLASSIFY_PROMPT.format(message=msg[:500])
    try:
        raw = generate_fn(prompt, 40)
        # Try to parse JSON from the response
        # Handle cases like ```json {...} ``` or just {...}
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r'^```\w*\s*', '', raw)
            raw = raw.rstrip("`").strip()
        match = re.search(r'\{[^}]+\}', raw)
        if match:
            parsed = json.loads(match.group())
            intent = parsed.get("intent", "").lower().strip()
            if intent in INTENTS:
                strategy = INTENT_SEARCH_STRATEGY.get(intent)
                return IntentDecision(
                    intent=intent,
                    confidence="model",
                    reason=f"model classified: {intent}",
                    should_search=strategy is not None,
                    search_sources=strategy,
                )
    except Exception:
        pass

    # Fallback on parse failure → analysis (no search, safe default)
    return IntentDecision(
        intent="analysis", confidence="model",
        reason="model classification fallback (parse failure)",
        should_search=False, search_sources=None,
    )


# ── IntentRouter ──────────────────────────────────────────────────────────────

DATA_DIR = "./manifold_data"
_LOG_PATH = f"{DATA_DIR}/logs/intent_decisions.jsonl"


class IntentRouter:
    """Routes user messages to search strategies based on intent classification.

    Usage:
        router = IntentRouter(generate_fn=model.generate)
        decision = router.classify("What is quantum entanglement?")
        if decision.should_search:
            results = search_by_intent(query, decision)
    """

    def __init__(self, generate_fn=None):
        """
        generate_fn: optional callable(prompt: str, max_tokens: int) -> str
                     If None, only the fast path is used.
        """
        self._generate_fn = generate_fn
        self._log_file = None

    def classify(self, msg: str, *, geometric_mode: str = "",
                 skip: bool = False) -> IntentDecision:
        """Classify a user message into an intent.

        Args:
            msg: the user message
            geometric_mode: current SnobLine mode (e.g. "anti", "lora")
            skip: if True, force no-search (used by /continue)
        """
        if skip:
            decision = IntentDecision(
                intent="meta", confidence="high",
                reason="search skipped (continuation turn)",
                should_search=False, search_sources=None,
            )
            self._log(msg, decision)
            return decision

        # Geometric override: if anti-LoRA is active, bias toward no-search
        # The system is already intervening — don't compound with external data
        geo_override = False
        if geometric_mode == "anti":
            geo_override = True

        # Tier 1: fast path
        decision = _classify_fast(msg)

        # Tier 2: slow path for ambiguous messages (>8 words, no fast match)
        if decision is None:
            if self._generate_fn is not None:
                decision = _classify_model(msg, self._generate_fn)
            else:
                # No model available — default to analysis
                decision = IntentDecision(
                    intent="analysis", confidence="high",
                    reason="ambiguous, no model for classification",
                    should_search=False, search_sources=None,
                )

        # Apply geometric override
        if geo_override and decision.should_search:
            decision = IntentDecision(
                intent=decision.intent,
                confidence=decision.confidence,
                reason=f"{decision.reason} [overridden: anti-LoRA active]",
                should_search=False,
                search_sources=None,
                geometric_override=True,
            )

        self._log(msg, decision)
        return decision

    def _log(self, msg: str, decision: IntentDecision):
        """Append decision to JSONL log."""
        try:
            os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
            if self._log_file is None:
                self._log_file = open(_LOG_PATH, "a", buffering=1)  # line-buffered
            record = {
                "ts": time.time(),
                "msg_preview": msg[:100],
                **decision.to_dict(),
            }
            self._log_file.write(json.dumps(record, default=str) + "\n")
        except Exception:
            pass  # logging failure must never block chat

    def close(self):
        """Flush and close the log file."""
        if self._log_file:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None
