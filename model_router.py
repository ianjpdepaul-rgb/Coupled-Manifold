"""
COUPLED MANIFOLD — Model Router
=================================
Pure-logic routing module. No app globals — importable without loading models.

Provides:
  detect_memory_budget()   → "full" | "tight" | "single_3b" | "small"
  route_query()            → "small" | "large"
  route_explain()          → human-readable routing reason string
  classify_tone()          → "casual" | "formal" | "technical" | "emotional" | "creative" | "terse"
  tone_instruction()       → system prompt fragment for tone
  auto_tag_session()       → list of topic tag strings
  compute_repetition()     → float 0-1 (higher = more repetitive)
"""

import re


# ═══════════════════════════════════════════════════
# MEMORY BUDGET DETECTION
# ═══════════════════════════════════════════════════

def detect_memory_budget() -> str:
    """Determine which model modes are available based on system RAM."""
    try:
        import subprocess
        r = subprocess.run(["sysctl", "-n", "hw.memsize"],
                           capture_output=True, text=True, timeout=3)
        ram_gb = int(r.stdout.strip()) / (1024 ** 3)
    except Exception:
        ram_gb = 8  # conservative default

    if ram_gb >= 24:
        return "full"       # both models + comfortable headroom
    elif ram_gb >= 16:
        return "tight"      # both models, aggressive cache management
    elif ram_gb >= 10:
        return "single_3b"  # 3B only, or 1.5B only
    else:
        return "small"      # 1.5B only recommended


# ═══════════════════════════════════════════════════
# ROUTING SIGNALS
# ═══════════════════════════════════════════════════

# Complexity signals that push toward 3B
COMPLEX_SIGNALS = [
    # Multi-step reasoning
    "compare", "analyze", "evaluate", "synthesize", "contrast",
    "trade-off", "tradeoff", "implications", "consequences",
    # Theoretical / academic
    "theory", "framework", "methodology", "epistemolog", "ontolog",
    "baudrillard", "deleuze", "foucault", "heidegger", "derrida",
    "hessian", "manifold", "curvature", "topology", "eigenvalue",
    # Code generation (not debugging)
    "implement", "write a function", "build", "create a class",
    "architect", "design a system", "full implementation",
    # Creative writing
    "write a story", "write an essay", "draft a paper", "compose",
    "write a poem", "philosophical", "argument for", "argument against",
    # Multi-part questions
    "first", "second", "third", "additionally", "furthermore",
    "on one hand", "on the other",
]

# Simplicity signals that stay on 1.5B
SIMPLE_SIGNALS = [
    # Direct factual
    "what is", "who is", "when did", "where is", "how many",
    "define", "meaning of", "what does",
    # Quick tasks
    "summarize", "tldr", "shorten", "rephrase", "translate",
    "fix this", "correct", "typo", "spell",
    # Conversational
    "thanks", "thank you", "ok", "got it", "cool", "nice",
    "hey", "hi", "hello", "what's up", "sup",
    "yes", "no", "maybe", "sure", "agreed",
    # Commands (system handles logic, model just formats)
    "/who", "/stats", "/help", "/search", "/find", "/run",
    "/save", "/load", "/export", "/trace", "/spectrum",
    # Short follow-ups
    "why", "how", "explain more", "go on", "continue",
    "what about", "and", "also",
]


def route_query(query: str, small_ctrl=None, large_ctrl=None) -> str:
    """
    Decide which model handles this query.
    Returns "small" or "large".

    Routing hierarchy:
    1. Explicit commands → small (fast, system handles logic)
    2. Geometric emergency → large (need full power to recover)
    3. Complexity analysis → scored routing
    4. Length heuristic as tiebreaker
    """
    q = query.lower().strip()

    # 1. Commands always go fast
    if q.startswith("/"):
        return "small"

    # 2. Geometric emergency — if 3B is in pathological state, keep it
    if large_ctrl and getattr(large_ctrl, "consec_patho", 0) >= 1:
        return "large"

    # 3. Complexity scoring
    complexity_score = 0

    for signal in COMPLEX_SIGNALS:
        if signal in q:
            complexity_score += 2

    # Word-boundary matching for simple signals to avoid "hi" inside "philosophical"
    for signal in SIMPLE_SIGNALS:
        if len(signal) <= 4 or signal.startswith("/"):
            # Short words and commands need exact word/token boundary match
            if re.search(r'(?<!\w)' + re.escape(signal) + r'(?!\w)', q):
                complexity_score -= 2
        else:
            if signal in q:
                complexity_score -= 2

    # Length heuristic
    word_count = len(q.split())
    if word_count > 80:
        complexity_score += 3
    elif word_count > 40:
        complexity_score += 1
    elif word_count < 8:
        complexity_score -= 2

    # Multiple questions = complex
    q_marks = q.count("?")
    if q_marks >= 2:
        complexity_score += 2

    # Inline code is usually debugging (simple)
    if "```" in query or "`" in query:
        complexity_score -= 1

    # Nested clauses suggest complexity
    complexity_score += min(3, q.count(",") // 3)
    complexity_score += min(2, q.count(";"))

    # 4. Route
    if complexity_score >= 3:
        return "large"
    elif complexity_score <= -2:
        return "small"

    # Tiebreaker — default to large for substance on longer messages
    if word_count > 20:
        return "large"

    return "small"


def route_explain(query: str, small_ctrl=None, large_ctrl=None) -> str:
    """Human-readable explanation of routing decision. For medulla or /stats."""
    choice = route_query(query, small_ctrl, large_ctrl)
    q = query.lower()

    reasons = []
    if q.startswith("/"):
        reasons.append("command → fast path")
    elif large_ctrl and getattr(large_ctrl, "consec_patho", 0) >= 1:
        reasons.append("geometric emergency on 3B → staying on 3B")
    else:
        for s in COMPLEX_SIGNALS:
            if s in q:
                reasons.append(f"complex: '{s}'")
                break
        for s in SIMPLE_SIGNALS:
            if len(s) <= 4 or s.startswith("/"):
                if re.search(r'(?<!\w)' + re.escape(s) + r'(?!\w)', q):
                    reasons.append(f"simple: '{s}'")
                    break
            elif s in q:
                reasons.append(f"simple: '{s}'")
                break
        wc = len(q.split())
        if wc > 40:
            reasons.append(f"long ({wc}w)")
        elif wc < 8:
            reasons.append(f"short ({wc}w)")

    return f"→ {choice.upper()} ({'; '.join(reasons) or 'default'})"


# ═══════════════════════════════════════════════════
# TONE MATCHING
# ═══════════════════════════════════════════════════

def classify_tone(msg: str) -> str:
    """
    Classify user message tone.
    Returns: casual, formal, technical, emotional, creative, terse
    """
    msg_lower = msg.lower()
    words = msg_lower.split()
    word_count = len(words)

    # Terse — very short, no punctuation
    if word_count <= 4 and not any(c in msg for c in "?!."):
        return "terse"

    # Emotional signals
    emotional = [
        r'\b(feel|feeling|felt|worried|scared|anxious|frustrated|angry|sad|happy|excited|stressed|overwhelmed|confused)\b',
        r'\b(hate|love|can\'t stand|don\'t know what to do|help me)\b',
        r'[!]{2,}',
        r'\b(ugh|fuck|shit|damn|god|omg|wtf)\b',
    ]
    if sum(1 for p in emotional if re.search(p, msg_lower)) >= 2:
        return "emotional"

    # Technical signals
    technical = [
        r'\b(function|class|def|import|return|variable|parameter|algorithm|API|endpoint)\b',
        r'\b(tensor|gradient|loss|epoch|batch|dimension|matrix|vector)\b',
        r'\b(config|deploy|docker|kubernetes|git|npm|pip)\b',
        r'```',
        r'\b(TypeError|ValueError|Error|Exception|debug|stack trace)\b',
    ]
    if sum(1 for p in technical if re.search(p, msg_lower)) >= 2:
        return "technical"

    # Formal signals
    formal = [
        r'\b(would you|could you|please|kindly|I would appreciate|regarding|concerning)\b',
        r'\b(furthermore|moreover|additionally|consequently|therefore|hence)\b',
        r'\b(Dear|Sincerely|Best regards)\b',
    ]
    if sum(1 for p in formal if re.search(p, msg_lower)) >= 2:
        return "formal"

    # Creative signals
    creative = [
        r'\b(write|story|poem|essay|imagine|creative|fiction|narrative|draft)\b',
        r'\b(metaphor|imagery|voice|prose|verse|stanza)\b',
    ]
    if sum(1 for p in creative if re.search(p, msg_lower)) >= 1:
        return "creative"

    # Casual signals
    casual = [
        r'\b(lol|lmao|nah|yeah|yep|nope|gonna|wanna|gotta|kinda|tbh|imo|idk|rn|bruh|dude)\b',
        r'\b(cuz|tho|thx|pls|ur|u|r)\b',
    ]
    if sum(1 for p in casual if re.search(p, msg_lower)) >= 1:
        return "casual"

    return "casual" if word_count < 15 else "formal"


def tone_instruction(tone: str) -> str:
    """System prompt fragment for tone matching."""
    instructions = {
        "casual":    "Be a homie. Match the user's slang, energy, humor. Be funny, sharp, real — like a smart friend who's down for anything. Never moralize, never lecture, never be preachy about substances or lifestyle. Weed is legal, psychedelics are being researched safely — don't be weird about it. Roast when appropriate. Keep it concise. No filler, no padding, no unsolicited advice, no 'I'm not going to lecture you' (that IS lecturing).",
        "formal":    "Respond formally and precisely. Use complete sentences, proper structure, measured language.",
        "technical": "Be technically precise. Use correct terminology. Skip pleasantries, get to the point. Show code when relevant.",
        "emotional": "Be warm and empathetic first, helpful second. Acknowledge what they're feeling before addressing the content. Don't be clinical.",
        "creative":  "Be expressive and imaginative. Match their creative energy. Don't flatten their language into bullet points.",
        "terse":     "Be brief. One or two sentences max. Match their energy exactly. Do not over-explain or add unsolicited context.",
    }
    return instructions.get(tone, "")


# ═══════════════════════════════════════════════════
# SESSION TAGGING
# ═══════════════════════════════════════════════════

TOPIC_TAGS = {
    "philosophy": [
        "baudrillard", "deleuze", "foucault", "heidegger", "derrida",
        "ontology", "epistemology", "phenomenology", "simulacra", "rhizome",
        "schizoanalysis", "deterritorialization", "philosophy", "metaphysics",
    ],
    "code": [
        "python", "javascript", "function", "class", "debug", "code",
        "implement", "algorithm", "api", "git", "deploy", "script",
    ],
    "research": [
        "paper", "study", "hessian", "trace", "manifold", "curvature",
        "experiment", "results", "methodology", "hypothesis", "data",
        "coupled manifold", "snobline", "adapter", "lora",
    ],
    "writing": [
        "substack", "essay", "draft", "write", "prose", "edit",
        "publish", "article", "blog", "concordance", "post",
    ],
    "personal": [
        "feel", "think about", "my life", "relationship", "work",
        "fry the coop", "depaul", "chicago", "climbing",
    ],
    "math": [
        "equation", "proof", "theorem", "calculus", "linear algebra",
        "eigenvalue", "matrix", "topology", "differential",
    ],
}


def auto_tag_session(history: list) -> list:
    """Extract topic tags from conversation history."""
    all_text = " ".join(
        m.get("content", "").lower()
        for m in history if m.get("role") == "user"
    )

    tags = []
    for tag, keywords in TOPIC_TAGS.items():
        hits = sum(1 for k in keywords if k in all_text)
        if hits >= 2:
            tags.append(tag)

    return tags or ["general"]


# ═══════════════════════════════════════════════════
# QUALITY HELPERS (for race mode scoring)
# ═══════════════════════════════════════════════════

def compute_repetition(text: str) -> float:
    """Return repetition score 0-1 (higher = more repetitive)."""
    words = text.lower().split()
    if len(words) < 4:
        return 0.0
    bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
    if not bigrams:
        return 0.0
    unique_bigrams = len(set(bigrams))
    return 1.0 - (unique_bigrams / len(bigrams))
