"""Shared embedding singleton with prototype sentences — Plan Phase 1.

Provides semantic scoring infrastructure that replaces keyword heuristics.
Lazy-loaded: zero startup cost. Falls back gracefully if sentence-transformers
is unavailable.
"""

import re
import threading

import numpy as np

_instance = None
_lock = threading.Lock()
_prototypes_cache: dict = {}


def get_shared_embedder():
    """Lazy singleton — returns the same embedder used by memory.py."""
    global _instance
    if _instance is not None:
        return _instance
    with _lock:
        if _instance is None:
            from memory import get_embedder
            _instance = get_embedder()
    return _instance


def _has_embed_method(embedder) -> bool:
    """Check if embedder supports real vector embeddings (not TF-IDF dicts)."""
    if embedder is None:
        return False
    test = embedder.embed("test")
    return hasattr(test, '__len__') and not isinstance(test, dict)


# ═══════════════════════════════════════════════════
# NEGATION DETECTION
# ═══════════════════════════════════════════════════

_NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "without",
    "hardly", "barely", "scarcely", "none", "nothing",
    "nobody", "nowhere", "isn't", "aren't", "wasn't",
    "weren't", "won't", "wouldn't", "shouldn't", "couldn't",
    "doesn't", "don't", "didn't", "hasn't", "haven't",
    "hadn't", "can't", "cannot", "mustn't",
}

# Contractions that embed negation
_CONTRACTION_NEG = re.compile(r"\b\w+n[''']t\b", re.IGNORECASE)


def has_negation(text: str) -> bool:
    """Check if text contains negation."""
    words = set(text.lower().split())
    if words & _NEGATION_WORDS:
        return True
    if _CONTRACTION_NEG.search(text):
        return True
    return False


def has_negation_near(text: str, target_words: set, window: int = 4) -> bool:
    """Check if negation appears within `window` words before any target word."""
    tokens = text.lower().split()
    for i, tok in enumerate(tokens):
        clean = tok.strip(".,!?;:'\"")
        if clean in target_words:
            # Look back up to `window` tokens for negation
            start = max(0, i - window)
            preceding = set(t.strip(".,!?;:'\"") for t in tokens[start:i])
            if preceding & _NEGATION_WORDS:
                return True
            # Check contractions in the window
            window_text = " ".join(tokens[start:i])
            if _CONTRACTION_NEG.search(window_text):
                return True
    return False


# ═══════════════════════════════════════════════════
# PROTOTYPE SENTENCES
# ═══════════════════════════════════════════════════

# OCEAN — designed for orthogonality (each dimension's prototypes
# should be semantically distinct from other dimensions)

OCEAN_PROTOTYPES = {
    "O_high": [
        "I want to explore unconventional philosophical ideas",
        "Let's try an experimental approach nobody has considered",
        "I'm fascinated by abstract theoretical frameworks",
        "What if we reimagined this from a completely different angle",
        "I love brainstorming creative and novel possibilities",
        "This reminds me of an intriguing hypothetical scenario",
        "I want to think about this more imaginatively",
    ],
    "O_low": [
        "Let's stick with the proven standard approach",
        "I prefer practical straightforward solutions",
        "What's the conventional way to handle this",
        "Keep it simple and traditional please",
        "I'd rather use something reliable and well-tested",
        "No need to overthink it just do what normally works",
    ],
    "C_high": [
        "I need to organize this into a detailed step-by-step plan",
        "Let me create a precise checklist with deadlines",
        "Can you verify every detail carefully and thoroughly",
        "I want to structure this systematically and methodically",
        "Please review this with extreme accuracy and precision",
        "Let me schedule and prioritize these tasks properly",
    ],
    "C_low": [
        "Whatever works is fine I don't need a plan",
        "Just wing it and figure it out as we go",
        "Roughly good enough is fine not everything needs to be exact",
        "I'll get around to it eventually no rush",
        "Meh close enough let's just move on already",
        "Don't overthink the details let's just try something",
    ],
    "E_high": [
        "This is so exciting I can't wait to tell everyone about it",
        "Let's get the whole team together to discuss and collaborate",
        "I absolutely love working with other people on this",
        "Wow this is amazing I'm thrilled about this idea",
        "Let's share this with the community and get everyone involved",
        "I feel so energized and enthusiastic right now",
    ],
    "E_low": [
        "I prefer working on this alone and quietly",
        "I'd rather not discuss this with others right now",
        "Let me think about this by myself in private",
        "I'm more of a solo independent worker honestly",
        "I need some quiet time to process this on my own",
        "I'd rather keep this to myself for now",
    ],
    "A_high": [
        "Thank you so much I really appreciate your patience and help",
        "I think we can find a fair compromise that works for everyone",
        "I understand your perspective and I agree with your reasoning",
        "Sorry about that let me cooperate and work with you on this",
        "You make a great point and I want to support your approach",
        "I empathize with that situation and want to be helpful",
    ],
    "A_low": [
        "That is completely wrong and I strongly disagree",
        "This is unacceptable and I refuse to go along with it",
        "What a ridiculous and absurd suggestion honestly",
        "I demand this be fixed immediately no excuses",
        "That's a terrible idea and I reject it entirely",
        "No that doesn't work at all and I won't compromise on this",
    ],
    "N_high": [
        "I'm really worried and anxious about this situation",
        "I feel so stressed and overwhelmed I don't know what to do",
        "This makes me nervous and afraid of what might happen",
        "I'm frustrated and angry that nothing is working right",
        "I feel lost and hopeless like I'm completely stuck",
        "Everything is falling apart and I'm panicking",
    ],
    "N_low": [
        "I feel calm and relaxed about this whole situation",
        "Everything is fine and I'm quite confident it will work out",
        "I'm comfortable and secure with how things are going",
        "No worries at all I'm feeling steady and stable",
        "I'm at peace with this decision and feel good about it",
        "This doesn't bother me I'm feeling assured and peaceful",
    ],
}

MOOD_PROTOTYPES = {
    "positive": [
        "This is wonderful and I'm really happy with the result",
        "Great job I love how this turned out",
        "I'm satisfied and pleased with this progress",
        "Awesome work thank you this is exactly what I needed",
    ],
    "negative": [
        "This is terrible and I'm very disappointed",
        "I hate how this turned out it's completely wrong",
        "This is frustrating and nothing is working properly",
        "I'm angry and upset about this whole situation",
    ],
    "anxious": [
        "I'm worried this won't work and I'm running out of time",
        "This is stressful and I'm nervous about the deadline",
        "I'm afraid something will go wrong with this approach",
        "I feel overwhelmed and don't know where to start",
    ],
    "curious": [
        "How does this actually work under the hood",
        "I wonder what would happen if we tried a different way",
        "Can you explain the reasoning behind this approach",
        "I'm interested in understanding the deeper mechanism here",
    ],
    "focused": [
        "Fix the bug on line 42",
        "Change the color to blue",
        "Run the tests and show me the output",
        "Update the config file with these values",
    ],
    "playful": [
        "Haha that's hilarious I love it",
        "Lol this is so funny and ridiculous",
        "Just kidding around having fun with this",
        "This is wild and entertaining let's keep going",
    ],
    "confused": [
        "I don't understand what you mean by that at all",
        "Wait what I'm completely lost here",
        "This doesn't make sense can you explain differently",
        "I have no idea what's going on help me understand",
    ],
}

PROFANITY_PROTOTYPES = {
    "intensifier": [
        "Holy shit that's amazing",
        "Damn this is really good work",
        "Fucking brilliant I love it",
        "Hell yes this is exactly right",
        "Oh shit that actually worked perfectly",
    ],
    "negative_emotion": [
        "This is complete shit and totally broken",
        "What the fuck is wrong with this garbage",
        "Damn it nothing works this is awful",
        "This fucking sucks I hate it",
        "Hell no this is terrible and unacceptable",
    ],
}

CONFIDENCE_PROTOTYPES = {
    "hedge": [
        "I think this might possibly be correct but I'm not sure",
        "If I recall correctly this could be the answer maybe",
        "This is probably approximately right but don't quote me",
        "I believe this seems like it might work but I could be wrong",
        "It appears that this is roughly what you need perhaps",
    ],
    "confident": [
        "This is definitively the correct answer without any doubt",
        "The solution is exactly this and I'm completely certain",
        "This is clearly and precisely what needs to happen",
        "Without question this is the right approach guaranteed",
        "I know for sure this is the answer specifically",
    ],
}


# ═══════════════════════════════════════════════════
# PROTOTYPE EMBEDDING CACHE
# ═══════════════════════════════════════════════════

def _get_prototype_embeddings(embedder, category: str, prototypes: dict) -> dict:
    """Get or compute cached prototype embeddings for a category."""
    cache_key = f"{category}_{id(embedder)}"
    if cache_key in _prototypes_cache:
        return _prototypes_cache[cache_key]

    result = {}
    for label, sentences in prototypes.items():
        vecs = [embedder.embed(s) for s in sentences]
        result[label] = np.array(vecs)

    _prototypes_cache[cache_key] = result
    return result


def _cosine_sim(a, b) -> float:
    """Cosine similarity between two vectors."""
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-9:
        return 0.0
    return float(np.dot(a, b) / norm)


def _max_sim_to_group(vec, group_vecs) -> float:
    """Max cosine similarity between vec and a group of vectors."""
    if len(group_vecs) == 0:
        return 0.0
    sims = np.dot(group_vecs, vec)
    norms = np.linalg.norm(group_vecs, axis=1) * np.linalg.norm(vec)
    norms = np.maximum(norms, 1e-9)
    cosines = sims / norms
    return float(np.max(cosines))


def _mean_sim_to_group(vec, group_vecs) -> float:
    """Mean cosine similarity between vec and a group of vectors."""
    if len(group_vecs) == 0:
        return 0.0
    sims = np.dot(group_vecs, vec)
    norms = np.linalg.norm(group_vecs, axis=1) * np.linalg.norm(vec)
    norms = np.maximum(norms, 1e-9)
    cosines = sims / norms
    return float(np.mean(cosines))


# ═══════════════════════════════════════════════════
# SCORING FUNCTIONS
# ═══════════════════════════════════════════════════

def score_ocean_embedding(text: str, embedder) -> dict:
    """Score OCEAN dimensions using sentence embeddings.

    Returns dict with dimension keys O/C/E/A/N, values 0-1 or None.
    """
    if not text.strip():
        return {d: None for d in "OCEAN"}

    protos = _get_prototype_embeddings(embedder, "ocean", OCEAN_PROTOTYPES)

    # Split into sentences for finer-grained analysis
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 3]
    if not sentences:
        sentences = [text]

    dim_scores = {}
    for dim in "OCEAN":
        high_key = f"{dim}_high"
        low_key = f"{dim}_low"
        if high_key not in protos or low_key not in protos:
            dim_scores[dim] = None
            continue

        high_vecs = protos[high_key]
        low_vecs = protos[low_key]

        sent_scores = []
        for sent in sentences:
            vec = embedder.embed(sent)
            negated = has_negation(sent)

            high_sim = _mean_sim_to_group(vec, high_vecs)
            low_sim = _mean_sim_to_group(vec, low_vecs)

            # If negation detected, flip the interpretation
            if negated:
                high_sim, low_sim = low_sim, high_sim

            total = high_sim + low_sim
            if total < 0.01:
                continue
            score = high_sim / total
            sent_scores.append(score)

        if not sent_scores:
            dim_scores[dim] = None
        else:
            dim_scores[dim] = sum(sent_scores) / len(sent_scores)

    return dim_scores


def score_mood_embedding(text: str, embedder) -> dict:
    """Score mood categories using sentence embeddings.

    Returns dict with mood category keys, values are similarity scores.
    """
    if not text.strip():
        return {}

    protos = _get_prototype_embeddings(embedder, "mood", MOOD_PROTOTYPES)
    vec = embedder.embed(text)

    scores = {}
    for mood, group_vecs in protos.items():
        scores[mood] = _mean_sim_to_group(vec, group_vecs)

    return scores


def classify_profanity_context(text: str, embedder) -> str:
    """Classify profanity usage: 'intensifier', 'negative_emotion', or 'ambiguous'.

    Only call when profanity is detected in text.
    """
    protos = _get_prototype_embeddings(embedder, "profanity", PROFANITY_PROTOTYPES)
    vec = embedder.embed(text)

    int_sim = _mean_sim_to_group(vec, protos["intensifier"])
    neg_sim = _mean_sim_to_group(vec, protos["negative_emotion"])

    diff = int_sim - neg_sim
    if diff > 0.03:
        return "intensifier"
    elif diff < -0.03:
        return "negative_emotion"
    return "ambiguous"


def score_confidence_embedding(text: str, embedder) -> dict:
    """Score confidence using sentence embeddings.

    Returns dict with 'hedge_sim' and 'confident_sim' values.
    """
    if not text.strip():
        return {"hedge_sim": 0.0, "confident_sim": 0.0}

    protos = _get_prototype_embeddings(embedder, "confidence", CONFIDENCE_PROTOTYPES)

    # Per-sentence scoring with negation awareness
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 3]
    if not sentences:
        sentences = [text]

    hedge_scores = []
    confident_scores = []

    for sent in sentences:
        vec = embedder.embed(sent)
        negated = has_negation(sent)

        h_sim = _mean_sim_to_group(vec, protos["hedge"])
        c_sim = _mean_sim_to_group(vec, protos["confident"])

        if negated:
            h_sim, c_sim = c_sim, h_sim

        hedge_scores.append(h_sim)
        confident_scores.append(c_sim)

    return {
        "hedge_sim": sum(hedge_scores) / len(hedge_scores),
        "confident_sim": sum(confident_scores) / len(confident_scores),
    }
