"""
COUPLED MANIFOLD — Sleep Consolidator
======================================
Idle-time memory consolidation, modeled on biological sleep.

When the user goes idle (2 min), the system enters "sleep":
  1. Flushes 12B KV cache (Metal drops to ~5GB base weights)
  2. Runs consolidation on CPU:
     - Re-embeds un-indexed archive turns (sentence-transformers)
     - Generates extractive key sentences per turn
     - Generates abstractive session digest (BART, loaded on demand)
  3. Stores digest for next wake — tighter, more relevant context

On wake, the 12B weights are still loaded (never unloaded).
The "waking up" lines cover the normal ~13s first-token prefill.

Usage from app.py:
    from sleep_consolidator import SleepConsolidator
    consolidator = SleepConsolidator(mem, data_dir="./manifold_data")

    # On every user interaction:
    consolidator.on_activity()

    # Get staged context for prompt:
    context = consolidator.stage_context(query, budget=1500)

    # Check state:
    if consolidator.is_sleeping:
        ...
"""

import os, json, time, re, math, threading, gc
from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# EXTRACTIVE SUMMARIZER — pure Python, no model needed
# ═══════════════════════════════════════════════════════════════

def _tokenize(text: str) -> list[str]:
    return re.findall(r'\b[a-z]{2,}\b', text.lower())


def _sentence_split(text: str) -> list[str]:
    """Split text into sentences. Simple but robust."""
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if len(s.strip()) > 10]


def extractive_summary(turns: list[dict], topic_words: list[str] = None,
                       sentences_per_turn: int = 2) -> str:
    """
    Pick the most informative sentences from each turn.
    Scores by: sentence length + topic word overlap + position (first/last bias).
    """
    if not turns:
        return ""

    # Build topic vocabulary from all turns if not provided
    if not topic_words:
        all_text = " ".join(t.get("content", "") for t in turns)
        words = _tokenize(all_text)
        # Simple TF: most frequent non-stop words
        freq = {}
        stop = {"the", "and", "is", "in", "to", "of", "it", "that", "this",
                "for", "was", "are", "with", "on", "as", "at", "be", "have",
                "from", "or", "an", "but", "not", "you", "all", "can", "had",
                "her", "his", "one", "our", "out", "they", "been", "has",
                "its", "just", "like", "more", "about", "would", "what", "your",
                "some", "them", "than", "other", "into", "could", "time", "very",
                "when", "come", "made", "after", "back", "only", "me", "my", "do"}
        for w in words:
            if w not in stop and len(w) > 2:
                freq[w] = freq.get(w, 0) + 1
        topic_words = sorted(freq, key=freq.get, reverse=True)[:20]

    topic_set = set(topic_words)
    key_sentences = []

    for turn in turns:
        content = turn.get("content", "")
        role = turn.get("role", "")
        sents = _sentence_split(content)
        if not sents:
            continue

        scored = []
        for i, sent in enumerate(sents):
            words = set(_tokenize(sent))
            # Score: topic overlap + length penalty + position bias
            topic_score = len(words & topic_set) / max(len(topic_set), 1)
            length_score = min(len(sent) / 200, 1.0)  # prefer medium-length
            position_score = 0.3 if i == 0 else (0.2 if i == len(sents) - 1 else 0)
            score = topic_score * 0.5 + length_score * 0.2 + position_score * 0.3
            scored.append((score, sent))

        scored.sort(key=lambda x: x[0], reverse=True)
        label = "Ian" if role == "user" else "Graceful"
        for _, sent in scored[:sentences_per_turn]:
            key_sentences.append(f"{label}: {sent}")

    return "\n".join(key_sentences)


# ═══════════════════════════════════════════════════════════════
# ABSTRACTIVE SUMMARIZER — BART on CPU, loaded on demand
# ═══════════════════════════════════════════════════════════════

class AbstractiveSummarizer:
    """
    Wraps a BART-family model for session digest generation.
    Loads lazily on first call, unloads on request.
    Uses distilbart-cnn-6-6 (~600MB) — good quality, reasonable size.
    Uses model/tokenizer directly (transformers 5.x dropped 'summarization' pipeline).
    Falls back to extractive if model unavailable.
    """
    MODEL_NAME = "sshleifer/distilbart-cnn-6-6"

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._available = None  # None = unchecked, True/False after first attempt

    def _load(self):
        if self._available is False:
            return False
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.MODEL_NAME, torch_dtype=torch.float32
            )
            self._model.eval()
            self._available = True
            return True
        except Exception as e:
            print(f"  [consolidator] BART load failed: {e}", flush=True)
            self._available = False
            return False

    def summarize(self, text: str, max_length: int = 120, min_length: int = 30) -> str:
        """Generate abstractive summary. Returns empty string on failure."""
        if not text or len(text) < 50:
            return ""
        if self._model is None and not self._load():
            return ""
        try:
            import torch
            # BART has a 1024 token limit — truncate input
            truncated = text[:3000]
            inputs = self._tokenizer(
                truncated, return_tensors="pt",
                max_length=1024, truncation=True
            )
            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    min_new_tokens=min_length,
                    do_sample=False,
                )
            return self._tokenizer.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            print(f"  [consolidator] summarize failed: {e}", flush=True)
            return ""

    def unload(self):
        """Free model from RAM."""
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            gc.collect()


# ═══════════════════════════════════════════════════════════════
# DREAM SUMMARY — the poetic version of consolidation
# ═══════════════════════════════════════════════════════════════

def _make_dream(digest: str, topic_words: list[str]) -> str:
    """
    Generate a brief dream text — just topic words, no raw chat leakage.
    Format: "dreaming about: grace, coupling, resistance"
    """
    if not topic_words and not digest:
        return ""
    # Filter out HTML/CSS artifacts and common junk
    _junk = {"color", "span", "text", "style", "div", "font", "background", "border",
             "padding", "margin", "display", "width", "height", "none", "solid",
             "rgba", "flex", "inline", "block", "content", "size", "weight",
             "line", "auto", "left", "right", "top", "bottom", "center",
             "hidden", "visible", "relative", "absolute", "position",
             "transition", "opacity", "transform", "scale", "ease",
             "monospace", "serif", "sans", "code", "pre", "html", "css",
             "data", "class", "title", "href", "target", "rel",
             # Common stopwords that aren't real topics
             "the", "and", "that", "this", "with", "from", "have", "has",
             "been", "were", "was", "are", "for", "not", "but", "what",
             "all", "can", "had", "her", "his", "him", "how", "its",
             "may", "new", "now", "old", "see", "way", "who", "did",
             "get", "got", "let", "say", "she", "too", "use", "yes",
             "here", "there", "just", "like", "also", "than", "then",
             "them", "they", "into", "some", "could", "other", "about",
             "which", "their", "will", "each", "make", "more", "very",
             "when", "come", "know", "take", "want", "does", "thing",
             "much", "because", "good", "give", "most", "only", "tell",
             "one", "two", "you", "your", "yeah", "okay", "sure", "well"}
    clean = [w for w in topic_words if w not in _junk and len(w) > 2]
    themes = clean[:4] if clean else ["the conversation"]
    return "dreaming about: " + ", ".join(themes)


# ═══════════════════════════════════════════════════════════════
# WAKE-UP LINES — shown in chat during wake transition
# ═══════════════════════════════════════════════════════════════

WAKE_LINES = [
    "stretching neurons...",
    "recalling where we left off...",
    "surfacing from the manifold...",
    "pulling threads back together...",
    "shaking off the static...",
    "finding the thread again...",
]

JOSTLED_LINES = [
    "mmh... wasn't done yet...",
    "already? give me a sec...",
    "roused mid-thought...",
    "jostled awake...",
    "blinking... one moment...",
    "waking up groggy...",
    "interrupted mid-dream...",
    "yeah yeah, coming to...",
]

DROWSY_LINES = [
    "getting sleepy...",
    "eyelids heavy...",
    "about to drift off...",
    "gonna nap soon...",
    "winding down...",
    "one more minute...",
]

ENTERING_LINES = [
    "drifting off...",
    "lights out...",
    "going to bed...",
    "falling asleep...",
    "goodnight...",
]

SLEEP_LINES = [
    "consolidating what we discussed...",
    "entering dream cycle...",
    "compressing memories...",
    "running the dream manifold...",
    "deep in the manifold...",
]


# ═══════════════════════════════════════════════════════════════
# SLEEP CONSOLIDATOR — main class
# ═══════════════════════════════════════════════════════════════

class SleepConsolidator:
    """
    Manages idle-time memory consolidation.

    States:
      AWAKE       — user active, timers counting
      DROWSY      — 1 min warning before sleep ("getting sleepy...")
      ENTERING    — falling asleep, consolidation starting
      SLEEPING    — consolidating/dreaming, forced minimum 30s
      WAKING      — user returned, showing wake-up lines

    The 12B model weights stay in Metal memory at all times.
    Only the KV cache is flushed on sleep entry.
    """

    IDLE_MIN          = 120.0   # 2 min — minimum (recommended)
    IDLE_MAX          = 420.0   # 7 min — maximum (memory management needs this cap)
    IDLE_DEFAULT      = 120.0
    DROWSY_WARN       = 60.0    # warn this many seconds before sleep
    MIN_SLEEP_SECS    = 30.0    # forced minimum sleep for consolidation
    DIGEST_FILE       = "session_digest.json"

    def __init__(self, memory, data_dir: str = "./manifold_data", idle_timeout: float = 120.0):
        self.memory = memory
        self.data_dir = Path(data_dir)
        self._digest_path = self.data_dir / "sessions" / self.DIGEST_FILE
        self._state = "awake"  # awake | drowsy | entering | sleeping | waking
        self._drowsy_timer: Optional[threading.Timer] = None
        self._sleep_timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()
        self._summarizer = AbstractiveSummarizer()
        self._last_activity = time.time()
        self._has_interacted = False  # don't sleep until user has chatted at least once
        self._idle_timeout = max(self.IDLE_MIN, min(self.IDLE_MAX, idle_timeout))
        self._sleep_entered_at: Optional[float] = None  # when consolidation actually started
        self._consolidation_count = 0
        self._dream_text = ""
        self._topic_words: list[str] = []
        self._jostled = False  # True if woken during forced consolidation
        # Load previous digest if exists
        self._digest = self._load_digest()
        self._on_sleep_callbacks: list = []   # called when entering sleep
        self._on_wake_callbacks: list = []    # called when waking

    # ── State properties ───────────────────────────────────────

    @property
    def is_sleeping(self) -> bool:
        return self._state in ("entering", "sleeping")

    @property
    def is_waking(self) -> bool:
        return self._state == "waking"

    @property
    def jostled(self) -> bool:
        """True if woken mid-consolidation (before MIN_SLEEP_SECS elapsed)."""
        return self._jostled

    @property
    def state(self) -> str:
        return self._state

    @property
    def dream(self) -> str:
        return self._dream_text

    @property
    def digest(self) -> str:
        return self._digest

    @property
    def idle_seconds(self) -> float:
        return time.time() - self._last_activity

    @property
    def can_wake(self) -> bool:
        """Whether the consolidator has had enough time to finish."""
        if self._state not in ("entering", "sleeping"):
            return True
        if self._sleep_entered_at is None:
            return False
        return (time.time() - self._sleep_entered_at) >= self.MIN_SLEEP_SECS

    @property
    def sleep_remaining(self) -> float:
        """Seconds left in forced sleep. 0 if wakeable."""
        if self._sleep_entered_at is None:
            return self.MIN_SLEEP_SECS
        remaining = self.MIN_SLEEP_SECS - (time.time() - self._sleep_entered_at)
        return max(0.0, remaining)

    # ── Activity tracking ──────────────────────────────────────

    def on_activity(self):
        """Called on every user interaction (chat message). Resets idle timer."""
        self._last_activity = time.time()
        self._has_interacted = True

        if self._state == "drowsy":
            # Caught the warning in time — go back to awake
            self._cancel_timers()
            self._state = "awake"
            self._reset_timers()
            return

        if self._state in ("entering", "sleeping"):
            # Always allow wake — but mark whether consolidation finished
            self._jostled = not self.can_wake
            self._state = "waking"
            self._sleep_entered_at = None
            self._summarizer.unload()
            self._reset_timers()
            for cb in self._on_wake_callbacks:
                try:
                    cb()
                except Exception:
                    pass
            return

        # Normal awake state
        self._state = "awake"
        self._reset_timers()

    def mark_awake(self):
        """Explicitly mark as fully awake (after wake-up lines shown)."""
        self._state = "awake"
        self._dream_text = ""
        self._jostled = False

    def set_timeout(self, seconds: float):
        """Update idle timeout (clamped to 2-7 min)."""
        self._idle_timeout = max(self.IDLE_MIN, min(self.IDLE_MAX, seconds))
        if self._state == "awake" and self._has_interacted:
            self._reset_timers()

    @property
    def idle_timeout(self) -> float:
        return self._idle_timeout

    def ping(self):
        """
        Lightweight activity signal — mouse movement, scrolling, etc.
        Resets the idle timer but does NOT wake from sleep.
        Clears drowsy state if user is still around.
        """
        if self._state == "awake":
            self._last_activity = time.time()
            if self._has_interacted:
                self._reset_timers()
        elif self._state == "drowsy":
            # Mouse moved during drowsy — user is still here, cancel drowsy
            self._last_activity = time.time()
            self._cancel_timers()
            self._state = "awake"
            self._reset_timers()

    def _cancel_timers(self):
        with self._lock:
            if self._drowsy_timer is not None:
                self._drowsy_timer.cancel()
                self._drowsy_timer = None
            if self._sleep_timer is not None:
                self._sleep_timer.cancel()
                self._sleep_timer = None

    def _reset_timers(self):
        self._cancel_timers()
        if not self._has_interacted:
            return
        with self._lock:
            # Drowsy warning fires DROWSY_WARN seconds before sleep
            drowsy_delay = max(1.0, self._idle_timeout - self.DROWSY_WARN)
            self._drowsy_timer = threading.Timer(drowsy_delay, self._go_drowsy)
            self._drowsy_timer.daemon = True
            self._drowsy_timer.start()
            # Sleep fires at the full timeout
            self._sleep_timer = threading.Timer(self._idle_timeout, self._enter_sleep)
            self._sleep_timer.daemon = True
            self._sleep_timer.start()

    def trigger_sleep(self):
        """Manual sleep trigger — user pressed the sleep button."""
        if self._state in ("entering", "sleeping"):
            return  # already asleep
        self._cancel_timers()
        self._has_interacted = True  # allow sleep even if no chat yet
        # Skip drowsy, go straight to entering
        self._enter_sleep()

    # ── Drowsy (1 min warning) ─────────────────────────────────

    def _go_drowsy(self):
        if not self._has_interacted:
            return
        if self._state != "awake":
            return
        self._state = "drowsy"
        print(f"  😴 Getting drowsy (idle {self.idle_seconds:.0f}s, sleeping in ~{self.DROWSY_WARN:.0f}s)", flush=True)

    # ── Sleep entry ────────────────────────────────────────────

    def _enter_sleep(self):
        """Transition to sleep state. Runs consolidation in background."""
        if not self._has_interacted:
            return
        self._state = "entering"
        self._sleep_entered_at = time.time()
        print(f"  💤 Entering sleep mode (idle {self.idle_seconds:.0f}s)", flush=True)

        for cb in self._on_sleep_callbacks:
            try:
                cb()
            except Exception:
                pass

        # Run consolidation in background thread
        t = threading.Thread(target=self._consolidate, daemon=True)
        t.start()

    # ── Consolidation (the dream cycle) ────────────────────────

    def _consolidate(self):
        """
        The dream cycle:
        1. Flush GPU cache
        2. Back-fill archive embeddings
        3. Extractive summary of recent turns
        4. Abstractive digest via BART
        5. Store results
        """
        t0 = time.time()
        try:
            # 1. Flush GPU cache to free Metal memory
            try:
                import mlx.core as mx
                mx.synchronize()
                mx.clear_cache()
                print("  💤 KV cache flushed", flush=True)
            except Exception:
                pass

            # 2. Back-fill archive embeddings (sentence-transformers)
            try:
                self.memory.history._boot_index()
                print(f"  💤 Archive index: {self.memory.history.index.size()} turns embedded", flush=True)
            except Exception as e:
                print(f"  💤 Archive index error: {e}", flush=True)

            # 3. Gather recent turns for summarization
            recent = self.memory.history.recent
            if len(recent) < 2:
                self._state = "sleeping"
                return

            # Build topic words from recent conversation
            all_text = " ".join(t.get("content", "") for t in recent)
            words = _tokenize(all_text)
            freq = {}
            stop = {"the", "and", "is", "in", "to", "of", "it", "that", "this",
                    "for", "was", "are", "with", "on", "as", "at", "be", "have",
                    "from", "or", "an", "but", "not", "you", "all", "can", "had",
                    "her", "his", "one", "our", "out", "they", "been", "has",
                    "its", "just", "like", "more", "about", "would", "what", "your",
                    "some", "them", "than", "other", "into", "could", "time", "very",
                    "when", "come", "made", "after", "back", "only", "me", "my", "do"}
            for w in words:
                if w not in stop and len(w) > 2:
                    freq[w] = freq.get(w, 0) + 1
            self._topic_words = sorted(freq, key=freq.get, reverse=True)[:20]

            # 4. Extractive summary
            extractive = extractive_summary(recent, self._topic_words, sentences_per_turn=2)
            print(f"  💤 Extractive summary: {len(extractive)} chars", flush=True)

            # 5. Abstractive digest via BART (loaded on demand, CPU only)
            abstractive = ""
            # Format input for BART: conversational text
            conv_text = ""
            for t in recent[-10:]:  # last 10 turns max
                role = "Ian" if t.get("role") == "user" else "Graceful"
                content = t.get("content", "")[:300]
                conv_text += f"{role}: {content}\n"

            if conv_text:
                print("  💤 Loading summarizer...", flush=True)
                abstractive = self._summarizer.summarize(conv_text, max_length=150, min_length=40)
                if abstractive:
                    print(f"  💤 Abstractive digest: {len(abstractive)} chars", flush=True)
                # Unload BART immediately to free CPU RAM
                self._summarizer.unload()
                import gc; gc.collect()

            # 6. Compose final digest — abstractive preferred, extractive fallback
            self._digest = abstractive if abstractive else extractive[:500]

            # 7. Dream text
            self._dream_text = _make_dream(self._digest, self._topic_words)

            # 8. Persist
            self._save_digest()
            self._consolidation_count += 1

            elapsed = time.time() - t0
            print(f"  💤 Consolidation complete ({elapsed:.1f}s) — sleeping", flush=True)
            if self._dream_text:
                print(f"  💤 Dream: {self._dream_text[:100]}", flush=True)

        except Exception as e:
            print(f"  💤 Consolidation error: {e}", flush=True)
        finally:
            # Free any residual consolidation memory (BART, embeddings, etc.)
            import gc; gc.collect()
            try:
                import mlx.core as mx
                mx.clear_cache()
            except Exception:
                pass
            if self._state == "entering":
                self._state = "sleeping"

    # ── Digest persistence ─────────────────────────────────────

    def _save_digest(self):
        try:
            self._digest_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "digest": self._digest,
                "dream": self._dream_text,
                "topic_words": self._topic_words,
                "ts": time.time(),
                "consolidation_count": self._consolidation_count,
            }
            self._digest_path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def _load_digest(self) -> str:
        try:
            if self._digest_path.exists():
                data = json.loads(self._digest_path.read_text())
                self._dream_text = data.get("dream", "")
                self._topic_words = data.get("topic_words", [])
                self._consolidation_count = data.get("consolidation_count", 0)
                return data.get("digest", "")
        except Exception:
            pass
        return ""

    # ── Staged context retrieval ───────────────────────────────

    def stage_context(self, query: str, budget: int = 1500,
                      identity_block: str = "",
                      corpus_block: str = "",
                      intero_block: str = "",
                      search_block: str = "") -> str:
        """
        Staged memory retrieval — returns only what's needed for this query.

        L0: Identity (always)                     ~200 chars
        L1: Session digest (always, if available)  ~300 chars
        L2: Semantic recall (if relevant)          ~500 chars
        L3: Live context (if provided)             remaining budget

        Returns a single context string, budget-aware.
        """
        parts = []
        used = 0

        # L0: Identity — always present
        if identity_block:
            l0 = identity_block[:min(300, budget // 4)]
            parts.append(l0)
            used += len(l0)

        # L1: Session digest — first line of defense
        if self._digest and used < budget:
            remaining = budget - used
            digest_cap = min(400, remaining // 2)
            l1 = f"[SESSION CONTEXT — consolidated summary]\n{self._digest[:digest_cap]}"
            parts.append(l1)
            used += len(l1)

        # L2: Semantic recall from archive — only if query is substantive
        if used < budget and len(query.split()) > 3:
            remaining = budget - used
            recall_cap = min(500, remaining // 2)
            try:
                recalled = self.memory.history.index.search(query, k=3, skip_last=5)
                if recalled:
                    recall_lines = []
                    char_count = 0
                    for t in recalled:
                        role = "Ian" if t.get("role") == "user" else "Graceful"
                        line = f"{role}: {t.get('content', '')[:200]}"
                        if char_count + len(line) > recall_cap:
                            break
                        recall_lines.append(line)
                        char_count += len(line)
                    if recall_lines:
                        l2 = "[RECALLED — relevant from memory]\n" + "\n".join(recall_lines)
                        parts.append(l2)
                        used += len(l2)
            except Exception:
                pass

        # L3: Live context (corpus, search, intero) — only if budget remains
        if used < budget:
            remaining = budget - used
            live_parts = []
            for block, label in [(corpus_block, "corpus"), (search_block, "search"),
                                 (intero_block, "intero")]:
                if block and remaining > 100:
                    cap = min(len(block), remaining)
                    live_parts.append(block[:cap])
                    remaining -= cap
            if live_parts:
                l3 = "\n\n".join(live_parts)
                parts.append(l3)
                used += len(l3)

        return "\n\n".join(parts) if parts else ""

    # ── Callback registration ──────────────────────────────────

    def on_sleep(self, callback):
        """Register a callback for sleep entry (e.g., SSE push)."""
        self._on_sleep_callbacks.append(callback)

    def on_wake(self, callback):
        """Register a callback for wake-up (e.g., SSE push)."""
        self._on_wake_callbacks.append(callback)
