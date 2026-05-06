"""
COUPLED MANIFOLD — Memory & Identity System
============================================
Two-layer persistent history + corpus RAG + identity model.

Usage from app.py:
    from memory import Memory
    mem = Memory(data_dir="./manifold_data")

    # On every turn:
    mem.append_turn(role, content)
    context_block = mem.build_context(current_query)

    # Index your writing once:
    mem.index_corpus("./my_notes.md")
    mem.index_corpus("./concordance.md")
    mem.index_directory("./notes/")
"""

import os, json, time, re, math, threading
from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# TINY LOCAL EMBEDDER — no external deps, pure Python TF-IDF
# Falls back to sentence-transformers if available (much better)
# ═══════════════════════════════════════════════════════════════

class TFIDFEmbedder:
    """Lightweight local embedder. No GPU, no downloads."""
    def __init__(self):
        self.vocab = {}
        self.idf   = {}
        self.docs  = []

    def _tokenize(self, text):
        return re.findall(r'\b[a-z]{2,}\b', text.lower())

    def fit(self, documents):
        self.docs = documents
        df = {}
        for doc in documents:
            for w in set(self._tokenize(doc)):
                df[w] = df.get(w, 0) + 1
        N = len(documents)
        self.idf = {w: math.log((N + 1) / (c + 1)) for w, c in df.items()}
        self.vocab = {w: i for i, w in enumerate(self.idf)}

    def embed(self, text):
        tokens = self._tokenize(text)
        tf = {}
        for w in tokens:
            tf[w] = tf.get(w, 0) + 1
        vec = {}
        for w, count in tf.items():
            if w in self.idf:
                vec[w] = (count / len(tokens)) * self.idf[w]
        norm = math.sqrt(sum(v*v for v in vec.values())) or 1.0
        return {w: v/norm for w, v in vec.items()}

    def similarity(self, a, b):
        return sum(a.get(w, 0) * b.get(w, 0) for w in a)


def get_embedder():
    """Use sentence-transformers if available, else TF-IDF."""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

        class STEmbedder:
            def __init__(self, m): self.m = m; self.docs = []; self.vecs = []
            def fit(self, docs):
                self.docs = docs
                self.vecs = self.m.encode(docs, show_progress_bar=False)
            def embed(self, text):
                return self.m.encode([text], show_progress_bar=False)[0]
            def similarity(self, a, b):
                n = (np.linalg.norm(a) * np.linalg.norm(b))
                return float(np.dot(a, b) / n) if n > 0 else 0.0

        print("  Memory: using sentence-transformers embedder")
        return STEmbedder(model)
    except ImportError:
        print("  Memory: using TF-IDF embedder (pip install sentence-transformers for better retrieval)")
        return TFIDFEmbedder()


# ═══════════════════════════════════════════════════════════════
# CORPUS — chunked document store with similarity search
# ═══════════════════════════════════════════════════════════════

class Corpus:
    def __init__(self, store_path: str):
        self.path    = Path(store_path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.chunks_file = self.path / "chunks.json"
        self.chunks  = []       # list of {text, source, chunk_id, ts}
        self.embedder = None
        self._load()

    def _load(self):
        if self.chunks_file.exists():
            try:
                self.chunks = json.loads(self.chunks_file.read_text())
            except (json.JSONDecodeError, OSError) as e:
                print(f"  Corpus: chunks.json load failed ({e}), starting fresh")
                self.chunks = []
            print(f"  Corpus: {len(self.chunks)} chunks loaded")

    def _save(self):
        self.chunks_file.write_text(json.dumps(self.chunks, indent=2))

    def _fit(self):
        if not self.chunks:
            return
        if self.embedder is None:
            self.embedder = get_embedder()
        self.embedder.fit([c["text"] for c in self.chunks])

    def _chunk(self, text: str, source: str, chunk_size=400, overlap=80):
        words  = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i+chunk_size])
            chunks.append({
                "text":     chunk,
                "source":   source,
                "chunk_id": len(self.chunks) + len(chunks),
                "ts":       time.time(),
            })
            i += chunk_size - overlap
        return chunks

    def _is_near_duplicate(self, new_text: str, threshold: float = 0.72) -> bool:
        """Check if new_text is too similar to any existing chunk (Jaccard similarity)."""
        if not self.chunks:
            return False
        new_words = set(new_text.lower().split())
        if not new_words:
            return False
        for chunk in self.chunks:
            existing_words = set(chunk.get("text", "").lower().split())
            if not existing_words:
                continue
            intersection = len(new_words & existing_words)
            union = len(new_words | existing_words)
            if union > 0 and intersection / union >= threshold:
                return True
        return False

    def _rebuild_index(self):
        """Re-fit embedder on current chunks (alias for reindex, used after manual chunk append)."""
        self._fit()

    def index(self, text: str, source: str = "manual"):
        """Add a document. Idempotent — skips if source already indexed.
        Deduplicates chunks at 0.92 Jaccard similarity threshold."""
        existing = {c["source"] for c in self.chunks}
        if source in existing:
            print(f"  Corpus: '{source}' already indexed, skipping")
            return 0
        new_chunks = self._chunk(text, source)
        added = 0
        skipped = 0
        for chunk in new_chunks:
            if self._is_near_duplicate(chunk["text"]):
                skipped += 1
            else:
                self.chunks.append(chunk)
                added += 1
        self._save()
        self._fit()
        print(f"  Corpus: indexed '{source}' → {added} chunks added, {skipped} duplicates skipped ({len(self.chunks)} total)")
        return added

    def _extract_text(self, path: Path) -> str:
        """Extract plain text from any supported file type."""
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            try:
                import fitz  # pymupdf
                doc = fitz.open(str(path))
                return "\n".join(page.get_text() for page in doc)
            except ImportError:
                try:
                    import pdfplumber
                    with pdfplumber.open(str(path)) as pdf:
                        return "\n".join(p.extract_text() or "" for p in pdf.pages)
                except ImportError:
                    print(f"  Corpus: PDF support needs pymupdf — pip install pymupdf")
                    return ""

        if suffix in (".docx", ".doc"):
            try:
                from docx import Document
                doc = Document(str(path))
                return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            except ImportError:
                print(f"  Corpus: Word support needs python-docx — pip install python-docx")
                return ""

        if suffix == ".html":
            try:
                from html.parser import HTMLParser
                class _Strip(HTMLParser):
                    def __init__(self):
                        super().__init__()
                        self.parts = []
                    def handle_data(self, d): self.parts.append(d)
                p = _Strip(); p.feed(path.read_text(errors="ignore"))
                return " ".join(p.parts)
            except Exception:
                return path.read_text(errors="ignore")

        if suffix == ".csv":
            try:
                import csv, io
                rows = list(csv.reader(io.StringIO(path.read_text(errors="ignore"))))
                return "\n".join(", ".join(row) for row in rows)
            except Exception:
                return path.read_text(errors="ignore")

        # Plaintext fallback — md, txt, json, py, etc.
        return path.read_text(errors="ignore")

    def index_file(self, path: str):
        path = Path(path)
        if not path.exists():
            print(f"  Corpus: file not found: {path}")
            return 0
        text = self._extract_text(path)
        if not text.strip():
            print(f"  Corpus: no text extracted from {path.name}")
            return 0
        return self.index(text, source=str(path.name))

    def index_directory(self, directory: str,
                        extensions=(".md", ".txt", ".json", ".py",
                                    ".pdf", ".docx", ".html", ".csv")):
        """Index all files in a directory. Fits embedder once at the end (not per file)."""
        d = Path(directory)
        if not d.exists():
            print(f"  Corpus: directory not found: {directory}")
            return
        existing = {c["source"] for c in self.chunks}
        total_added = 0
        total_skipped = 0
        files = [f for f in sorted(d.rglob("*"))
                 if f.is_file() and f.suffix.lower() in extensions
                 and str(f.name) not in existing]
        for f in files:
            text = self._extract_text(f)
            if not text.strip():
                continue
            new_chunks = self._chunk(text, source=str(f.name))
            for chunk in new_chunks:
                if self._is_near_duplicate(chunk["text"]):
                    total_skipped += 1
                else:
                    self.chunks.append(chunk)
                    total_added += 1
        if total_added > 0:
            self._save()
            self._fit()  # fit once for all files, not per file
        print(f"  Corpus: directory indexed — {total_added} chunks added, {total_skipped} duplicates skipped ({len(self.chunks)} total)")

    def _bm25_scores(self, query: str, k1: float = 1.5, b: float = 0.75) -> list[float]:
        """BM25 relevance scores for all chunks against query."""
        import re as _re
        tokens = _re.findall(r'\b[a-z]{2,}\b', query.lower())
        if not tokens or not self.chunks:
            return [0.0] * len(self.chunks)
        doc_texts = [c["text"] for c in self.chunks]
        doc_lens  = [len(_re.findall(r'\b\w+\b', t)) for t in doc_texts]
        avg_dl    = sum(doc_lens) / len(doc_lens) if doc_lens else 1.0
        N = len(self.chunks)
        # IDF per query term
        idf = {}
        for term in set(tokens):
            df = sum(1 for t in doc_texts if term in t.lower())
            idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
        scores = []
        for text, dl in zip(doc_texts, doc_lens):
            term_freq: dict = {}
            for w in _re.findall(r'\b[a-z]{2,}\b', text.lower()):
                term_freq[w] = term_freq.get(w, 0) + 1
            score = 0.0
            for term in tokens:
                tf = term_freq.get(term, 0)
                score += idf.get(term, 0) * (tf * (k1 + 1)) / (
                    tf + k1 * (1 - b + b * dl / avg_dl)
                )
            scores.append(score)
        return scores

    def search(self, query: str, k: int = 5, return_scores: bool = False) -> list:
        """Hybrid BM25 + semantic retrieval (60/40 weighted).

        If return_scores=True, returns list of (score, chunk) tuples.
        Otherwise returns list of chunk dicts (backward-compatible).
        """
        if not self.chunks:
            return []
        if self.embedder is None:
            self._fit()

        # Semantic scores
        q_vec = self.embedder.embed(query)
        sem_scores = []
        for i, chunk in enumerate(self.chunks):
            try:
                c_vec = (self.embedder.vecs[i]
                         if hasattr(self.embedder, "vecs")
                         else self.embedder.embed(chunk["text"]))
                sem_scores.append(self.embedder.similarity(q_vec, c_vec))
            except Exception:
                sem_scores.append(0.0)

        # BM25 scores
        bm25 = self._bm25_scores(query)

        # Normalize both to [0,1] then combine with meaningful recency boost
        import time as _ct
        _now = _ct.time()
        s_max  = max(sem_scores) or 1.0
        b_max  = max(bm25)       or 1.0
        combined = []
        for i, chunk in enumerate(self.chunks):
            _age_days = (_now - chunk.get("ts", _now)) / 86400.0
            _recency  = 1.0 / (1.0 + _age_days * 0.05)
            # 0.06→0.15 recency weight — makes recently indexed content rank higher
            score = 0.51 * (sem_scores[i] / s_max) + 0.34 * (bm25[i] / b_max) + 0.15 * _recency
            combined.append((score, chunk))
        combined.sort(key=lambda x: x[0], reverse=True)
        if return_scores:
            return combined[:k]
        return [c for _, c in combined[:k]]

    def reindex(self):
        """Re-fit embedder on all chunks (call after bulk imports)."""
        self._fit()
        print(f"  Corpus: reindexed {len(self.chunks)} chunks")


# ═══════════════════════════════════════════════════════════════
# IDENTITY MODEL — builds a living picture of who Ian is
# ═══════════════════════════════════════════════════════════════

IDENTITY_KEYS = [
    "name", "age", "role", "location", "education",
    "projects", "thinkers", "concepts", "voice", "goals",
    "writing_style", "recurring_themes", "tools", "raw_notes",
]

# Stop-words for concept extraction — common English words that leak into the graph
_CONCEPT_STOPWORDS = frozenset({
    "about", "after", "again", "already", "also", "always", "because", "before",
    "being", "below", "between", "border", "button", "certainly", "change", "click",
    "color", "column", "continue", "could", "currently", "display", "doing", "during",
    "element", "every", "example", "false", "first", "float", "focus", "follow",
    "found", "function", "going", "height", "immediately", "include", "index",
    "input", "inside", "instead", "issue", "itself", "large", "later", "length",
    "level", "likely", "logically", "margin", "maybe", "might", "model", "never",
    "number", "other", "output", "padding", "panel", "please", "point", "position",
    "previous", "proceed", "prompt", "question", "quite", "radius", "really",
    "reason", "regarding", "repeat", "response", "result", "right", "running",
    "saying", "seems", "should", "since", "small", "solid", "something", "sorry",
    "specific", "start", "state", "still", "string", "style", "their", "there",
    "these", "thing", "think", "those", "through", "under", "until", "using",
    "value", "where", "which", "while", "width", "within", "without", "would",
    "write", "action", "class", "event", "check", "based", "makes", "means",
    "needs", "order", "place", "given", "taken", "leave", "error", "block",
    "frame", "build", "added", "clear", "close", "whole", "looks", "hence",
    "above", "along", "break", "bring", "cause", "enter", "exist", "fully",
    "great", "heavy", "human", "image", "known", "light", "local", "lower",
    "match", "moved", "named", "noted", "often", "opens", "parts", "plain",
    "raise", "range", "reach", "refer", "rough", "scene", "sense", "serve",
    "shall", "shape", "shift", "short", "shown", "since", "space", "speak",
    "spent", "stand", "stuck", "table", "taken", "terms", "third", "times",
    "total", "touch", "track", "tried", "turns", "typed", "upper", "users",
    "valid", "watch", "words", "works", "worth", "zeros", "apply", "avoid",
    "basic", "begin", "could", "cover", "dozen", "draft", "early", "empty",
    "equal", "exact", "extra", "final", "fixed", "fresh", "guess", "happy",
    "hence", "ideal", "keeps", "later", "learn", "legal", "lines", "lived",
    "major", "meant", "minor", "mixed", "never", "occur", "offer", "outer",
    "owned", "paper", "patch", "pause", "phase", "piece", "print", "prior",
    "proof", "quick", "quite", "ready", "shows", "solve", "sorry", "still",
    "store", "stuff", "teach", "those", "title", "topic", "truly", "trust",
    "twice", "unity", "usual", "voice", "waste", "which", "whole", "worry",
    "worse", "yield", "young", "complex", "already", "another", "because",
    "between", "certain", "clearly", "contain", "correct", "current", "default",
    "defined", "despite", "display", "element", "enabled", "exactly", "example",
    "explain", "follows", "general", "however", "include", "instead", "measure",
    "mention", "nothing", "noticed", "obvious", "perhaps", "present", "produce",
    "provide", "reading", "receive", "regular", "related", "remains", "replace",
    "require", "results", "running", "section", "similar", "support", "through",
    "thought", "trigger", "trouble", "turning", "updated", "version", "visible",
    "whether", "working", "written", "various", "process", "message", "request",
    "setting", "feature", "content", "pattern", "context", "looking", "summary",
    "further", "exactly", "finally", "getting", "happens", "history", "nothing",
    "overall", "problem", "project", "section", "several", "testing", "updated",
    "waiting", "calling", "changed", "checked", "confirm", "details", "exactly",
    "keeping", "missing", "passing", "reading", "removed", "sending", "started",
    "stopped", "thought", "writing", "approach",
    # CSS/HTML leaks
    "border", "padding", "margin", "radius", "height", "width", "color", "style",
    "display", "position", "overflow", "opacity", "transition", "background",
    "cursor", "outline", "resize", "bottom", "center", "column", "inline",
    "repeat", "scroll", "shadow", "source", "weight", "letter", "spacing",
    # Framework artifacts
    "response", "confidence", "threshold", "coupled",
})


class IdentityModel:
    """
    Persistent identity model. Extracted from conversation + corpus.
    Grows over time. Never overwrites — only extends.
    """
    def __init__(self, path: str):
        self.path = Path(path)
        self.data = {k: [] for k in IDENTITY_KEYS}
        self.data["raw_notes"] = []
        self.data["node_weights"] = {}
        self.data["edge_counts"] = {}
        self.data["corpus_weights"] = {}
        self.data["conversation_weights"] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                saved = json.loads(self.path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                print(f"  Identity: identity.json load failed ({e}), starting fresh")
                saved = {}
            for k in IDENTITY_KEYS:
                self.data[k] = saved.get(k, [])
            self.data["node_weights"] = saved.get("node_weights", {})
            self.data["edge_counts"] = saved.get("edge_counts", {})
            self.data["corpus_weights"] = saved.get("corpus_weights", {})
            self.data["conversation_weights"] = saved.get("conversation_weights", {})
            self._clean_concepts()

    def _clean_concepts(self):
        """Remove stop-words from concept list and prune their weights/edges."""
        before = len(self.data.get("concepts", []))
        self.data["concepts"] = [
            c for c in self.data.get("concepts", [])
            if c.lower() not in _CONCEPT_STOPWORDS
        ]
        removed = before - len(self.data["concepts"])
        if removed > 0:
            # Prune node_weights for removed concepts
            valid = set(c.lower() for c in self.data["concepts"] + self.data.get("thinkers", []))
            self.data["node_weights"] = {
                k: v for k, v in self.data["node_weights"].items()
                if k.lower() in valid
            }
            # Prune edge_counts where either endpoint was removed
            self.data["edge_counts"] = {
                k: v for k, v in self.data["edge_counts"].items()
                if all(p.lower() in valid for p in k.split("|||"))
            }
            self._save()
            print(f"  Identity: cleaned {removed} stop-word concepts, pruned weights/edges")

    def _save(self):
        self.path.write_text(json.dumps(self.data, indent=2))

    def observe(self, turn_text: str, source: str = "conversation"):
        """
        Light pattern extraction from a single turn.
        Accumulates signals over time without overwriting.

        source: "corpus" or "conversation" — tracked separately so
                conversation-derived nodes aren't invisible against corpus bulk.
        """
        text = turn_text.lower()

        # Thinker name-drops
        thinkers = [
            "baudrillard","deleuze","irigaray","foucault","hegel","hume",
            "whitehead","massumi","spinoza","du bois","dubois","gramsci",
            "veblen","freud","jung","lacan","bourdieu","mauss","eco",
            "spivak","derrida","merleau-ponty","nietzsche","plato","danto",
            "fromm","giddens","riesman","hardt","debord","girard","kaufman",
            "kibran","gibran","ortega","barrett","cervantes","douglass",
        ]
        for t in thinkers:
            if t in text and t not in self.data["thinkers"]:
                self.data["thinkers"].append(t)

        # Concept signals
        concepts = [
            "coupled manifold","snobline","hessian","loss landscape",
            "curvature","anti-lora","pain proxy","phyllotaxis","irigaray",
            "substrate non-independence","coupled system","pain","finitude",
            "schizoanalysis","deterritorialization","simulacra","rhizome",
        ]
        for c in concepts:
            if c in text and c not in self.data["concepts"]:
                self.data["concepts"].append(c)

        # Lightweight semantic concept extraction — only when sentence-transformers available
        # Picks up novel domain terms Ian uses that aren't in the hardcoded list
        try:
            import re as _re, numpy as _np
            from sentence_transformers import SentenceTransformer as _ST
            if not hasattr(self, "_st_enc"):
                self._st_enc = _ST("all-MiniLM-L6-v2", device="cpu")
            _words = list(set(_re.findall(r'\b[A-Za-z][a-z]{4,}\b', turn_text)))
            _existing = list(self.data["concepts"])[-8:]
            if _existing and len(self.data["concepts"]) < 150:
                _ex_vecs = self._st_enc.encode(_existing, show_progress_bar=False)
                for _w in _words[:20]:
                    _wl = _w.lower()
                    if _wl in self.data["concepts"] or _wl in self.data["thinkers"]:
                        continue
                    if _wl in _CONCEPT_STOPWORDS:
                        continue
                    _wv = self._st_enc.encode([_wl], show_progress_bar=False)[0]
                    _norm = max(_np.linalg.norm(_wv), 1e-6)  # safe division
                    _sims = [float(_np.dot(_wv, _ev) / (_norm * max(_np.linalg.norm(_ev), 1e-6)))
                             for _ev in _ex_vecs]
                    if _sims and max(_sims) < 0.52:
                        self.data["concepts"].append(_wl)
        except Exception:
            pass  # sentence-transformers not installed — skip semantic expansion

        # Dedup lists — preserve order, remove duplicates
        self.data["thinkers"] = list(dict.fromkeys(self.data["thinkers"]))
        self.data["concepts"] = list(dict.fromkeys(self.data["concepts"]))

        # Accumulate persistent node weights + edge co-occurrences for Borges Map
        _all_terms = self.data["concepts"] + self.data["thinkers"]
        _present = [t for t in _all_terms if t.lower() in text]
        # Source-specific weight tracking
        _sw_key = "corpus_weights" if source == "corpus" else "conversation_weights"
        for t in _present:
            self.data["node_weights"][t] = self.data["node_weights"].get(t, 0) + 1
            self.data[_sw_key][t] = self.data[_sw_key].get(t, 0) + 1
        for _i in range(len(_present)):
            for _j in range(_i + 1, len(_present)):
                _ekey = "|||".join(sorted([_present[_i], _present[_j]]))
                self.data["edge_counts"][_ekey] = self.data["edge_counts"].get(_ekey, 0) + 1

        self._save()

    def ingest_corpus_summary(self, summary: str):
        """Store a high-level summary extracted from the corpus."""
        if summary and summary not in self.data["raw_notes"]:
            self.data["raw_notes"].append(summary[:500])
            self._save()

    def set(self, key: str, value):
        """Manually set an identity field."""
        if key in self.data:
            if isinstance(self.data[key], list):
                if value not in self.data[key]:
                    self.data[key].append(value)
            else:
                self.data[key] = value
            self._save()

    def to_block(self) -> str:
        """Render identity as a compact context block about the USER."""
        lines = [
            "[ABOUT THE USER YOU ARE TALKING TO]",
            "You are an AI assistant. The following describes the HUMAN user in this conversation.",
            "Do not confuse this information with your own identity.",
        ]
        # Core identity fields
        for key, label in [("name", "User name"), ("age", "User age"),
                           ("role", "User role"), ("location", "User location")]:
            val = self.data.get(key, [])
            if val:
                lines.append(f"{label}: {', '.join(str(v) for v in val[:12]) if isinstance(val, list) else val}")
        # User's own statements — placed early so the model attends to them
        if self.data.get("raw_notes"):
            _notes = self.data['raw_notes'][-5:]
            lines.append(f"The user has told you: {' | '.join(_notes)}")
        # Extended profile
        for key, label in [("education", "User education"), ("projects", "User projects"),
                           ("thinkers", "Thinkers user references"),
                           ("concepts", "Concepts user works with"),
                           ("voice", "User writing voice"), ("goals", "User goals"),
                           ("writing_style", "User writing style"),
                           ("recurring_themes", "User recurring themes"),
                           ("tools", "User tools")]:
            val = self.data.get(key, [])
            if val:
                lines.append(f"{label}: {', '.join(str(v) for v in val[:12]) if isinstance(val, list) else val}")
        lines.append("[END USER INFO — you are the AI assistant, not this person]")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# ARCHIVE INDEX — semantic search over full conversation history
# ═══════════════════════════════════════════════════════════════

class ArchiveIndex:
    """
    Persistent semantic index over the full conversation archive.
    Embeds every turn as a 384-dim vector (MiniLM-L6-v2).
    Enables nearest-neighbor retrieval across all history, not just
    the last N turns — same principle as corpus RAG, applied to memory.

    On first boot it back-fills any un-indexed archive turns in the background.
    Each new turn is embedded asynchronously so the main thread is never blocked.
    """

    BOOT_CAP = 1000   # max archive turns to back-fill at boot (most recent N)
    THRESHOLD = 0.32  # min cosine similarity to return a result

    def __init__(self, index_path: Path):
        self._path   = index_path.with_suffix(".pkl")
        self._turns: list[dict] = []   # {"role", "content", "ts"}
        self._vecs   = None            # np.ndarray (N, 384) float32, or None
        self._lock   = threading.Lock()
        self._enc    = None            # lazy-loaded SentenceTransformer
        self._load()

    # ── encoder ─────────────────────────────────────────────────
    def _encoder(self):
        if self._enc is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._enc = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            except Exception:
                pass
        return self._enc

    # ── persistence ─────────────────────────────────────────────
    def _load(self):
        if not self._path.exists():
            return
        try:
            import pickle
            # TRUST BOUNDARY: pickle files are self-generated by _save() below.
            # No external input path — safe for local-only use.
            with open(self._path, "rb") as f:
                data = pickle.load(f)
            self._turns = data.get("turns", [])
            self._vecs  = data.get("vecs")
        except Exception:
            self._turns = []
            self._vecs  = None

    def _flush(self):
        try:
            import pickle
            with open(self._path, "wb") as f:
                pickle.dump({"turns": self._turns, "vecs": self._vecs}, f)
        except Exception:
            pass

    def size(self) -> int:
        return len(self._turns)

    # ── indexing ────────────────────────────────────────────────
    def add_batch(self, turns: list[dict]):
        """Embed a batch of turns and persist. Safe to call from background thread."""
        enc = self._encoder()
        if enc is None or not turns:
            return
        try:
            import numpy as np
            texts = [t.get("content", "")[:600] for t in turns]
            vecs  = enc.encode(texts, show_progress_bar=False, batch_size=32).astype(np.float32)
            with self._lock:
                self._turns.extend(turns)
                self._vecs = np.vstack([self._vecs, vecs]) if self._vecs is not None else vecs
                self._flush()
        except Exception:
            pass

    def add(self, role: str, content: str, ts: float):
        self.add_batch([{"role": role, "content": content[:600], "ts": ts}])

    # ── retrieval ───────────────────────────────────────────────
    def search(self, query: str, k: int = 4, skip_last: int = 0) -> list[dict]:
        """
        Return up to k turns most semantically similar to query.
        skip_last excludes the N most recent turns (they're already in the
        recency window and don't need to be recalled separately).
        Results sorted chronologically so context reads naturally.
        """
        with self._lock:
            n = len(self._turns) - skip_last
            if self._vecs is None or n <= 0:
                return []
            turns_slice = self._turns[:n]
            vecs_slice  = self._vecs[:n].copy()

        enc = self._encoder()
        if enc is None:
            return []
        try:
            import numpy as np
            q_vec = enc.encode([query], show_progress_bar=False)[0].astype(np.float32)
            q_norm = float(np.linalg.norm(q_vec))
            if q_norm < 1e-9:
                return []
            row_norms = np.linalg.norm(vecs_slice, axis=1)
            row_norms[row_norms < 1e-9] = 1.0
            scores = (vecs_slice / row_norms[:, None]) @ (q_vec / q_norm)
            top_idx = np.argsort(scores)[::-1][:k * 2]
            results = [turns_slice[i] for i in top_idx if scores[i] >= self.THRESHOLD][:k]
            results.sort(key=lambda x: x.get("ts", 0))
            return results
        except Exception:
            return []


# ═══════════════════════════════════════════════════════════════
# TWO-LAYER HISTORY
# ═══════════════════════════════════════════════════════════════

class LayeredHistory:
    """
    Three tiers:
      archive.jsonl   — every turn ever, append-only, never trimmed
      last_history.json — last N turns for direct context injection
      summaries.json  — rolling compressed summaries of older sessions
    """
    def __init__(self, data_dir: str, window: int = 20, summarize_every: int = 40):
        d = Path(data_dir) / "sessions"
        d.mkdir(parents=True, exist_ok=True)
        self.archive_path  = d / "archive.jsonl"
        self.history_path  = d / "last_history.json"
        self.summary_path  = d / "summaries.json"
        self.window        = window
        self.summarize_every = summarize_every
        self.recent: list[dict] = []
        self.summaries: list[str] = []
        self._session_start_ts: float = 0  # epoch — recall only returns turns after this
        self.index = ArchiveIndex(d / "archive_index")
        self._load()
        # Back-fill any archive turns not yet in the semantic index
        threading.Thread(target=self._boot_index, daemon=True).start()

    def start_new_session(self):
        """Reset session-scoped state so a new chat starts clean."""
        self.recent = []
        self.summaries = []
        self._session_start_ts = time.time()

    def _load(self):
        if self.history_path.exists():
            try:
                self.recent = json.loads(self.history_path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                print(f"  History: last_history.json load failed ({e}), starting fresh")
                self.recent = []
        if self.summary_path.exists():
            try:
                self.summaries = json.loads(self.summary_path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                print(f"  History: summaries.json load failed ({e}), starting fresh")
                self.summaries = []
        if self.archive_path.exists():
            with open(self.archive_path) as _f:
                n_archive = sum(1 for _ in _f)
        else:
            n_archive = 0
        print(f"  History: {len(self.recent)} recent | {len(self.summaries)} summaries | {n_archive} archived")

    def _boot_index(self):
        """Back-fill any un-indexed archive turns. Runs once at boot on background thread."""
        if not self.archive_path.exists():
            return
        try:
            with open(self.archive_path) as f:
                all_turns = []
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        all_turns.append(json.loads(ln))
                    except json.JSONDecodeError:
                        continue  # skip corrupted lines silently
            already = self.index.size()
            new_turns = all_turns[already:]
            # Cap back-fill to avoid booting for hours on massive archives
            if new_turns:
                new_turns = new_turns[-ArchiveIndex.BOOT_CAP:]
                self.index.add_batch(new_turns)
        except Exception:
            pass

    def append(self, role: str, content: str):
        clean   = _strip_medulla_static(content)
        record  = {"role": role, "content": clean, "ts": time.time()}

        # Archive — append only, never lost
        with self.archive_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

        # Recent window
        self.recent.append({"role": role, "content": clean})
        if len(self.recent) > self.window * 2:
            self._maybe_summarize()
            self.recent = self.recent[-self.window:]
        self.history_path.write_text(json.dumps(self.recent, indent=2))

        # Semantic index — background embed so main thread isn't blocked
        def _add_safe():
            try: self.index.add(role, clean, record["ts"])
            except Exception: pass  # Index failure is non-fatal; search will use keyword fallback

        threading.Thread(target=_add_safe, daemon=True).start()

    def _maybe_summarize(self):
        """Compress the oldest half of recent into a summary string."""
        to_compress = self.recent[:self.window]
        lines = []
        for h in to_compress:
            role    = h.get("role", "?")
            content = h.get("content", "")[:200]
            lines.append(f"{role}: {content}")
        summary = "Earlier: " + " | ".join(lines)
        self.summaries.append(summary)
        # Keep last 10 summaries
        self.summaries = self.summaries[-10:]
        self.summary_path.write_text(json.dumps(self.summaries, indent=2))

    def get_context_messages(self, max_turns: int = None, query: str = None) -> list[dict]:
        """
        Returns messages for the model context window.

        max_turns: how many recent turns to include (shallow = fast, deep = thorough)
        query: if provided, also retrieves semantically relevant turns from the full
               archive via ArchiveIndex — enabling recall across all conversation history,
               not just the sliding window.
        """
        cap = max_turns if max_turns is not None else self.window
        msgs = []

        # System anchor
        system_parts = [
            "You are a local AI assistant called Coupled Manifold.",
            "You are NOT the user. You are NOT Ian Preston-Campbell.",
            "You are an AI. The user is a human named Ian.",
            "Do not claim to have done research, written papers, or managed anything.",
            "Respond helpfully and directly. Never roleplay as the user.",
        ]
        if self.summaries and cap >= 8 and not self._session_start_ts:
            summary_block = "\n".join(self.summaries[-2:])
            system_parts.append(f"\n[CONVERSATION HISTORY SUMMARY]\n{summary_block}")
        msgs.append({"role": "system", "content": " ".join(system_parts)})

        # Semantic recall from full archive (if query provided and index populated)
        if query and self.index.size() > cap:
            _k = min(4 + max(0, self.index.size() - 50) // 100, 10)
            recalled = [t for t in self.index.search(query, k=_k, skip_last=cap)
                        if t.get("ts", 0) >= self._session_start_ts]
            if recalled:
                lines = []
                for t in recalled:
                    label = "Ian" if t["role"] == "user" else "Assistant"
                    lines.append(f"{label}: {t['content'][:300]}")
                msgs.append({
                    "role": "system",
                    "content": (
                        "[RECALLED — semantically relevant turns from earlier in your history]\n"
                        + "\n".join(lines)
                        + "\n[Use this for context. Do not repeat it verbatim.]"
                    )
                })

        # Recent turns — always included for immediate coherence
        # Cap assistant messages to prevent model's own verbose responses
        # from re-poisoning subsequent context with invented concepts
        for m in self.recent[-cap:]:
            if m.get("role") == "assistant":
                c = m.get("content", "")
                msgs.append({"role": "assistant", "content": c[:1500] + ("…" if len(c) > 1500 else "")})
            else:
                msgs.append(m)
        return msgs

    def all_text(self) -> str:
        """All archived text for corpus indexing."""
        if not self.archive_path.exists():
            return ""
        return self.archive_path.read_text(errors="ignore")


def _strip_medulla_static(content: str) -> str:
    if isinstance(content, str) and "<div style=" in content:
        return content.split("\n\n<div style=")[0]
    return str(content) if not isinstance(content, str) else content


# ═══════════════════════════════════════════════════════════════
# MEMORY — top-level interface
# ═══════════════════════════════════════════════════════════════

class Memory:
    """
    Single interface the app talks to.

    mem = Memory("./manifold_data")
    mem.append_turn("user", "hello")
    mem.append_turn("assistant", "hi")
    context = mem.build_context("what is the pain proxy?")
    """

    def __init__(self, data_dir: str = "./manifold_data"):
        self.data_dir = Path(data_dir)
        self.history  = LayeredHistory(data_dir)
        self.corpus   = Corpus(str(self.data_dir / "corpus"))
        self.identity = IdentityModel(str(self.data_dir / "identity.json"))

        # Seed known identity facts about Ian
        self._seed_identity()
        # One-time: feed corpus chunks through observe() to populate the knowledge graph
        self._seed_from_corpus()

    def _seed_from_corpus(self):
        """Run observe() on each corpus chunk to populate identity graph. Runs once."""
        sentinel = self.data_dir / ".corpus_seeded"
        if sentinel.exists() or not self.corpus.chunks:
            return
        n = len(self.corpus.chunks)
        print(f"  Identity: seeding from {n} corpus chunks …")
        for chunk in self.corpus.chunks:
            self.identity.observe(chunk["text"], source="corpus")
        sentinel.write_text(f"seeded {n} chunks\n")
        print(f"  Identity: seeded — {len(self.identity.data.get('thinkers', []))} thinkers, "
              f"{len(self.identity.data.get('concepts', []))} concepts, "
              f"{len(self.identity.data.get('node_weights', {}))} weighted nodes")

    def _seed_identity(self):
        """Pre-load known facts. Only adds if not already present."""
        seeds = {
            "name":      "Ian J. Preston-Campbell",
            "age":       "21",
            "role":      "General Manager, Fry The Coop Chicago; researcher",
            "education": "BSB + MS Marketing, DePaul Kellstadt",
            "projects":  "Coupled Manifold Safety Framework; Growth is Subversion (236pp autoethnography); Post-Humean Inferno",
            "tools":     "Substack (ijpc43), ResearchGate (CC BY 4.0), Colab A100, Python",
            "writing_style": "declarative, run-on, no dashes, no hedging, thinker names without explanation",
            "voice":     "rapid, fragmented, trust-based; drops names without scaffolding",
            "recurring_themes": "substrate non-independence, pain as curvature, finitude, coupled system, schizoanalysis, AI as tool not mind",
        }
        for k, v in seeds.items():
            existing = self.identity.data.get(k, [])
            if not existing:
                self.identity.set(k, v)

    def append_turn(self, role: str, content: str):
        """Call after every turn — both user and assistant."""
        self.history.append(role, content)
        # identity.observe runs up to 20 sentence-transformers encode calls — move to background
        # so it doesn't add 0.5-1s of latency before generation starts
        threading.Thread(target=self.identity.observe, args=(content,),
                         kwargs={"source": "conversation"}, daemon=True).start()

    def index_corpus(self, path: str):
        """Index a single file into the corpus."""
        return self.corpus.index_file(path)

    def index_directory(self, directory: str):
        """Index all text files in a directory."""
        self.corpus.index_directory(directory)

    def index_text(self, text: str, source: str = "manual"):
        """Index raw text string."""
        return self.corpus.index(text, source)

    def index_conversation_history(self):
        """Index the full conversation archive into the corpus."""
        text = self.history.all_text()
        if text:
            self.corpus.index(text, source="conversation_archive")

    def build_context(self, query: str, n_corpus: int = 3,
                      min_score: float = 0.20, compact_identity: bool = False) -> str:
        """
        Build the context block to inject into the prompt.
        - min_score: only include corpus chunks whose combined score >= this threshold.
          Prevents injecting irrelevant concordance/docs for off-topic queries.
        - compact_identity: if True, emit a single-line identity summary instead of
          the full block (saves ~150 tokens for simple turns).
        """
        parts = []

        # Identity — full or compact
        if compact_identity:
            name = self.identity.data.get("name", ["Ian"])[0] if isinstance(
                self.identity.data.get("name"), list) else self.identity.data.get("name", "Ian")
            role = self.identity.data.get("role", [""])[0] if isinstance(
                self.identity.data.get("role"), list) else self.identity.data.get("role", "")
            parts.append(f"[User: {name} — {role}]")
        else:
            id_block = self.identity.to_block()
            if id_block.strip() not in ("", "[IDENTITY]"):
                parts.append(id_block)

        # Corpus — only inject chunks that are actually relevant
        if n_corpus > 0:
            scored = self.corpus.search(query, k=n_corpus, return_scores=True)
            relevant = [(s, c) for s, c in scored if s >= min_score]
            if relevant:
                parts.append("[RELEVANT EXCERPTS FROM THE USER'S WRITING — for context only, not your own words]")
                for _, c in relevant:
                    src = c.get("source", "unknown")
                    txt = c.get("text", "")[:400]
                    parts.append(f"[user writing — {src}]\n{txt}")

        return "\n\n".join(parts) if parts else ""

    def get_history_messages(self, max_turns: int = None, query: str = None) -> list[dict]:
        """Return messages list for model context.
        max_turns limits recency depth; query enables semantic archive retrieval."""
        return self.history.get_context_messages(max_turns=max_turns, query=query)

    def search_archive(self, query: str, k: int = 10) -> list:
        """
        Search the full conversation archive for turns semantically related to query.
        Returns a list of dicts with keys: user, response, ts.
        Each dict represents a user+assistant exchange pair.
        Falls back to scanning archive.jsonl directly if the semantic index is empty.
        """
        # Try semantic index first
        raw_results = self.history.index.search(query, k=k * 2)

        if raw_results:
            # Pair user/assistant turns: for each result, find the adjacent turn
            # The archive stores individual turns {role, content, ts}.
            # We want to surface exchanges: {user, response, ts}.
            paired = []
            seen_ts = set()
            # Build a lookup of all archive turns for pairing
            archive_turns = []
            p = self.history.archive_path
            if p.exists():
                try:
                    with open(p) as _f:
                        archive_turns = []
                        for ln in _f:
                            ln = ln.strip()
                            if not ln:
                                continue
                            try:
                                archive_turns.append(json.loads(ln))
                            except json.JSONDecodeError:
                                continue  # skip corrupted lines silently
                except Exception:
                    pass
            # Index archive by position for fast adjacent lookup
            ts_to_idx = {}
            for i, t in enumerate(archive_turns):
                ts_to_idx[t.get("ts", 0)] = i

            for turn in raw_results:
                ts = turn.get("ts", 0)
                if ts in seen_ts:
                    continue
                seen_ts.add(ts)
                role = turn.get("role", "")
                content = turn.get("content", "")
                idx = ts_to_idx.get(ts)

                if role == "user":
                    # Find adjacent assistant response
                    resp = ""
                    if idx is not None and idx + 1 < len(archive_turns):
                        nxt = archive_turns[idx + 1]
                        if nxt.get("role") == "assistant":
                            resp = nxt.get("content", "")
                    paired.append({"user": content, "response": resp, "ts": ts})
                elif role == "assistant":
                    # Find preceding user message
                    user_text = ""
                    if idx is not None and idx > 0:
                        prev = archive_turns[idx - 1]
                        if prev.get("role") == "user":
                            user_text = prev.get("content", "")
                    paired.append({"user": user_text, "response": content, "ts": ts})

            return paired[:k]

        # Fallback: scan archive.jsonl directly for keyword matches
        p = self.history.archive_path
        if not p.exists():
            return []
        try:
            with open(p) as _f:
                all_turns = []
                for ln in _f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        all_turns.append(json.loads(ln))
                    except json.JSONDecodeError:
                        continue  # skip corrupted lines silently
        except Exception:
            return []

        q_lower = query.lower()
        paired = []
        for i, turn in enumerate(all_turns):
            if q_lower in turn.get("content", "").lower():
                role = turn.get("role", "")
                ts   = turn.get("ts", 0)
                if role == "user":
                    resp = all_turns[i + 1].get("content", "") if i + 1 < len(all_turns) and all_turns[i + 1].get("role") == "assistant" else ""
                    paired.append({"user": turn["content"], "response": resp, "ts": ts})
                elif role == "assistant":
                    user_text = all_turns[i - 1].get("content", "") if i > 0 and all_turns[i - 1].get("role") == "user" else ""
                    paired.append({"user": user_text, "response": turn["content"], "ts": ts})
                if len(paired) >= k:
                    break
        return paired

    def status(self) -> dict:
        p = self.history.archive_path
        if p.exists():
            with p.open() as f:
                archive_turns = sum(1 for _ in f)
        else:
            archive_turns = 0
        return {
            "corpus_chunks":  len(self.corpus.chunks),
            "archive_turns":  archive_turns,
            "summaries":      len(self.history.summaries),
            "recent_turns":   len(self.history.recent),
            "thinkers_known": len(self.identity.data.get("thinkers", [])),
            "concepts_known": len(self.identity.data.get("concepts", [])),
        }
