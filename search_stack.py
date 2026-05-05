"""
COUPLED MANIFOLD — Search Stack
================================
Ranked sources: Brave > YouTube > Google Custom > OpenAlex > arXiv > Semantic Scholar > Unpaywall > DDG
Smart query reformulation when direct search fails.
"""

import os, json, time
import urllib.parse

# All HTTP sources must send a User-Agent identifying the application.
# Many APIs (Wikipedia, GitHub, etc.) now require this and return 403 without it.
_USER_AGENT = "Graceful/1.0 (https://github.com/ianjpdepaul-rgb/Coupled-Manifold; local AI assistant) httpx"

# Safety layer — graceful fallback if module not present
try:
    from search_sanitizer import (
        sanitize_results, trust_score, compute_consensus,
        format_results_safe, build_consensus_note,
        should_search_v2, _search_limiter, _search_cache,
        relevance_score,
    )
    _SANITIZER_AVAILABLE = True
except ImportError as _e:
    _SANITIZER_AVAILABLE = False
    print(f"  ⚠ search_sanitizer not found — running without safety layer ({_e})")

# ── Search status disclosure ─────────────────────────────────────────────────
_last_search_query = [""]

def get_last_search_query() -> str:
    """Return the most recent search query issued this session."""
    return _last_search_query[0]

DATA_DIR = "./manifold_data"
_KEYS_PATH = f"{DATA_DIR}/keys.json"

def load_keys() -> dict:
    if os.path.exists(_KEYS_PATH):
        try:
            with open(_KEYS_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

KEYS = load_keys()


# ── Individual sources ───────────────────────────────────────────────────────

def _brave(query: str, n: int = 5) -> list[dict]:
    key = KEYS.get("brave")
    if not key: return []
    try:
        import httpx
        r = httpx.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"Accept": "application/json", "X-Subscription-Token": key,
                     "User-Agent": _USER_AGENT},
            params={"q": query, "count": n}, timeout=6.0
        )
        return [{"title": x.get("title",""), "body": x.get("description",""),
                 "url": x.get("url",""), "source": "brave"}
                for x in r.json().get("web",{}).get("results",[])]
    except Exception:
        return []


def _youtube(query: str, n: int = 3) -> list[dict]:
    key = KEYS.get("youtube")
    if not key: return []
    try:
        import httpx
        r = httpx.get(
            "https://www.googleapis.com/youtube/v3/search",
            headers={"User-Agent": _USER_AGENT},
            params={"key": key, "q": query, "part": "snippet",
                    "type": "video", "maxResults": n}, timeout=6.0
        )
        return [{"title": i["snippet"]["title"],
                 "body": f"{i['snippet']['description']} — https://youtube.com/watch?v={i['id']['videoId']}",
                 "source": "youtube"}
                for i in r.json().get("items",[])]
    except Exception:
        return []


def _google_custom(query: str, n: int = 5) -> list[dict]:
    key = KEYS.get("google_search")
    cx  = KEYS.get("google_cx")
    if not key or not cx: return []
    try:
        import httpx
        r = httpx.get(
            "https://www.googleapis.com/customsearch/v1",
            headers={"User-Agent": _USER_AGENT},
            params={"key": key, "cx": cx, "q": query, "num": n}, timeout=6.0
        )
        return [{"title": i.get("title",""), "body": i.get("snippet",""),
                 "url": i.get("link",""), "source": "google"}
                for i in r.json().get("items",[])]
    except Exception:
        return []


def _openalex(query: str, n: int = 3) -> list[dict]:
    try:
        import httpx
        r = httpx.get(
            "https://api.openalex.org/works",
            headers={"User-Agent": _USER_AGENT},
            params={"search": query, "per-page": n,
                    "mailto": KEYS.get("unpaywall_email","research@coupled.ai")},
            timeout=6.0
        )
        out = []
        for w in r.json().get("results",[]):
            title = w.get("title","")
            year  = w.get("publication_year","")
            cited = w.get("cited_by_count",0)
            doi   = w.get("doi","") or ""
            ab_inv = w.get("abstract_inverted_index") or {}
            if ab_inv:
                words = sorted(ab_inv.items(), key=lambda x: min(x[1]))
                ab = " ".join(wd for wd,_ in words[:60])
            else:
                ab = ""
            out.append({"title": f"{title} ({year}, {cited} citations)",
                        "body": f"{ab} {doi}".strip(), "source": "openalex"})
        return out
    except Exception:
        return []


def _arxiv(query: str, n: int = 3) -> list[dict]:
    try:
        import httpx, xml.etree.ElementTree as ET
        r = httpx.get(
            "https://export.arxiv.org/api/query",
            headers={"User-Agent": _USER_AGENT},
            params={"search_query": f"all:{query}", "start": 0,
                    "max_results": n, "sortBy": "relevance"}, timeout=6.0
        )
        ns  = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(r.text)
        out = []
        for entry in root.findall("a:entry", ns):
            title   = entry.findtext("a:title","",ns).strip().replace("\n"," ")
            summary = entry.findtext("a:summary","",ns).strip()[:300]
            link    = entry.findtext("a:id","",ns).strip()
            out.append({"title": f"[arXiv] {title}",
                        "body": f"{summary} — {link}", "source": "arxiv"})
        return out
    except Exception:
        return []


def _semantic_scholar(query: str, n: int = 3) -> list[dict]:
    try:
        import httpx
        r = httpx.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            headers={"User-Agent": _USER_AGENT},
            params={"query": query, "limit": n,
                    "fields": "title,abstract,year,citationCount,externalIds"},
            timeout=6.0
        )
        out = []
        for p in r.json().get("data",[]):
            title = p.get("title","")
            year  = p.get("year","")
            cited = p.get("citationCount",0)
            ab    = (p.get("abstract") or "")[:300]
            ids   = p.get("externalIds",{})
            doi   = ids.get("DOI","")
            arxiv = ids.get("ArXiv","")
            link  = f"https://arxiv.org/abs/{arxiv}" if arxiv else f"https://doi.org/{doi}" if doi else ""
            out.append({"title": f"[S2] {title} ({year}, {cited} citations)",
                        "body": f"{ab} {link}".strip(), "source": "semantic_scholar"})
        return out
    except Exception:
        return []


def _unpaywall(doi: str) -> str:
    """Given a DOI, find a free PDF link."""
    email = KEYS.get("unpaywall_email","research@coupled.ai")
    try:
        import httpx
        r = httpx.get(
            f"https://api.unpaywall.org/v2/{doi}",
            headers={"User-Agent": _USER_AGENT},
            params={"email": email}, timeout=5.0
        )
        data = r.json()
        loc  = data.get("best_oa_location") or {}
        return loc.get("url_for_pdf","") or loc.get("url","")
    except Exception:
        return ""


def _duckduckgo(query: str, n: int = 5) -> list[dict]:
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            return [{"title": r.get("title",""), "body": r.get("body",""),
                     "url": r.get("href", r.get("url","")),
                     "source": "ddg"}
                    for r in ddgs.text(query, max_results=n)]
    except Exception:
        return []


def _hackernews(query: str, n: int = 3) -> list[dict]:
    """Search HackerNews via Algolia API — free, no key."""
    try:
        import httpx
        r = httpx.get(
            "https://hn.algolia.com/api/v1/search",
            headers={"User-Agent": _USER_AGENT},
            params={"query": query, "tags": "story", "hitsPerPage": n},
            timeout=5.0
        )
        out = []
        for hit in r.json().get("hits", []):
            title = hit.get("title", "")
            url   = hit.get("url", "") or f"https://news.ycombinator.com/item?id={hit.get('objectID','')}"
            points = hit.get("points", 0)
            comments = hit.get("num_comments", 0)
            out.append({
                "title": f"[HN] {title} ({points}pts, {comments} comments)",
                "body":  hit.get("story_text", "")[:200] or f"Discussed on HackerNews",
                "url":   url,
                "source": "hackernews"
            })
        return out
    except Exception:
        return []


def _stack_exchange(query: str, n: int = 3, site: str = "stackoverflow") -> list[dict]:
    """Search Stack Exchange — free, no key required (rate limited without key)."""
    try:
        import httpx
        r = httpx.get(
            "https://api.stackexchange.com/2.3/search/advanced",
            headers={"User-Agent": _USER_AGENT},
            params={
                "q": query, "site": site, "pagesize": n,
                "order": "desc", "sort": "relevance",
                "filter": "withbody"
            },
            timeout=6.0
        )
        out = []
        for item in r.json().get("items", []):
            from html.parser import HTMLParser
            class _S(HTMLParser):
                def __init__(self): super().__init__(); self.t=[]
                def handle_data(self,d): self.t.append(d)
            s=_S(); s.feed(item.get("body","")[:500]); plain=" ".join(s.t)[:300]
            out.append({
                "title": f"[{site}] {item.get('title','')}",
                "body":  plain or item.get("excerpt","")[:300],
                "url":   item.get("link",""),
                "source": "stackexchange"
            })
        return out
    except Exception:
        return []


def _pubmed(query: str, n: int = 3) -> list[dict]:
    """Search PubMed via NCBI E-utilities — free, no key required."""
    try:
        import httpx
        # Step 1: search for IDs
        r = httpx.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            headers={"User-Agent": _USER_AGENT},
            params={"db": "pubmed", "term": query, "retmax": n, "retmode": "json"},
            timeout=6.0
        )
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        # Step 2: fetch summaries
        r2 = httpx.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            headers={"User-Agent": _USER_AGENT},
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
            timeout=6.0
        )
        result_data = r2.json().get("result", {})
        out = []
        for pmid in ids:
            doc = result_data.get(pmid, {})
            title = doc.get("title", "")
            year  = doc.get("pubdate", "")[:4]
            src   = doc.get("source", "")
            out.append({
                "title": f"[PubMed] {title} ({src}, {year})",
                "body":  f"PMID: {pmid} — {src}",
                "url":   f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "source": "pubmed"
            })
        return out
    except Exception:
        return []


def _core(query: str, n: int = 3) -> list[dict]:
    """Search CORE open access — free tier, no key for basic queries."""
    try:
        import httpx
        r = httpx.get(
            "https://api.core.ac.uk/v3/search/works",
            params={"q": query, "limit": n},
            headers={"Authorization": "", "User-Agent": _USER_AGENT},
            timeout=6.0
        )
        out = []
        for w in r.json().get("results", []):
            title    = w.get("title", "")
            abstract = (w.get("abstract") or "")[:300]
            year     = w.get("yearPublished", "")
            dl_url   = w.get("downloadUrl", "") or (w.get("sourceFulltextUrls", [""])[0] if w.get("sourceFulltextUrls") else "")
            out.append({
                "title": f"[CORE] {title} ({year})",
                "body":  abstract,
                "url":   dl_url or f"https://core.ac.uk/search?q={query}",
                "source": "core"
            })
        return out
    except Exception:
        return []


def _github(query: str, n: int = 3) -> list[dict]:
    """Search GitHub repositories — free, no key (10 req/min rate limit)."""
    try:
        import httpx
        r = httpx.get(
            "https://api.github.com/search/repositories",
            params={"q": query, "sort": "stars", "order": "desc", "per_page": n},
            headers={"Accept": "application/vnd.github.v3+json",
                     "User-Agent": _USER_AGENT},
            timeout=6.0
        )
        out = []
        for repo in r.json().get("items", []):
            out.append({
                "title": f"[GitHub] {repo.get('full_name','')} ⭐{repo.get('stargazers_count',0)}",
                "body":  (repo.get("description") or "")[:300],
                "url":   repo.get("html_url", ""),
                "source": "github"
            })
        return out
    except Exception:
        return []


def _crossref(query: str, n: int = 3) -> list[dict]:
    """Search Crossref academic citations — free, no key."""
    try:
        import httpx
        email = KEYS.get("unpaywall_email", "research@coupled.ai")
        r = httpx.get(
            "https://api.crossref.org/works",
            headers={"User-Agent": _USER_AGENT},
            params={"query": query, "rows": n,
                    "mailto": email,
                    "select": "title,author,published,DOI,abstract,container-title"},
            timeout=6.0
        )
        out = []
        for item in r.json().get("message", {}).get("items", []):
            title   = " ".join(item.get("title", [""])[:1])
            authors = ", ".join(a.get("family","") for a in item.get("author",[])[:2])
            year    = (item.get("published",{}).get("date-parts") or [[""]])[0][0]
            doi     = item.get("DOI", "")
            journal = " ".join(item.get("container-title",[""])[:1])
            ab_raw  = item.get("abstract","")
            # strip JATS XML tags from abstract
            import re as _re
            ab = _re.sub(r'<[^>]+>', '', ab_raw)[:300]
            out.append({
                "title": f"[Crossref] {title} — {authors} ({year})",
                "body":  ab or journal,
                "url":   f"https://doi.org/{doi}" if doi else "",
                "source": "crossref"
            })
        return out
    except Exception:
        return []


def _internet_archive(query: str, n: int = 3) -> list[dict]:
    """Search Internet Archive full-text — free, no key."""
    try:
        import httpx
        r = httpx.get(
            "https://archive.org/advancedsearch.php",
            headers={"User-Agent": _USER_AGENT},
            params={
                "q": query, "output": "json", "rows": n,
                "fl[]": ["identifier","title","description","date"],
                "sort[]": "downloads desc"
            },
            timeout=6.0
        )
        out = []
        for doc in r.json().get("response", {}).get("docs", []):
            ident = doc.get("identifier","")
            out.append({
                "title": f"[Archive.org] {doc.get('title',ident)}",
                "body":  (doc.get("description","")[:300] if isinstance(doc.get("description"), str)
                          else str(doc.get("description",""))[:200]),
                "url":   f"https://archive.org/details/{ident}",
                "source": "archive"
            })
        return out
    except Exception:
        return []


# ── Page content fetcher ─────────────────────────────────────────────────────

def _fetch_page_content(url: str, max_chars: int = 2000) -> str:
    """Fetch and extract main text content from a URL."""
    if not url or not url.startswith(("http://", "https://")):
        return ""
    try:
        import httpx
        from html.parser import HTMLParser
        r = httpx.get(url, timeout=8.0, follow_redirects=True,
                      headers={"User-Agent": _USER_AGENT})
        if r.status_code != 200:
            return ""
        class _P(HTMLParser):
            def __init__(self):
                super().__init__(); self.text=[]; self._skip=False
                self._skip_tags={"script","style","nav","footer","header","aside","form","button"}
            def handle_starttag(self,t,a):
                if t in self._skip_tags: self._skip=True
            def handle_endtag(self,t):
                if t in self._skip_tags: self._skip=False
            def handle_data(self,d):
                if not self._skip and d.strip(): self.text.append(d.strip())
        p=_P(); p.feed(r.text)
        return " ".join(p.text)[:max_chars]
    except Exception:
        return ""


# ── Wikipedia direct API ─────────────────────────────────────────────────────

def _wikipedia(query: str, n: int = 2) -> list[dict]:
    """Direct Wikipedia API — returns structured summaries."""
    try:
        import httpx, urllib.parse
        search_url = "https://en.wikipedia.org/w/api.php"
        _headers = {"User-Agent": _USER_AGENT}
        r = httpx.get(search_url, params={
            "action": "query", "list": "search", "srsearch": query,
            "srlimit": n, "format": "json", "utf8": 1
        }, headers=_headers, timeout=5.0)
        titles = [s["title"] for s in r.json().get("query",{}).get("search",[])]
        out = []
        for title in titles:
            r2 = httpx.get(search_url, params={
                "action": "query", "prop": "extracts", "exintro": True,
                "explaintext": True, "titles": title, "format": "json"
            }, headers=_headers, timeout=5.0)
            pages = r2.json().get("query",{}).get("pages",{})
            for page in pages.values():
                extract = (page.get("extract") or "")[:400]
                if extract:
                    out.append({
                        "title": f"[Wikipedia] {title}",
                        "body": extract,
                        "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title)}",
                        "source": "wikipedia"
                    })
        return out
    except Exception:
        return []


# ── Smart query reformulation ────────────────────────────────────────────────

def reformulate_query(original: str, context: str = "") -> list[str]:
    """
    Generate alternative search queries when the original fails.
    Tries: exact title quoted, author + keywords, DOI prefix, ResearchGate path.
    """
    queries = []
    q = original.strip()

    # Quoted exact match
    queries.append(f'"{q}"')

    # ResearchGate specific
    queries.append(f'site:researchgate.net "{q}"')

    # arXiv specific
    queries.append(f'site:arxiv.org {q}')

    # Semantic reformulation — drop stop words, keep content words
    stops = {"the","a","an","of","in","on","for","and","or","with","to","by","is","are","as"}
    keywords = [w for w in q.lower().split() if w not in stops and len(w) > 3]
    if keywords:
        queries.append(" ".join(keywords[:5]))

    # If it looks like a paper title, add "paper" or "preprint"
    if len(q.split()) > 3:
        queries.append(f"{q} preprint")
        queries.append(f"{q} paper PDF")

    return queries[:4]  # max 4 reformulations


# ── Query routing ────────────────────────────────────────────────────────────

def _is_academic(query: str) -> bool:
    signals = ["paper","study","research","journal","doi","arxiv","published",
               "abstract","citation","authors","findings","results","methodology",
               "coupled manifold","framework","analysis","model","theory"]
    return any(s in query.lower() for s in signals)

def _is_video(query: str) -> bool:
    return any(s in query.lower() for s in
               ["youtube","video","watch","tutorial","lecture","talk","podcast"])

def _is_social(query: str) -> bool:
    return any(s in query.lower() for s in
               ["instagram","linkedin","twitter","tiktok","profile","social",
                "grace","@","follow"])

def _is_code(query: str) -> bool:
    return any(s in query.lower() for s in [
        "code","function","class","implement","github","library","package",
        "npm","pip","python","javascript","typescript","how to build","api",
        "open source","repository","framework","sdk"
    ])

def _is_medical(query: str) -> bool:
    return any(s in query.lower() for s in [
        "disease","treatment","drug","clinical","medical","patient","health",
        "cancer","study","trial","diagnosis","therapy","syndrome","pubmed"
    ])


# ── Main search function ─────────────────────────────────────────────────────

_TIME_SIGNALS = ["today","yesterday","this week","this month","right now",
                 "currently","latest","recent","news","2024","2025","2026",
                 "price","stock","weather","score","released","launched"]


def search(query: str, force: bool = False) -> list[dict]:
    """
    Route query through full stack. Reformulate if primary returns nothing.
    Returns list of {title, body, source} dicts.
    Applies sanitization and rate limiting when search_sanitizer is available.
    """
    # ── Refresh keys every call — keys.json may be updated without restart ──
    global KEYS
    KEYS = load_keys()

    # ── Disclose search query ──
    _last_search_query[0] = query

    # ── Cache check ──
    if _SANITIZER_AVAILABLE and not force:
        cached = _search_cache.get(query)
        if cached is not None:
            return cached

    # ── Rate limiter ──
    if _SANITIZER_AVAILABLE and not force:
        allowed, reason = _search_limiter.allow()
        if not allowed:
            print(f"  🛡 Search blocked: {reason}")
            return []

    results = []

    # Route by type
    if _is_video(query):
        results += _youtube(query)

    if _is_social(query):
        results += _google_custom(f"{query}", n=5)
        results += _brave(query, n=3)

    elif _is_academic(query):
        if _is_medical(query):
            results += _pubmed(query)
        results += _openalex(query)
        results += _arxiv(query)
        results += _semantic_scholar(query)
        results += _crossref(query, n=2)
        # Also try Brave for ResearchGate hits
        results += _brave(f"site:researchgate.net {query}", n=3)

    else:
        # General: date-aware Brave + Wikipedia + DDG fallback
        _is_time_sensitive = any(s in query.lower() for s in _TIME_SIGNALS)
        if _is_time_sensitive:
            from datetime import datetime as _dt
            _date_suffix = _dt.now().strftime("%B %Y")
            results += _brave(f"{query} {_date_suffix}", n=5)
        else:
            results += _brave(query)
        if not results:
            results += _duckduckgo(query, n=8)
        # DDG secondary pass for thin results
        if len(results) < 3:
            results += _duckduckgo(f"{query} site:reddit.com OR site:stackoverflow.com", n=5)
        # Always enrich general queries with Wikipedia
        results += _wikipedia(query, n=2)
        # Archive.org for text/document queries
        if any(s in query.lower() for s in ["book","text","download","full text","pdf"]):
            results += _internet_archive(query, n=2)

    # Code queries get GitHub + Stack Exchange
    if _is_code(query):
        results += _github(query)
        results += _stack_exchange(query)

    # Tech/startup/HN queries
    if _is_code(query) or any(s in query.lower() for s in
                               ["startup","product","launch","hacker news","ycombinator"]):
        results += _hackernews(query)

    # Medical queries (standalone, outside academic branch)
    if _is_medical(query) and not _is_academic(query):
        results += _pubmed(query)

    # If empty — reformulate and retry
    if not results:
        for alt_query in reformulate_query(query):
            if _is_academic(query):
                results += _arxiv(alt_query)
                results += _semantic_scholar(alt_query)
                results += _brave(f"site:researchgate.net {alt_query}", n=2)
            else:
                results += _brave(alt_query, n=3)
                results += _duckduckgo(alt_query, n=3)
            if results:
                break  # stop at first reformulation that returns something

    # Deduplicate by title
    seen = set()
    deduped = []
    for r in results:
        # Use (title, url) pair — truncated title alone causes false collisions on long paper titles
        key = (r.get("title", ""), r.get("url", r.get("body", "")[:80]))
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    # Enrich top results with fetched content
    for _r in deduped[:3]:
        if _r.get("url") and len(_r.get("body","")) < 100:
            fetched = _fetch_page_content(_r["url"], 1500)
            if fetched:
                _r["body"] = fetched[:400]

    # Sort by combined quality score (trust + relevance + recency boost)
    if _SANITIZER_AVAILABLE:
        try:
            from search_sanitizer import trust_score as _ts, relevance_score as _rs
        except ImportError:
            def _ts(url): return 0.5
            def _rs(q, r): return 0.5
        import re as _re
        _cur_year = str(__import__('datetime').datetime.now().year)
        _prev_year = str(__import__('datetime').datetime.now().year - 1)

        def _quality_score(r, q):
            t = _ts(r) * 0.6 + _rs(q, r) * 0.4
            text = r.get("title","") + r.get("body","")
            if _cur_year in text or _prev_year in text:
                t += 0.08
            return t

        deduped.sort(key=lambda r: _quality_score(r, query), reverse=True)

    # Sanitize — strip injections, spam, HTML
    if _SANITIZER_AVAILABLE:
        deduped = sanitize_results(deduped)
        _search_limiter.record_result(len(deduped))
        _search_cache.put(query, deduped[:8])

    return deduped[:8]


# ── Intent-based source dispatch ────────────────────────────────────────────

_SOURCE_FN = {
    "brave":            _brave,
    "wikipedia":        _wikipedia,
    "ddg":              _duckduckgo,
    "openalex":         _openalex,
    "arxiv":            _arxiv,
    "semantic_scholar":  _semantic_scholar,
    "crossref":         _crossref,
    "pubmed":           _pubmed,
    "hackernews":       _hackernews,
    "brave_dated":      lambda q, n=5: _brave(
        f"{q} {__import__('datetime').datetime.now().strftime('%B %Y')}", n),
}


def search_by_intent(query: str, sources: list[str] | None) -> list[dict]:
    """Search only the sources specified by the intent router.

    If sources is None, returns [] (caller should not have called).
    Applies the same dedup/sanitize/cache pipeline as search().
    """
    if not sources:
        return []

    # Refresh keys
    global KEYS
    KEYS = load_keys()
    _last_search_query[0] = query

    # Cache check
    cache_key = f"intent:{','.join(sorted(sources))}:{query}"
    if _SANITIZER_AVAILABLE:
        cached = _search_cache.get(cache_key)
        if cached is not None:
            return cached
        allowed, reason = _search_limiter.allow()
        if not allowed:
            print(f"  🛡 Search blocked: {reason}")
            return []

    results = []
    for src in sources:
        fn = _SOURCE_FN.get(src)
        if fn:
            results += fn(query)

    # Deduplicate
    seen = set()
    deduped = []
    for r in results:
        key = (r.get("title", ""), r.get("url", r.get("body", "")[:80]))
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    # Enrich thin results
    for _r in deduped[:3]:
        if _r.get("url") and len(_r.get("body", "")) < 100:
            fetched = _fetch_page_content(_r["url"], 1500)
            if fetched:
                _r["body"] = fetched[:400]

    # Sort by quality
    if _SANITIZER_AVAILABLE:
        import re as _re
        _cur_year = str(__import__('datetime').datetime.now().year)
        _prev_year = str(__import__('datetime').datetime.now().year - 1)

        def _quality_score(r, q):
            t = trust_score(r) * 0.6 + relevance_score(q, r) * 0.4
            text = r.get("title", "") + r.get("body", "")
            if _cur_year in text or _prev_year in text:
                t += 0.08
            return t

        deduped.sort(key=lambda r: _quality_score(r, query), reverse=True)
        deduped = sanitize_results(deduped)
        _search_limiter.record_result(len(deduped))
        _search_cache.put(cache_key, deduped[:8])

    return deduped[:8]


def should_search(msg: str) -> bool:
    """
    Returns True if the message should trigger a web search.
    Delegates to should_search_v2 (intent-based) when sanitizer is available,
    falls back to keyword matching otherwise.
    """
    if _SANITIZER_AVAILABLE:
        result, _ = should_search_v2(msg)
        return result
    # Fallback: original keyword matching
    if msg.startswith("/search "):
        return True
    triggers = [
        "what is","who is","when did","where is","how does","how do",
        "latest","news","current","today","2024","2025","2026",
        "tell me about","find","look up","search","show me",
        "paper","research","study","published","youtube","video",
        "instagram","linkedin","grace","researchgate","arxiv",
    ]
    return any(t in msg.lower() for t in triggers)


def format_results(results: list, query: str = "") -> str:
    """
    Format search results for prompt injection.
    Uses safe framing with trust scores when sanitizer is available,
    falls back to plain format otherwise.
    """
    if not results:
        return ""
    if _SANITIZER_AVAILABLE:
        consensus = compute_consensus(results)
        return format_results_safe(results, consensus, query=query)
    # Fallback: plain format
    lines = ["[SEARCH RESULTS]"]
    for r in results:
        src   = r.get("source","web")
        title = r.get("title","")
        body  = r.get("body","")[:300]
        url   = r.get("url","")
        lines.append(f"[{src}] {title}")
        if body:
            lines.append(f"  {body}")
        if url:
            lines.append(f"  {url}")
    lines.append("[END SEARCH RESULTS]")
    return "\n".join(lines)
