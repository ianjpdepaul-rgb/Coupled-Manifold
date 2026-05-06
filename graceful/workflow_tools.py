"""Workflow tools — Phases 1-4.

Phase 1 (Group A): read_file, write_draft, list_directory, shell_readonly
Phase 2 (Group B): fetch_url, analyze_screenshot
Phase 3 (Group C): search_my_history, query_knowledge_graph, set_reminder
Phase 4 (Group D): draft_email

All filesystem paths validated against an allowlist using os.path.realpath()
to block symlink escapes.
"""

import glob as _glob_mod
import json
import os
import re
import shlex
import subprocess
import time
import urllib.parse
import webbrowser

# ── Allowlisted directories ──────────────────────────────────────────────────
# Tools can only touch paths under these roots.
_HOME = os.path.expanduser("~")

ALLOWED_ROOTS: list[str] = [
    os.path.join(_HOME, "Desktop"),
    os.path.join(_HOME, "Documents"),
    os.path.join(_HOME, "Downloads"),
    os.path.join(_HOME, "Code"),
    os.path.join(_HOME, "Projects"),
]

# Project dir added at init time (call init_project_root)
_project_root: str | None = None


def init_project_root(path: str):
    """Set the project root — called once at app startup."""
    global _project_root
    _project_root = os.path.realpath(path)


def _get_allowed_roots() -> list[str]:
    """Return all allowed roots (resolved to real paths)."""
    roots = [os.path.realpath(r) for r in ALLOWED_ROOTS]
    if _project_root:
        roots.append(_project_root)
    return roots


def _is_path_allowed(path: str) -> bool:
    """Check if a resolved path falls under any allowed root."""
    real = os.path.realpath(os.path.expanduser(path))
    return any(real.startswith(root + os.sep) or real == root
               for root in _get_allowed_roots())


def _resolve_and_validate(path: str) -> tuple[str | None, str]:
    """Resolve a path and validate it's allowed.

    Returns (resolved_path, error_message). If error, resolved_path is None.
    """
    if not path or not path.strip():
        return None, "No path provided."
    expanded = os.path.expanduser(path.strip())
    real = os.path.realpath(expanded)
    if not _is_path_allowed(real):
        return None, (f"Access denied: '{path}' is outside allowed directories. "
                      f"Allowed: ~/Desktop, ~/Documents, ~/Downloads, ~/Code, ~/Projects, "
                      f"and the project directory.")
    return real, ""


# ── read_file ────────────────────────────────────────────────────────────────
MAX_READ_BYTES = 200_000  # ~200KB cap


def read_file(arg: str) -> tuple[str, str]:
    """Read a file from the user's filesystem.

    arg: file path (absolute or ~-relative)
    Returns the file contents (truncated if over 200KB).
    """
    path, err = _resolve_and_validate(arg)
    if err:
        return (err, "")

    if not os.path.isfile(path):
        return (f"File not found: {arg}", "")

    try:
        size = os.path.getsize(path)
        with open(path, "r", errors="replace") as f:
            content = f.read(MAX_READ_BYTES)
        if size > MAX_READ_BYTES:
            content += f"\n\n[… truncated — file is {size:,} bytes, showed first {MAX_READ_BYTES:,}]"
        return (content, "")
    except IsADirectoryError:
        return (f"'{arg}' is a directory, not a file. Use list_directory instead.", "")
    except PermissionError:
        return (f"Permission denied: {arg}", "")
    except Exception as e:
        return (f"Error reading file: {e}", "")


# ── write_draft ──────────────────────────────────────────────────────────────
DRAFTS_SUBDIR = "drafts"


def _drafts_dir() -> str:
    """Return the drafts directory path (inside the project data dir)."""
    data = os.environ.get("GRACEFUL_DATA", "manifold_data")
    d = os.path.join(data, DRAFTS_SUBDIR)
    os.makedirs(d, exist_ok=True)
    return d


def write_draft(arg: str) -> tuple[str, str]:
    """Write a draft file to the managed drafts directory.

    arg: JSON with 'filename', 'content', and optional 'category'.
         Category creates a subdirectory (e.g. "code", "notes", "essays").
    """
    if not arg.strip():
        return ("No argument provided. Expected JSON with 'filename' and 'content'.", "")

    try:
        parsed = json.loads(arg)
    except json.JSONDecodeError as e:
        return (f"Invalid JSON: {e}", "")

    filename = parsed.get("filename", "").strip()
    content = parsed.get("content", "")
    category = parsed.get("category", "").strip()

    if not filename:
        return ("Missing 'filename' field.", "")
    if not content:
        return ("Missing 'content' field.", "")

    # Sanitize filename — no path traversal
    basename = os.path.basename(filename)
    if not basename or basename.startswith("."):
        return (f"Invalid filename: '{filename}'", "")

    base = _drafts_dir()
    if category:
        # Sanitize category too
        cat = os.path.basename(category)
        base = os.path.join(base, cat)
        os.makedirs(base, exist_ok=True)

    out_path = os.path.join(base, basename)

    try:
        with open(out_path, "w") as f:
            f.write(content)
        rel = os.path.relpath(out_path)
        return (f"Draft saved: {rel} ({len(content)} chars)", "")
    except Exception as e:
        return (f"Error writing draft: {e}", "")


# ── list_directory ───────────────────────────────────────────────────────────
MAX_ENTRIES = 200


def list_directory(arg: str) -> tuple[str, str]:
    """List contents of a directory.

    arg: directory path (absolute or ~-relative). Defaults to project root if empty.
    """
    if not arg.strip():
        # Default to project root
        target = _project_root or os.getcwd()
    else:
        target, err = _resolve_and_validate(arg)
        if err:
            return (err, "")

    if not os.path.isdir(target):
        if os.path.isfile(target):
            return (f"'{arg}' is a file, not a directory. Use read_file instead.", "")
        return (f"Directory not found: {arg}", "")

    try:
        entries = sorted(os.listdir(target))
    except PermissionError:
        return (f"Permission denied: {arg}", "")
    except Exception as e:
        return (f"Error listing directory: {e}", "")

    if not entries:
        return (f"Directory is empty: {arg}", "")

    lines = []
    for i, name in enumerate(entries):
        if i >= MAX_ENTRIES:
            lines.append(f"… and {len(entries) - MAX_ENTRIES} more entries")
            break
        full = os.path.join(target, name)
        suffix = "/" if os.path.isdir(full) else ""
        try:
            size = os.path.getsize(full) if not os.path.isdir(full) else None
            size_str = f"  ({size:,} bytes)" if size is not None else ""
        except OSError:
            size_str = ""
        lines.append(f"  {name}{suffix}{size_str}")

    header = f"{target}/  ({len(entries)} entries)"
    return (header + "\n" + "\n".join(lines), "")


# ── shell_readonly ───────────────────────────────────────────────────────────
ALLOWED_COMMANDS = {
    "ls", "cat", "head", "tail", "grep", "find", "wc",
    "file", "du", "df", "git",
}

# git subcommands that are read-only
ALLOWED_GIT_SUBCOMMANDS = {"status", "log", "diff", "branch", "show", "shortlog"}

SHELL_TIMEOUT = 10  # seconds


def shell_readonly(arg: str) -> tuple[str, str]:
    """Execute a read-only shell command.

    arg: the command string (e.g. "ls -la ~/Desktop", "git log --oneline -5")
    Only allowlisted commands are permitted. No writes, no pipes, no redirects.
    """
    if not arg.strip():
        return ("No command provided.", "")

    cmd = arg.strip()

    # Block shell metacharacters — no pipes, redirects, subshells, semicolons
    dangerous = set("|;&`$(){}><")
    if any(c in cmd for c in dangerous):
        return ("Shell metacharacters not allowed (no pipes, redirects, or chaining).", "")

    # Parse the command
    try:
        parts = shlex.split(cmd)
    except ValueError as e:
        return (f"Could not parse command: {e}", "")

    if not parts:
        return ("Empty command.", "")

    base_cmd = parts[0]

    # Validate command is allowlisted
    if base_cmd not in ALLOWED_COMMANDS:
        return (f"Command '{base_cmd}' is not allowed. "
                f"Allowed: {', '.join(sorted(ALLOWED_COMMANDS))}", "")

    # For git, validate the subcommand
    if base_cmd == "git":
        if len(parts) < 2:
            return ("git requires a subcommand (e.g. git status, git log).", "")
        sub = parts[1]
        if sub not in ALLOWED_GIT_SUBCOMMANDS:
            return (f"git {sub} is not allowed. "
                    f"Read-only git commands: {', '.join(sorted(ALLOWED_GIT_SUBCOMMANDS))}", "")

    # Determine working directory — use project root if set
    cwd = _project_root or os.getcwd()

    # For commands that take file paths, validate them
    # (ls, cat, head, tail, find, du, file, grep can all take path args)
    # We check any argument that looks like a path
    for part in parts[1:]:
        if part.startswith("-"):
            continue  # skip flags
        # Check if it looks like a path (contains / or ~)
        if "/" in part or part.startswith("~"):
            expanded = os.path.expanduser(part)
            if not os.path.isabs(expanded):
                expanded = os.path.join(cwd, expanded)
            if not _is_path_allowed(expanded):
                return (f"Access denied: path '{part}' is outside allowed directories.", "")

    try:
        result = subprocess.run(
            parts,
            capture_output=True,
            text=True,
            timeout=SHELL_TIMEOUT,
            cwd=cwd,
        )
        output = result.stdout
        if result.stderr:
            output += ("\n[stderr]\n" + result.stderr) if output else result.stderr
        if not output.strip():
            output = "(no output)"
        # Cap output
        if len(output) > MAX_READ_BYTES:
            output = output[:MAX_READ_BYTES] + "\n[… output truncated]"
        return (output, "")
    except subprocess.TimeoutExpired:
        return (f"Command timed out after {SHELL_TIMEOUT}s.", "")
    except FileNotFoundError:
        return (f"Command not found: {base_cmd}", "")
    except Exception as e:
        return (f"Error executing command: {e}", "")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Web and content (Group B)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_url(arg: str) -> tuple[str, str]:
    """Fetch and extract text from a URL.

    arg: the URL to fetch
    Returns extracted text content (HTML tags stripped), capped at 200KB.
    """
    url = arg.strip()
    if not url:
        return ("No URL provided.", "")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        import httpx
    except ImportError:
        return ("httpx not installed — cannot fetch URLs.", "")

    try:
        from html.parser import HTMLParser

        class _TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.parts: list[str] = []
                self._skip = False

            def handle_starttag(self, tag, attrs):
                if tag in ("script", "style", "noscript"):
                    self._skip = True

            def handle_endtag(self, tag):
                if tag in ("script", "style", "noscript"):
                    self._skip = False

            def handle_data(self, data):
                if not self._skip:
                    self.parts.append(data)

        resp = httpx.get(url, timeout=10.0, follow_redirects=True,
                         headers={"User-Agent": "Graceful/1.0"})
        resp.raise_for_status()

        ct = resp.headers.get("content-type", "")
        if "text/html" in ct or "application/xhtml" in ct:
            parser = _TextExtractor()
            parser.feed(resp.text)
            text = " ".join(parser.parts)
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
        else:
            text = resp.text

        if len(text) > MAX_READ_BYTES:
            text = text[:MAX_READ_BYTES] + "\n\n[… truncated]"
        return (text, "")
    except httpx.HTTPStatusError as e:
        return (f"HTTP error {e.response.status_code}: {url}", "")
    except httpx.TimeoutException:
        return (f"Timeout fetching {url}", "")
    except Exception as e:
        return (f"Error fetching URL: {e}", "")


def analyze_screenshot(arg: str) -> tuple[str, str]:
    """Run OCR on an image file and return extracted text.

    arg: path to image file
    Uses tesseract OCR if available, falls back to basic file info.
    """
    path, err = _resolve_and_validate(arg)
    if err:
        return (err, "")

    if not os.path.isfile(path):
        return (f"File not found: {arg}", "")

    # Check it's an image
    ext = os.path.splitext(path)[1].lower()
    image_exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}
    if ext not in image_exts:
        return (f"Not a recognized image format: {ext}", "")

    lines = [f"Image: {os.path.basename(path)}"]
    try:
        size = os.path.getsize(path)
        lines.append(f"Size: {size:,} bytes")
    except OSError:
        pass

    # Try tesseract OCR
    try:
        result = subprocess.run(
            ["tesseract", path, "stdout", "-l", "eng"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            lines.append(f"\nExtracted text:\n{result.stdout.strip()}")
        else:
            lines.append("\n(No text extracted — image may not contain readable text)")
    except FileNotFoundError:
        lines.append("\n(tesseract not installed — install with: brew install tesseract)")
    except subprocess.TimeoutExpired:
        lines.append("\n(OCR timed out)")
    except Exception as e:
        lines.append(f"\n(OCR error: {e})")

    return ("\n".join(lines), "")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — History and memory (Group C)
# ══════════════════════════════════════════════════════════════════════════════

def _sessions_dir() -> str:
    data = os.environ.get("GRACEFUL_DATA", "manifold_data")
    return os.path.join(data, "sessions")


def search_my_history(arg: str) -> tuple[str, str]:
    """Search across all session logs for a topic or thread.

    arg: JSON with 'query' (required) and optional 'time_range' (today|week|month|all).
         Or just a plain text query string.
    Returns matching exchanges with session IDs, dates, and context.
    """
    if not arg.strip():
        return ("No search query provided.", "")

    # Accept either JSON or plain text
    query = arg.strip()
    time_range = "all"
    try:
        parsed = json.loads(arg)
        query = parsed.get("query", "").strip()
        time_range = parsed.get("time_range", "all").strip().lower()
    except (json.JSONDecodeError, AttributeError):
        pass

    if not query:
        return ("No search query provided.", "")

    # Determine time cutoff
    now = time.time()
    cutoffs = {
        "today": now - 86400,
        "week": now - 7 * 86400,
        "month": now - 30 * 86400,
        "all": 0,
    }
    cutoff = cutoffs.get(time_range, 0)

    sessions_dir = _sessions_dir()
    if not os.path.isdir(sessions_dir):
        return ("No session history found.", "")

    query_lower = query.lower()
    matches = []

    for fname in sorted(os.listdir(sessions_dir)):
        if not fname.startswith("session_") or not fname.endswith(".json"):
            continue

        # Extract timestamp from filename
        try:
            ts = int(fname.replace("session_", "").replace(".json", ""))
        except ValueError:
            continue

        if ts < cutoff:
            continue

        fpath = os.path.join(sessions_dir, fname)
        try:
            with open(fpath) as f:
                sess = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        for turn in sess.get("turns", []):
            user_msg = turn.get("user", "")
            response = turn.get("response", "")
            if query_lower in user_msg.lower() or query_lower in response.lower():
                date_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
                matches.append({
                    "session": ts,
                    "date": date_str,
                    "turn": turn.get("turn", "?"),
                    "user": user_msg[:200],
                    "response": response[:200],
                })

    if not matches:
        return (f"No matches for '{query}' in {time_range} history.", "")

    # Cap results
    if len(matches) > 20:
        matches = matches[-20:]

    lines = [f"Found {len(matches)} matches for '{query}':"]
    for m in matches:
        lines.append(f"\n[{m['date']} · session {m['session']} · turn {m['turn']}]")
        lines.append(f"  User: {m['user']}")
        lines.append(f"  Response: {m['response']}")

    return ("\n".join(lines), "")


def query_knowledge_graph(arg: str) -> tuple[str, str]:
    """Query the identity/knowledge graph for concepts and thinkers.

    arg: search query string
    Returns matching concepts, thinkers, with weights and recent activation.
    """
    query = arg.strip().lower()
    if not query:
        return ("No query provided.", "")

    data_dir = os.environ.get("GRACEFUL_DATA", "manifold_data")
    identity_path = os.path.join(data_dir, "identity.json")

    if not os.path.isfile(identity_path):
        return ("No knowledge graph data found.", "")

    try:
        with open(identity_path) as f:
            identity = json.load(f)
    except (json.JSONDecodeError, OSError):
        return ("Could not read knowledge graph.", "")

    concepts = identity.get("concepts", [])
    thinkers = identity.get("thinkers", [])
    node_weights = identity.get("node_weights", {})
    raw_notes = identity.get("raw_notes", [])

    # Search concepts
    matched_concepts = [c for c in concepts if query in c.lower()]
    matched_thinkers = [t for t in thinkers if query in t.lower()]
    matched_notes = [n for n in raw_notes if query in n.lower()]

    if not matched_concepts and not matched_thinkers and not matched_notes:
        return (f"No matches for '{query}' in knowledge graph.", "")

    lines = []
    if matched_concepts:
        lines.append(f"Concepts ({len(matched_concepts)}):")
        for c in matched_concepts[:15]:
            w = node_weights.get(c, 0)
            lines.append(f"  · {c} (weight: {w:.2f})" if w else f"  · {c}")

    if matched_thinkers:
        lines.append(f"\nThinkers ({len(matched_thinkers)}):")
        for t in matched_thinkers[:10]:
            w = node_weights.get(t, 0)
            lines.append(f"  · {t} (weight: {w:.2f})" if w else f"  · {t}")

    if matched_notes:
        lines.append(f"\nNotes ({len(matched_notes)}):")
        for n in matched_notes[:10]:
            lines.append(f"  · {n[:150]}")

    return ("\n".join(lines), "")


_REMINDERS_DIR: str | None = None


def _get_reminders_dir() -> str:
    if _REMINDERS_DIR:
        return _REMINDERS_DIR
    return os.path.expanduser("~/Documents/Graceful_reminders")


def set_reminder(arg: str) -> tuple[str, str]:
    """Create a reminder that persists to disk.

    arg: JSON with 'text' (required) and optional 'trigger_context'
         (e.g. "next session", "when discussing X", "tomorrow").
    """
    if not arg.strip():
        return ("No reminder content provided. Expected JSON with 'text'.", "")

    try:
        parsed = json.loads(arg)
    except json.JSONDecodeError as e:
        return (f"Invalid JSON: {e}", "")

    text = parsed.get("text", "").strip()
    trigger = parsed.get("trigger_context", "next session").strip()

    if not text:
        return ("Missing 'text' field.", "")

    reminders_dir = _get_reminders_dir()
    os.makedirs(reminders_dir, exist_ok=True)

    reminder = {
        "text": text[:1000],
        "trigger_context": trigger[:200],
        "created": time.time(),
        "created_human": time.strftime("%Y-%m-%d %H:%M"),
        "fired": False,
        "dismissed": False,
    }

    # Append to reminders file
    reminders_path = os.path.join(reminders_dir, "reminders.jsonl")
    with open(reminders_path, "a") as f:
        f.write(json.dumps(reminder) + "\n")

    return (f"Reminder set: \"{text[:80]}\" (trigger: {trigger})", "")


def get_pending_reminders() -> list[dict]:
    """Return all unfired, undismissed reminders. Used by proactive system."""
    reminders_dir = _get_reminders_dir()
    path = os.path.join(reminders_dir, "reminders.jsonl")
    if not os.path.isfile(path):
        return []

    pending = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if not r.get("fired") and not r.get("dismissed"):
                        pending.append(r)
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return pending


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Communication (Group D)
# ══════════════════════════════════════════════════════════════════════════════

def draft_email(arg: str) -> tuple[str, str]:
    """Open the default mail client with a pre-composed draft.

    arg: JSON with 'to', 'subject', and 'body'. All optional but at least one needed.
    Uses mailto: URL scheme — user decides whether to send.
    """
    if not arg.strip():
        return ("No argument provided. Expected JSON with 'to', 'subject', 'body'.", "")

    try:
        parsed = json.loads(arg)
    except json.JSONDecodeError as e:
        return (f"Invalid JSON: {e}", "")

    to = parsed.get("to", "").strip()
    subject = parsed.get("subject", "").strip()
    body = parsed.get("body", "").strip()

    if not to and not subject and not body:
        return ("At least one of 'to', 'subject', or 'body' is required.", "")

    # Build mailto: URL
    params = {}
    if subject:
        params["subject"] = subject
    if body:
        params["body"] = body

    mailto = f"mailto:{urllib.parse.quote(to)}"
    if params:
        mailto += "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)

    try:
        webbrowser.open(mailto)
        parts = []
        if to:
            parts.append(f"To: {to}")
        if subject:
            parts.append(f"Subject: {subject}")
        parts.append(f"Body: {len(body)} chars")
        return (f"Email draft opened in mail client.\n" + "\n".join(parts), "")
    except Exception as e:
        return (f"Error opening mail client: {e}", "")
