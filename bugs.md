# Graceful Bug Audit — 2026-05-05

Systematic debug run across app.py, memory.py, search_stack.py, search_sanitizer.py,
index.html. Stages 1–3 complete. Stages 4–5 (synthetic chat agent, comparison runs)
require a running instance and are deferred to a follow-up pass.

## Fix Pass Status — 2026-05-05

All P0 and P1 bugs fixed. P2 bugs fixed (except #15/#17/#18 which were already resolved
or no-ops). P3 items are style/false-positive — no action needed.

| # | Priority | Status | Commit |
|---|----------|--------|--------|
| 1 | P0 | ✅ Fixed | `76c7ed7` — finally block in _stream_chat |
| 2 | P0 | ✅ Fixed | `b6e6b62` — _session_lock in /api/session_turns |
| 3 | P0 | ✅ Fixed | `790305b` — stderr logging on tokenizer encode |
| 4 | P1 | ✅ Fixed | `3cb73ae` — removed dead deque |
| 5 | P1 | ✅ Fixed | `bca2044` — removed dead summary_input |
| 6 | P1 | ✅ Fixed | `761458c` — removed unused body parse |
| 7 | P1 | ✅ Fixed | `d823605` — trust boundary comment on exec() |
| 8 | P1 | ✅ Fixed | `45d049c` — trust boundary comment on pickle.load |
| 9 | P1 | ✅ Fixed | `a57808c` — shell=False in launch.py |
| 10 | P1 | ✅ Fixed | `114675b` — bg trace error logging |
| 11 | P1 | ✅ Fixed | `8112665` — core-path exception logging |
| 12 | P2 | ✅ Fixed | `1170f80` — locked all _session_history reads |
| 13 | P2 | ✅ Fixed | `9b82d83` — locked _active_session_ts in api_fork/reset |
| 14 | P2 | ✅ Fixed | `d27a302` — added _pinned_context_lock |
| 15 | P2 | N/A | urlopen — hardcoded URLs, no user input vector |
| 16 | P2 | ✅ Fixed | `7844f5f` — md5 usedforsecurity=False |
| 17 | P2 | ✅ Already fixed | sendEdit variable — fixed in polish pass `e429189` |
| 18 | P2 | ✅ Already fixed | mode leak — fixed in polish pass `e429189` |
| 19 | P3 | No action | sympy imports — intentional for exec() namespace |
| 20 | P3 | No action | datetime re-import — function-local, harmless |
| 21–31 | P3 | No action | Style / false positives |

Additional fix not in original audit:
- `1b03c9f` — clear `_recent_response_vecs` on session reset (/api/new, /api/reset)

---

## P0 — Fix immediately

### 1. `_user_request_pending` counter never decremented for commands
**File:** `app.py:6172, 5064`
**Impact:** Background trace computation and online learning permanently suppressed after enough slash commands.

`_user_request_pending[0]` is incremented at every `/api/chat`, `/api/regen`, `/api/edit` entry
(lines 6172, 6188, 6460). It is only decremented when `_model_lock.acquire()` is reached (line 5064).
Commands like `/stats`, `/search`, `/export`, `/check`, `/who`, `/save`, `/find`, `/experiment`, etc.
return early from `chat()` via `yield; return` — never reaching `_model_lock.acquire()`.

Each such command inflates the counter by 1. Once it's >0, all background tasks
(`_bg_trace`, `_bg_learn`, `_bg_learn_step`, `check_background_health`) check
`if _user_request_pending[0] > 0: return` and silently abort.

**Repro:** Send 5 slash commands (`/stats`, `/who`, `/check`, `/export`, `/stats`) in a session.
The counter will be 5. All background trace/learn threads will skip forever.

**Fix:** Decrement `_user_request_pending[0]` at the top of every early-return command handler,
or (better) restructure so the increment only happens when generation will actually occur.


### 2. `_session_history` read without lock in `/api/session_turns` (line 6903)
**File:** `app.py:6903`
**Impact:** Potential data race — iteration over `_session_history` while another thread
mutates it (e.g., `_stream_chat` doing `_session_history[:] = last_hist`).

`list(_session_history)` is used (a snapshot), which mitigates the worst case
but is not atomic if the source list is being modified concurrently.

**Fix:** Wrap in `with _session_lock:`.


### 3. Tokenizer encode failure silently produces empty trace
**File:** `app.py:5171-5172`
**Impact:** If `tok.encode()` raises, `_out_token_list = []` → `out_ids` is empty →
`compute_trace` is called with no tokens → trace is 0.0 or skipped → SnobLine
controller sees no signal → learning gate never opens.

Silent failure in the core trace computation path. No log, no print.

**Fix:** Add at minimum a `print()` in the except block so the failure is visible in logs.


---

## P1 — Fix in next pass

### 4. `_recent_response_vecs` declared twice with different types
**File:** `app.py:1647, 1912`
**Impact:** Line 1647 declares `collections.deque(maxlen=5)`. Line 1912 overwrites it
with `[]` (plain list). The deque's auto-eviction is lost. Functionally OK because
lines 1944–1945 manually cap with `pop(0)`, but `pop(0)` on a list is O(n) and
the deque declaration is misleading dead code.

**Fix:** Remove line 1647 (dead code). Or convert line 1912 to `deque(maxlen=5)` and
remove the manual pop.


### 5. `summary_input` built but never used
**File:** `app.py:2948`
**Impact:** Dead code. `summary_input` is constructed from session log turns but never
passed to any model or function. Likely a remnant from when `/summary` was going to
use the model for extractive summarization.

**Fix:** Remove the dead variable.


### 6. `data` parsed but never used in `api_save`
**File:** `app.py:6209-6210`
**Impact:** `api_save` reads and parses the request body but discards it. The function
saves whatever is in `_session_history`, ignoring the client payload entirely. If the
client sends history in the body (e.g., from `beforeunload`), it's silently ignored.

**Fix:** Either use `data` (to save client-side history as a fallback) or remove the
parse entirely to avoid confusion.


### 7. `exec()` used for code execution with no filesystem sandbox
**File:** `app.py:1140`
**Impact:** The `/run` feature executes user code via `exec()`. The namespace is
restricted (no `__builtins__`), but there's no filesystem isolation — code can
`open('/etc/passwd')`, make network requests, or access `os` if injected via imports.
This is local-only so the blast radius is limited to the user's own machine, but
it's still worth documenting.

Bandit flags: B102 (exec_used).

**Mitigation:** Local-only app, intentional feature. Add a comment documenting the
trust boundary.


### 8. `pickle.load()` on session files without integrity check
**File:** `memory.py:550`
**Impact:** `ConversationHistory` loads pickled data from disk. If the pickle file is
corrupted or tampered with, arbitrary code execution is possible.

Bandit flags: B301 (pickle).

**Mitigation:** Local-only app, files are self-generated. No external input path to
the pickle files. Low real-world risk but worth noting.


### 9. `subprocess.Popen(shell=True)` in launcher
**File:** `launch.py:515`
**Impact:** The launcher runs commands with `shell=True`. The command string is
constructed from internal constants, not user input, so injection risk is minimal.

Bandit flags: B602.

**Fix:** Switch to `shell=False` with a list argument.


### 10. Silent trace failure in background thread
**File:** `app.py:5246-5247`
**Impact:** If `compute_trace_for_model()` raises in the background trace thread,
the exception is silently caught and the trace result is lost. No log entry.
The SnobLine controller sees stale data.

**Fix:** Add `print(f"bg trace error: {e}")` to the except block.


### 11. 99 silent exception handlers in app.py
**File:** `app.py` (various)
**Impact:** 99 except blocks that either `pass` or handle without logging. Most are
in non-critical paths (log file writes, optional features), but some are in core
paths (tokenizer, trace, prompt assembly). The total count makes it hard to
distinguish intentional defensive coding from hidden bugs.

**Priority handlers to audit (in core paths):**
- Line 5023: chat template fallback (could hide model incompatibility)
- Line 5171: tokenizer encode (trace path — P0 #3 above)
- Line 5246: background trace computation (P1 #10 above)
- Line 3260, 5416, 7096, 7189: model forward pass fallbacks (could hide OOM)
- Line 1947: `compute_drift()` returns `False, 0.0` on error (hides embedding failures)

**Low-priority (non-critical logging):**
- Lines 5381, 5394: learn_decisions.jsonl write failures
- Lines 5214, 5652, 5689: trace log writes
- Lines 6624, 6640, 6665: file cleanup (temp files)

**Fix:** Add `print()` or `logging.warning()` to the core-path handlers. Leave the
log-write handlers as-is (they're correctly defensive).


---

## P2 — Investigate later

### 12. `_session_history` accessed without lock in multiple read paths
**File:** `app.py:6342, 6364, 6130, 6148, 6164, 6177`
**Impact:** Several API endpoints read `_session_history` without `_session_lock`.
CPython's GIL makes single-operation reads safe, but multi-step reads (like iterating
and checking length) are not atomic.

Most of these are in session-management endpoints that are called when no generation
is active, so the practical risk is low.

**Locations:**
- `api_init` (line 6130): reads for response
- `api_load_session` (line 6342): reads after lock release
- `api_session_turns` (line 6903): copies without lock → P0 #2


### 13. `_active_session_ts` not always under lock
**File:** `app.py:6304, 6232, 6340`
**Impact:** `_active_session_ts[0]` is read and written from multiple endpoints.
Some writes are inside `_session_lock` (line 6340), others aren't (line 6304).
Could cause session file to be saved under wrong ID if a race occurs between
`api_fork` and `_save_session`.


### 14. `_pinned_context` has no thread safety
**File:** `app.py:2199`
**Impact:** `_pinned_context` is a plain list, mutated by `/pin` and `/unpin` commands,
read during prompt assembly. No lock protects it. If a user sends `/pin` while
generation is mid-prompt-assembly, the list could be modified during iteration.

Practical risk: very low (users don't type commands during generation).


### 15. `urllib.request.urlopen()` for model downloads
**File:** `app.py:325`
**Impact:** Downloads model assets over HTTP(S) with `ssl.create_default_context()`.
The `urlopen` call accepts any URL scheme. Bandit flags this for `file://` scheme
abuse. In practice, URLs are hardcoded in `MANIFEST`.

Bandit flags: B310.


### 16. MD5 used for content hashing (non-security)
**File:** `ingest_concordance.py:149`, `search_sanitizer.py:480,540`
**Impact:** MD5 is used for content deduplication keys and cache keys, not for
security. No vulnerability, but `hashlib.md5(... usedforsecurity=False)` would
silence the bandit warning.

Bandit flags: B324.


### 17. `/api/edit` variable reference bug (already fixed in polish pass)
**File:** `manifold_data/static/index.html:1744` (was `text`, now `newText`)
**Impact:** Was referencing `text` (undefined in `sendEdit` scope). Fixed in commit
`e429189`. Listed here for completeness.


### 18. Socratic/compress mode leak across sessions (already fixed in polish pass)
**File:** `app.py:6249-6250` (added in commit `e429189`)
**Impact:** `_socratic_mode[0]` and `_compress_mode[0]` were not reset on `/api/new`.
Fixed. Listed here for completeness.


### 19. Multiple sympy imports marked unused
**File:** `app.py:1089-1096`
**Impact:** ~30 sympy symbols imported inside a try block. These are intentionally
injected into the `exec()` namespace for the `/calc` feature. Ruff correctly
flags them as unused since they're consumed indirectly.

Ruff flags: F401 (intentional).


### 20. `datetime` redefined at line 5780
**File:** `app.py:14, 5780`
**Impact:** `datetime` is imported at line 14 as the module, then re-imported at
line 5780. Both refer to the same module. Ruff flags: F811. Harmless.


---

## P3 — False positives / style issues

### 21. `_socket` reported as undefined (F821)
**File:** `app.py:102`
**Impact:** `_socket` is imported at line 93 (`import socket as _socket`). Ruff reports
F821 because the import is after module-level code (E402). False positive.

### 22. E501 line length violations (~200+ instances)
**Impact:** Style only. The codebase uses a dense style with long lines.

### 23. E701/E702 multiple statements per line (~50+ instances)
**Impact:** Style only. Dense one-liners like `ctrl.mode = "lora"; ctrl.consec_patho = 0`.

### 24. SIM102/SIM105/SIM115 simplification suggestions
**Impact:** Style only. Nested ifs, try-except-pass → contextlib.suppress, etc.

### 25. RUF005 iterable unpacking suggestions
**Impact:** Style only. `list(a) + [b]` → `[*a, b]`.

### 26. B905 `zip()` without `strict=`
**Impact:** Style. The paired iterables are always the same length by construction.

### 27. B008 `Body()` in argument default
**File:** `app.py:6298`
**Impact:** FastAPI convention. `Body()` in function defaults is idiomatic FastAPI, not a bug.

### 28. B904 `raise` without `from` in except blocks
**File:** `app.py:6334, 6336, 6478, 6480`
**Impact:** Style. Missing `raise X from e` loses the exception chain but doesn't
affect functionality.

### 29. RUF001/RUF002 ambiguous Unicode characters
**Impact:** Intentional use of `×` (multiplication sign) and `−` (minus sign) in
display strings and docstrings.

### 30. F541 f-strings without placeholders
**Impact:** Style. Several f-strings that could be plain strings. No functional impact.

### 31. test_full_ux.py unpacked variables never used (RUF059)
**Impact:** Test file style. Unpacking `text` to verify structure but not asserting on it.

---

## Silent failures — notable entries from Stage 2

| Line | Except type | Body | Verdict |
|------|------------|------|---------|
| 5023 | `Exception` | fallback chat template | **P1** — could hide model incompatibility |
| 5171 | `Exception` | `_out_token_list = []` | **P0** — silent trace failure (#3) |
| 5246 | `Exception` | `pass` | **P1** — silent bg trace failure (#10) |
| 3260 | `Exception` | `out = mdl(inp)` fallback | P2 — masks OOM in LoRA step |
| 1947 | `Exception` | `return False, 0.0` | P2 — masks embedding failures in drift |
| 4567 | `Exception` | `_last_web_query = search_query` | P3 — harmless fallback |
| 4579 | `Exception` | `_ctx_route = "large"` | P3 — harmless fallback |
| 5381 | `Exception` | `pass` (log write) | P3 — defensive, correct |
| 6209 | `Exception` | `data = {}` | P3 — api_save body parse failure |

Total silent handlers: 99 in app.py, 7 in memory.py, 1 in search_stack.py, 3 in search_sanitizer.py.

---

## State consistency — Stage 3 summary

### Locks audit

| State variable | Lock | Protected? | Notes |
|----------------|------|-----------|-------|
| `_session_history` | `_session_lock` | Partially | Writes mostly locked; some reads unlocked (P0 #2, P2 #12) |
| `_active_session_ts` | `_session_lock` | Partially | Some writes unlocked (P2 #13) |
| `_model_lock` | (itself) | Yes | Properly acquired/released in all model paths |
| `_pending_trace` | `_pending_trace_lock` | Yes | All reads/writes under lock |
| `_user_request_pending` | None | **No** | Simple int in list — GIL-safe for single ops but counter drift bug (P0 #1) |
| `session_log` | None | No | Only written from main chat path; read from commands. GIL protects. |
| `trace_history_live` | None | No | Same pattern as session_log. |
| `_pinned_context` | None | No | Low risk (P2 #14) |
| `_socratic_mode` | None | No | Single-writer (command handler). Low risk. |
| `_compress_mode` | None | No | Same. |
| `_named_memory` | None | No | Written by /remember command, read by prompt assembly. Low risk. |
| `_recent_response_vecs` | None | No | Written during generation, read during diversity check. Same thread. |
| `_skip_search_this_turn` | None | No | Set by /continue, read once next turn. Same thread. |
| `online_learning` | None | No | Toggle flag. GIL-safe. |
| `system_prompt` | None | No | Written by settings API, read by chat. GIL-safe. |

### Session reset completeness

| State | Reset on /api/new? | Reset on /api/reset? |
|-------|-------------------|---------------------|
| `_session_history` | Yes (line 6231) | Yes (line 6291) |
| `session_log` | Yes (line 6236) | Yes (line 6267) |
| `turn_count` | Yes (line 6237) | Yes (line 6268) |
| `trace_history_live` | Yes (line 6238) | Yes (line 6269) |
| `ctrl.*` | Yes (lines 6239-6246) | Yes (lines 6270-6280) |
| `_socratic_mode` | Yes (line 6249) | Yes (line 6284) |
| `_compress_mode` | Yes (line 6250) | Yes (line 6285) |
| `_active_persona` | Conditional (line 6248) | Not checked |
| `_pinned_context` | **No** | **No** |
| `_named_memory` | **No** | **No** |
| `_skip_search_this_turn` | **No** (but auto-resets) | **No** |
| `_user_request_pending` | **No** (P0 #1) | **No** |
| `_recent_response_vecs` | **No** | **No** |

**Notable:** `_pinned_context` and `_named_memory` are intentionally persistent
across sessions (user-managed). `_recent_response_vecs` should probably reset on
new session (stale vectors from prior conversation affect diversity gate).

---

## Stages 4–5: Deferred

Stages 4 (synthetic chat agent) and 5 (comparison runs) require a running Graceful
instance with model loaded. The test harness design is outlined below for future use.

### Stage 4 design: `tests/chat_agent.py`

```python
# Connects to running instance, sends messages via /api/chat, reads SSE stream.
# Logs full response + medulla state per turn to JSONL.
#
# Personas:
#   normal_user       — varied conversational messages
#   recursive_continuer — /continue x10
#   search_spammer    — alternating search/non-search queries
#   mode_switcher     — rapid /socratic, /compress, /antagonist, /mode, normal
#   long_session      — 50 messages, varied content
#   edge_caser        — empty, very long, all-emoji, all-code messages
#
# Analysis:
#   - Scan logs for exceptions, trace anomalies, mode flips
#   - Check _user_request_pending counter drift (P0 #1 repro)
#   - Verify diversity gate scores
```

### Stage 5 design: comparison runs

Run same conversation under different conditions, diff logs.
Key comparisons: search on/off, personality default/custom, learning on/off,
normal/test mode, fresh/loaded session.

---

## Summary

| Priority | Count | Key items |
|----------|-------|-----------|
| P0 | 3 | Counter drift (#1), unlocked read (#2), silent trace (#3) |
| P1 | 8 | Duplicate decl (#4), dead code (#5-6), exec sandboxing (#7), pickle (#8), silent failures (#10-11) |
| P2 | 9 | Lock gaps (#12-14), urllib (#15), md5 (#16), already-fixed (#17-18), imports (#19-20) |
| P3 | 11 | False positives, style issues (#21-31) |
| **Total** | **31** | |
