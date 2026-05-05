# Audit: Current Session State — 2026-04-18

## 1. Change Inventory (index.html, session-related, today)

| # | Change | Lines | Status |
|---|--------|-------|--------|
| 1 | Added per-sid sessionStorage keys (`graceful_history_<sid>`) — renderHistory wrapper writes `sessionStorage.setItem("graceful_history_"+activeSid, ...)` | ~2905 | Active |
| 2 | Added `graceful_sid` write in renderHistory wrapper (`sessionStorage.setItem("graceful_sid", activeSid)`) | ~2905 | Active but **never read** |
| 3 | Removed legacy `sessionStorage.removeItem("graceful_history")` / `removeItem("graceful_sid")` from newChat() | ~1928 (deleted) | Done |
| 4 | Added legacy cleanup in init(): `sessionStorage.removeItem("graceful_history")` | ~2009 | Active |
| 5 | Moved sessionStorage restore in init() to AFTER `/api/init` fetch, keyed by `active_sid` | ~2020-2023 | Active |
| 6 | Added `_tempMood` guard: `typeof _tempMood==="function"` on line 2042 | ~2042 | Active |
| 7 | Added redundant DOM clear in newChat() before renderHistory() | ~1934 | Active (harmless) |
| 8 | Added redundant DOM clear in loadSession() before renderHistory() | ~1894 | Active (harmless) |

## 2. Current Function Behaviors

### init() (line 2007)

Order of operations:
1. Delete legacy `graceful_history` key from sessionStorage (line 2009)
2. Fetch `/api/init` with up to 8 retries (lines 2012-2015)
3. If fetch fails, show #empty div and return (line 2016)
4. Set `history = d.history`, `_allSessions = d.sessions`, `activeSid = d.active_sid` (line 2018)
5. **If history is empty AND activeSid exists**: read `sessionStorage.getItem("graceful_history_"+activeSid)`, if found and non-empty, set `history = parsed` (lines 2020-2023)
6. If no activeSid: auto-create session via `/api/new` (lines 2026-2028)
7. Render sidebar, set UI toggles (lines 2030-2048)
8. Call `renderHistory()` (line 2050) — this renders `history` array to DOM and triggers the wrapper
9. The wrapper (line 2902) saves `history` to `sessionStorage.graceful_history_<activeSid>` and sets `graceful_sid = activeSid`

**Key observation**: init() never reads `graceful_sid` from sessionStorage. It uses `d.active_sid` from the server to decide which sessionStorage key to check. `graceful_sid` is written but never consumed.

### newChat() (line 1924)

Order of operations:
1. If generating, stop and wait 80ms (lines 1925-1929)
2. POST `/api/new` → get `d` with empty history and new sid (line 1931)
3. Set `history = []`, `activeSid = d.active_sid` (line 1932)
4. Clear all `.msg-row` from DOM (line 1934) — redundant, renderHistory does this too
5. Call `renderHistory()` (line 1935) — renders empty, wrapper saves `[]` to `graceful_history_<new_sid>`, sets `graceful_sid = new_sid`
6. Update sidebar (line 1936)

**No issues here.** newChat() correctly clears state and renders empty.

### renderHistory() (line 1358, wrapped at line 2902)

**Base function (line 1358):**
- Reads from: `history` array (global variable)
- Writes to DOM: clears all `.msg-row` from `#msg-wrap`, then appends `buildRow()` for each entry in `history`
- If `history` empty: shows `#empty` div, hides load-older button

**Wrapper (line 2902):**
- Calls base function
- Writes: `sessionStorage.graceful_history_<activeSid>` = JSON of `history`
- Writes: `sessionStorage.graceful_sid` = `activeSid`

### loadSession(id) — sidebar click handler (line 1884)

Order of operations:
1. If `id === activeSid && history.length > 0`: **early return, does nothing** (line 1885)
2. If generating: early return (line 1886)
3. Auto-save current session via `/api/save` if history has ≥2 entries (line 1888)
4. GET `/api/sessions/<id>` (line 1890)
5. If 404: set `history=[]`, `activeSid=id`, renderHistory (line 1891)
6. Set `history = d.history`, `activeSid = id` (line 1893)
7. Clear DOM `.msg-row` (line 1894) — redundant
8. Call `renderHistory()` + `renderSessions()` (line 1895)

**Key observation**: `/api/sessions/<id>` on the server (line 6260) also sets `_session_history[:] = data.history` and `_active_session_ts[0] = session_id`. So the server switches its active session. This means subsequent `/api/init` calls will return this session's data.

### graceful_sid — all read/write locations

**Writes:**
- Line 2905: `sessionStorage.setItem("graceful_sid", activeSid)` — in renderHistory wrapper, on every render

**Reads:**
- **NOWHERE.** `graceful_sid` is written but never read by any code path.

### activeSid — all assignment locations

| Line | Context |
|------|---------|
| 1053 | Declaration: `activeSid=null` |
| 1272 | After `/api/clear` response |
| 1891 | loadSession 404 path |
| 1893 | loadSession success path |
| 1907 | deleteSession — deleted active, created new |
| 1918 | deleteSession — fallback path |
| 1932 | newChat() — from `/api/new` response |
| 1977 | Session notes save — re-fetches from server |
| 2018 | init() — from `/api/init` response |
| 2028 | init() — auto-create fallback |
| 3754 | After `/api/clear` in another path |

**activeSid reads:** Used in renderHistory wrapper (sessionStorage key), renderSessions (highlight active), loadSession guard, deleteSession, save, notes, tags, and many other places.

## 3. Observed Behavior Test

### a. Fresh page load (after server restart)

- `/api/init` returns: `active_sid = "2026-04-18_16-24-57"`, history length 8 (the omelette session restored by `_load_today_session()`)
- Server restores most recent session with content on boot — this is by design
- DOM renders 8 messages from that session
- Sidebar shows that session as active
- `graceful_sid` in sessionStorage: set to `2026-04-18_16-24-57` after renderHistory fires

### b. Click "+ new"

- `/api/new` returns: new sid (e.g. `2026-04-18_16-29-18`), empty history
- Server clears `_session_history`, sets `_active_session_ts[0]` to new sid
- Client sets `history=[]`, `activeSid=new_sid`
- DOM cleared, empty state shown
- `graceful_sid` updated to new sid (via renderHistory wrapper)
- Sidebar shows new session highlighted, old session below it
- `/api/init` now returns empty history with new sid — **correct**

### c. Send "test message"

- Client pushes user msg to `history`, streams response, pushes assistant msg
- renderHistory fires during/after stream → saves to `sessionStorage.graceful_history_<new_sid>`
- Message stored under correct key
- `/api/init` returns history with the new messages, correct sid — **correct**

### d. Click old session in sidebar

- `loadSession("2026-04-18_16-24-57")` called
- GET `/api/sessions/2026-04-18_16-24-57` → server loads that session's file, sets `_active_session_ts[0]` = old sid, returns 8 messages
- Client sets `history = old_messages`, `activeSid = old_sid`
- renderHistory clears DOM, renders old messages
- `graceful_sid` updated to old sid
- `/api/init` now returns old session data — **correct**

### e. Click back to new session

- `loadSession("2026-04-18_16-29-18")` called
- GET `/api/sessions/2026-04-18_16-29-18` → **this file may not exist on disk** if the session was never saved (only 1 message sent, and `_save_session()` requires ≥2 entries)
- If 404: client sets `history=[]`, `activeSid=new_sid`, renders empty — **BUG: loses the "test message" that was sent in step c**
- The message exists in `sessionStorage.graceful_history_<new_sid>` but is NOT consulted here — only init() reads from sessionStorage, not loadSession()

### f. Refresh page

- init() fetches `/api/init`
- Server returns whatever `_session_history` and `_active_session_ts[0]` are (depends on which session was last loaded in step d or e)
- If server has old session loaded (from step d): returns old session data → DOM shows old session, even though user was viewing new session
- init() checks sessionStorage only if server returned empty history — but server returned NON-empty old session history, so sessionStorage is not consulted
- **BUG: user sees wrong session after refresh**

## 4. Mismatch Map

### Mismatch 1: `graceful_sid` is written but never read

**What the code does:** renderHistory wrapper writes `sessionStorage.graceful_sid = activeSid` on every render. No code ever reads it.

**What's wrong:** The intent was to use it as a pointer to know which session to restore, but init() uses the server's `active_sid` instead.

**Minimum fix:** Not a bug per se — `graceful_sid` is currently dead code. Either remove it or start reading it in init() as a fallback when the server returns wrong/stale session data (see Mismatch 3).

### Mismatch 2: loadSession() doesn't consult sessionStorage

**What the code does:** loadSession() fetches `/api/sessions/<id>`. If the session file doesn't exist (404), it sets `history=[]` — losing any messages that exist only in sessionStorage (unsaved new sessions).

**What's wrong:** New sessions with <2 messages may not have been saved to disk. Their history only exists in sessionStorage and in-memory `_session_history`. Switching away and back loses them.

**Minimum fix:** In loadSession()'s 404 handler, check `sessionStorage.getItem("graceful_history_"+id)` before falling back to empty:
```js
if(r.status===404){
  var _ss; try{_ss=sessionStorage.getItem("graceful_history_"+id);}catch(e){}
  history=_ss?JSON.parse(_ss):[];
  activeSid=id; renderHistory(); ...
}
```

### Mismatch 3: Server determines active session on boot via `_load_today_session()`, ignoring which session the user was actually using

**What the code does:** On server restart, `_load_today_session()` scans session files newest-first, loads the first one with content. This becomes the active session. Empty new sessions (never saved) are invisible.

**What's wrong:** If the user was in a new empty session, restarts the server (or it crashes), the server restores an old session. The client's `graceful_sid` (in sessionStorage) points to the new session, but the server returns the old one. init() trusts the server, so the user sees old session content.

**Minimum fix:** In init(), after getting server's `active_sid`, compare with `sessionStorage.graceful_sid`. If they differ and sessionStorage has a more recent sid, use sessionStorage's history for that sid (and optionally POST to server to switch). This is the only place `graceful_sid` would be useful.

### Mismatch 4: `/api/save` is only called on `beforeunload` and before session switches — new sessions with only 1 turn may never be saved to disk

**What the code does:** `_save_session()` requires `len(_session_history) >= 2` to actually write a file (line ~5832). A session with 1 user + 1 assistant message = 2 entries, so that should be OK. But a session with 0 messages (clicked "+ new" but never typed) is never saved.

**What's wrong:** This isn't directly wrong — empty sessions shouldn't be saved. But the interaction with loadSession()'s 404 path means switching away from a session with exactly 1 turn (before auto-save fires) and back loses its content.

**Minimum fix:** Same as Mismatch 2 — loadSession's 404 path should check sessionStorage.

### Summary: The Two Real Bugs

1. **loadSession() 404 path ignores sessionStorage** — switching away from a new/unsaved session and back loses messages that only exist in sessionStorage. Fix: check sessionStorage in the 404 handler.

2. **init() trusts server's active_sid without comparing to client-side graceful_sid** — after server restart, wrong session loads. Fix: read `graceful_sid` from sessionStorage in init(), and if it differs from server's `active_sid`, prefer sessionStorage's cached history for that sid (and call loadSession on the server to sync).
