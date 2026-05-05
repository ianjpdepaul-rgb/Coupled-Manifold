# Feature Inventory

Generated: 2026-04-17 (line numbers updated through 2026-05-05 fix pass)

> **Note:** Line numbers are approximate — the fix pass (2026-05-05, 20 commits)
> shifted endpoints by ~30–50 lines due to added locks, logging, dead code removal,
> and thread safety improvements. Endpoint paths and command names are unchanged.

---

## Section 1: Backend Endpoints

Every `@api.get`, `@api.post`, and `@api.delete` route in `app.py`.

| Line | Method | Path | Description |
|------|--------|------|-------------|
| 5875 | GET | `/` | Serve the main SPA HTML page |
| 5879 | GET | `/api/health` | Health check returning status and model name |
| 5884 | GET | `/api/init` | Return initial state (active session ID, settings, metadata) |
| 5907 | POST | `/api/chat` | Main chat endpoint — send a user message, stream assistant response |
| 5927 | POST | `/api/regen` | Regenerate the last assistant response |
| 5943 | POST | `/api/stop` | Stop the current streaming generation |
| 5948 | POST | `/api/save` | Lightweight session save (called by beforeunload) |
| 5954 | POST | `/api/new` | Save current session and start a new one |
| 5978 | POST | `/api/reset` | Full state reset (clear + new session, for test seed independence) |
| 6007 | POST | `/api/fork` | Fork conversation at a point into a new session |
| 6028 | GET | `/api/sessions` | List all saved sessions |
| 6032 | GET | `/api/sessions/{session_id}` | Load a specific session by ID |
| 6054 | GET | `/api/history_page` | Return a page of messages from the active session (pagination) |
| 6076 | POST | `/api/sessions/clear_all` | Delete all session files and reset in-memory state |
| 6097 | DELETE | `/api/sessions/{session_id}` | Delete a specific session by ID |
| 6112 | POST | `/api/edit` | Edit a past user message and regenerate from that point |
| 6136 | POST | `/api/sessions/{session_id}/rename` | Rename a session |
| 6163 | GET | `/api/sessions/{session_id}/notes` | Get notes for a session |
| 6173 | POST | `/api/sessions/{session_id}/notes` | Set notes for a session |
| 6184 | GET | `/api/corpus_files` | List all sources indexed in the corpus with chunk counts |
| 6202 | POST | `/api/corpus_files/remove` | Remove all chunks from a specific corpus source |
| 6280 | POST | `/api/upload` | Upload a file (PDF, CSV, image, etc.) for corpus indexing or vision |
| 6419 | POST | `/api/export_notebook` | Export current session as a Jupyter notebook (.ipynb) |
| 6482 | GET | `/api/export_csv` | Export session_log as an Excel-friendly CSV |
| 6545 | POST | `/api/export_df` | Download a DataFrame from the persistent code namespace as CSV |
| 6569 | GET | `/api/export_json` | Export session history + stats as JSON |
| 6590 | POST | `/api/reset_namespace` | Clear the persistent Python code namespace |
| 6596 | POST | `/api/settings` | Update settings (temp, think_mode, think_budget, user_name, system_prompt, etc.) |
| 6636 | POST | `/api/model_mode` | Set the model routing mode (e.g. big, small, auto) |
| 6651 | GET | `/api/model_mode` | Get the current model routing mode |
| 6659 | POST | `/api/index` | Index a local file or directory into the corpus |
| 6683 | POST | `/api/finetune` | Fine-tune LoRA adapter on corpus chunks |
| 6766 | GET | `/api/mem_status` | Return memory subsystem status (corpus, archive, identity) |
| 6772 | POST | `/api/reinforce` | Positive/negative reinforcement signal for the last response |
| 6813 | GET | `/api/trace` | Cross-session Hessian trace analytics as JSON |
| 6832 | GET | `/api/spectrum` | LoRA adapter SVD spectral analysis as JSON |
| 6857 | POST | `/api/speak` | Text-to-speech via macOS `say` command |
| 6869 | POST | `/api/pin` | Pin text to the conversation context |
| 6881 | POST | `/api/unpin` | Unpin a specific pinned context item by index |
| 6890 | GET | `/api/pins` | List all pinned context items |
| 6897 | POST | `/api/react` | Log a thumbs-up/down reaction on a response |
| 6948 | GET | `/api/export_all` | Export all sessions as a single markdown file |
| 6986 | GET | `/api/export` | Export traces, logs, ctrl state, and session data as a zip |
| 7046 | POST | `/api/experiment` | Run the adversarial self-test experiment |
| 7059 | POST | `/api/adapter/save` | Save the current LoRA adapter as a named checkpoint |
| 7081 | POST | `/api/adapter/load` | Load a named LoRA adapter checkpoint |
| 7105 | GET | `/api/adapter/list` | List all saved adapter checkpoints |
| 7120 | POST | `/api/fetch_url` | Fetch and extract text content from a URL |
| 7168 | POST | `/api/sessions/{session_id}/tags` | Set tags on a session |
| 7195 | GET | `/api/search_sessions` | Full-text search across all session JSON files |
| 7258 | GET | `/api/backup` | Trigger backup creation and return metadata |
| 7269 | GET | `/api/backup/download` | Download the latest backup zip file |
| 7294 | GET | `/api/analytics` | Aggregate stats across all sessions (turns, words, dates, traces) |
| 7371 | GET | `/api/knowledge_graph` | Knowledge graph from identity concepts + session co-occurrence |
| 7417 | GET | `/api/named_memory` | Return all named memory slots as JSON |
| 7422 | DELETE | `/api/named_memory/{key}` | Delete a single named memory slot |
| 7434 | GET | `/api/stats_json` | Structured stats JSON for the analytics panel |
| 7553 | POST | `/api/_test/trace_inject` | (Test-only, requires GRACEFUL_TEST_MODE=1) Inject synthetic trace values |

---

## Section 2: Slash Commands with Handlers in app.py

Every slash command that has an explicit `if user_msg...` handler block inside the chat function.

| Line | Command | Description |
|------|---------|-------------|
| 2533 | `/help` (alias `/?`) | Display the full command reference table |
| 2613 | `/continue` (alias `/cont`) | Resume a response that was cut short |
| 2519 | `/who` | Show what the model knows about the user (identity model) |
| 2642 | `/stats` | System and memory status dashboard |
| 2743 | `/version` | App version, Python/MLX versions, and system info |
| 2764 | `/iam <statement>` | Add a fact directly to the identity model |
| 2789 | `/forget <topic>` | Remove identity notes mentioning a topic |
| 2810 | `/recap` | Show the last 2-3 session summaries |
| 2828 | `/summarize` | Summarize the current session in plain language |
| 2863 | `/check` | Verify confidence and hedging in the last response |
| 2905 | `/rephrase` | Request a different angle on the last response |
| 2929 | `/mood` | Trace-based tone and energy analysis of the session |
| 2967 | `/find <query>` | Search corpus and return top chunks with scores |
| 2986 | `/export` | Dump session as a markdown file |
| 3007 | `/clear` | Reset session history (keeps identity and corpus) |
| 3024 | `/pin <text>` | Pin a message to persistent context |
| 3046 | `/save [label]` | Snapshot the current conversation as a branch |
| 3076 | `/load [label]` | Restore a snapshot, or list available ones |
| 3132 | `/finetune [steps]` | Fine-tune LoRA adapter on corpus from chat |
| 3217 | `/run <code>` | Execute Python in a safe sandbox |
| 3257 | `/plot <expr>` | Instant math expression plotter |
| 3284 | `/calc <expr>` (alias `/math`) | Instant symbolic + numeric math evaluator |
| 3312 | `/analyze [df]` | Full auto-profile of an uploaded dataframe |
| 3449 | `/trace` | Cross-session Hessian trace analytics |
| 3465 | `/spectrum` | LoRA adapter SVD spectral analysis |
| 3483 | `/experiment` | Run 10-turn adversarial self-test |
| 3514 | `/learn [on\|off\|status]` | Toggle online LoRA learning |
| 3535 | `/trace-mode [async\|sync]` | Toggle trace computation mode (sync adds 2-5s) |
| 3554 | `/adapter [save\|load\|list] <name>` | Manage LoRA adapter profiles |
| 3624 | `/visionquality [level]` | Tune Gemma 4 vision token budget (low/med/high/ultra/max) |
| 3656 | `/antagonist` | Arm single-turn antagonist (opposing argument) mode |
| 3666 | `/socratic` | Toggle Socratic question-only mode |
| 3679 | `/compress` | Toggle one-sentence compression mode |
| 3692 | `/backup` | Zip manifold_data/ (excluding checkpoints/logs) for download |
| 3719 | `/scaffold <topic>` | Structured layered framework generation |
| 3736 | `/dream` | Free ideation mode (model generates freely) |
| 3745 | `/reading` | Personalized reading list based on user interests |
| 3755 | `/knowledge` | Static text summary of the knowledge graph |
| 3777 | `/data` | Export research data bundle as zip |
| 3825 | `/remember <name> = <content>` | Store a persistent named memory slot |
| 3846 | `/recall <name>` | Retrieve a named memory slot |
| 3869 | `/timer <duration>` | Set a countdown timer (e.g. 25m, 1h, 30s) |
| 3894 | `/search <query>` | Full-text search across all session history |
| 3969 | `/week` | Weekly digest of sessions from the last 7 days |
| 4010 | `/brief` | Yesterday's key threads and takeaways |
| 4058 | `/debate <topic>` | Present both sides of a topic with equal force |
| 4068 | `/eli5 <topic>` | Explain like I'm 5 (concrete analogy, no jargon) |
| 4077 | `/teacher <question>` | Socratic teacher mode (questions only) |
| 4086 | `/brainstorm <topic>` | Rapid ideation: 10-15 ideas, no critique |
| 4095 | `/devil <claim>` | Devil's advocate: find every flaw in a claim |
| 4104 | `/peer [text]` | Peer review of last response or provided text |
| 4118 | `/hypothesis <statement>` | Hypothesis test: evidence, falsifiers, assumptions |
| 4127 | `/quiz` | Generate 5 quiz questions from session topics |
| 4140 | `/swot <topic>` | SWOT analysis (Strengths/Weaknesses/Opportunities/Threats) |
| 4149 | `/glossary` | Extract and define terms from the last response |
| 4165 | `/counterpoint` | Strongest counterarguments to the last response |
| 4181 | `/risk <plan>` | Risk assessment table (likelihood, impact, mitigation) |
| 4190 | `/flashcards` | Generate 8-10 flashcards from session concepts |
| 4202 | `/translate <lang>` | Translate the last response to another language |
| 4219 | `/contradict` | Find archive statements contradicting the last user message |
| 4237 | `/elaborate [N]` | Expand on point N from the last response |
| 4253 | `/abstract` | Write an academic abstract for the current session |
| 4267 | `/rhetorical [text]` | Rhetorical structure analysis (logos/ethos/pathos) |
| 4293 | `/evolve <concept>` | Trace how a concept has evolved across sessions |
| 4322 | `/thread [topic]` | Resume the last discussion of a topic |
| 4355 | `/zettelkasten` | Export session as Zettelkasten atomic notes (saves to file) |

---

## Section 3: Slash Commands in the Help Dropdown (UI)

### Command Palette (Cmd+K) — `CMDS` array (index.html line 1056)

This is the comprehensive command list shown when the user presses Cmd+K.

| Command | Description (as shown in UI) | Group |
|---------|------------------------------|-------|
| `/who` | Identity model — who I am | Memory |
| `/iam` | Append fact to identity model | Memory |
| `/forget` | Remove a fact from identity | Memory |
| `/remember` | Store a persistent note | Memory |
| `/recall` | Retrieve a stored note | Memory |
| `/knowledge` | Borgesian Map — knowledge graph summary | Memory |
| `/export` | Export session as markdown | Session |
| `/save` | Snapshot session with label | Session |
| `/load` | Load / list session snapshots | Session |
| `/clear` | Clear current session | Session |
| `/summarize` | Summarize this conversation | Session |
| `/pin` | Pin a message to context | Session |
| `/backup` | Backup all manifold data | Session |
| `/search` | Force web search | Search |
| `/find` | Semantic corpus search | Corpus |
| `/run` | Execute Python snippet | Code |
| `/plot` | Plot math instantly — /plot sin(x), cos(x) | Code |
| `/calc` | Symbolic math — /calc integrate(x**2, x) | Code |
| `/trace` | Cross-session trace analytics | Analysis |
| `/spectrum` | LoRA adapter spectral analysis | Analysis |
| `/swot` | SWOT analysis of a topic | Analysis |
| `/risk` | Risk analysis of a topic | Analysis |
| `/hypothesis` | Generate and stress-test a hypothesis | Analysis |
| `/experiment` | Run adversarial self-test | Research |
| `/debate` | Steel-man debate on a topic | Research |
| `/devil` | Devil's advocate on a claim | Research |
| `/peer` | Peer-review mode for text | Research |
| `/data` | Export research data bundle | Research |
| `/scaffold` | Structural scaffold for a topic | Writing |
| `/elaborate` | Elaborate on the last response | Writing |
| `/rhetorical` | Rhetorical analysis of text | Writing |
| `/evolve` | Evolve / sharpen the last argument | Writing |
| `/thread` | Thread-form breakdown of topic | Writing |
| `/translate` | Translate to another language | Writing |
| `/eli5` | Explain like I'm 5 | Teaching |
| `/teacher` | Socratic teacher mode on topic | Teaching |
| `/brainstorm` | Freeform brainstorm | Teaching |
| `/antagonist` | Antagonist mode — challenge next response | Modes |
| `/socratic` | Socratic mode toggle | Modes |
| `/compress` | Compression mode toggle | Modes |
| `/dream` | Dream mode — lateral generative response | Modes |
| `/reading` | Personalized reading list | Modes |
| `/finetune` | Fine-tune LoRA on corpus | Training |
| `/adapter` | Save/load/list LoRA adapter profiles | Training |
| `/learn` | Toggle online LoRA updates (on/off) | Training |
| `/visionquality` | Vision token budget (low/medium/high/ultra) | Vision |
| `/timer` | Set a countdown timer | Utility |
| `/stats` | System status | Utility |
| `/?` | All commands | Utility |

### Autocomplete Dropdown (typing `/` in input) — `CMD_DROPDOWN_LIST` (index.html line 2658)

A subset of 21 commands shown as quick autocomplete when the user types `/`.

| Command | Description (as shown in UI) |
|---------|------------------------------|
| `/who` | What the model knows about you |
| `/recap` | Summarize recent conversation |
| `/find` | Semantic corpus search |
| `/iam` | Append fact to identity |
| `/search` | Force web search |
| `/save` | Snapshot session with label |
| `/load` | Load / list snapshots |
| `/export` | Export session as markdown |
| `/run` | Execute Python code |
| `/plot` | Plot math — /plot sin(x), cos(x) |
| `/calc` | Symbolic math — /calc integrate(x**2, x) |
| `/trace` | Cross-session trace analytics |
| `/spectrum` | LoRA adapter spectral analysis |
| `/stats` | System status |
| `/help` | Show all commands |
| `/clear` | Clear current chat |
| `/pin` | Pin text to context |
| `/version` | Show model/server version |
| `/summarize` | Summarize current conversation |
| `/check` | Health check |
| `/rephrase` | Rephrase last response |
| `/finetune` | Fine-tune on corpus |

---

## Section 4: Cross-Reference

### 4a. Hidden Commands

Commands with handlers in app.py but **not** listed in the Cmd+K command palette (`CMDS` array).

| Command | Handler Line | Notes |
|---------|-------------|-------|
| `/help` | 2533 | Has alias `/?` which IS in CMDS; `/help` itself is not a separate CMDS entry |
| `/continue` | 2613 | Also accepts alias `/cont` |
| `/version` | 2743 | Only in autocomplete dropdown, not in Cmd+K palette |
| `/recap` | 2810 | Only in autocomplete dropdown, not in Cmd+K palette |
| `/check` | 2863 | Only in autocomplete dropdown, not in Cmd+K palette |
| `/rephrase` | 2905 | Only in autocomplete dropdown, not in Cmd+K palette |
| `/mood` | 2929 | Not in any UI list |
| `/trace-mode` | 3535 | Not in any UI list; also not in `_known_cmds` guard |
| `/math` | 3284 | Alias for `/calc`; not listed separately |
| `/analyze` | 3312 | Not in any UI list |
| `/quiz` | 4127 | Not in any UI list |
| `/glossary` | 4149 | Not in any UI list |
| `/counterpoint` | 4165 | Not in any UI list |
| `/flashcards` | 4190 | Not in any UI list |
| `/contradict` | 4219 | Not in any UI list |
| `/abstract` | 4253 | Not in any UI list |
| `/zettelkasten` | 4355 | Not in any UI list |
| `/week` | 3969 | Not in any UI list |
| `/brief` | 4010 | Not in any UI list |

### 4b. Dead Entries

Commands listed in the Cmd+K palette but **without** a handler in app.py.

| Command | Palette Entry | Notes |
|---------|---------------|-------|
| (none) | — | All CMDS palette entries have corresponding handlers |

### 4c. Frontend Integration

For each backend endpoint, whether it is called from `index.html` or is backend-only.

| Endpoint | Method | Called from index.html? | Notes |
|----------|--------|------------------------|-------|
| `/` | GET | N/A | Serves the HTML page itself |
| `/api/health` | GET | No | Backend-only |
| `/api/init` | GET | Yes | Startup initialization (lines 2004, 1970, 2336) |
| `/api/chat` | POST | Yes | Main chat send (line 1697) |
| `/api/regen` | POST | Yes | Regenerate response (line 1753) |
| `/api/stop` | POST | Yes | Stop generation (lines 1552, 1780, 1923, 2400) |
| `/api/save` | POST | Yes | beforeunload beacon (line 2296), session switch (line 1885) |
| `/api/new` | POST | Yes | New session (lines 1901, 1913, 1927, 2012) |
| `/api/reset` | POST | No | Backend-only (test harness) |
| `/api/fork` | POST | Yes | Fork conversation (lines 1268, 3723) |
| `/api/sessions` | GET | Yes | List sessions (line 1907) |
| `/api/sessions/{session_id}` | GET | Yes | Load session (line 1887) |
| `/api/sessions/{session_id}` | DELETE | Yes | Delete session (line 1898) |
| `/api/history_page` | GET | Yes | Paginated history loading (lines 1376, 1393) |
| `/api/sessions/clear_all` | POST | No | Backend-only |
| `/api/edit` | POST | Yes | Edit and regenerate (line 1645) |
| `/api/sessions/{id}/rename` | POST | Yes | Rename session (lines 1864, 1875, 3614) |
| `/api/sessions/{id}/notes` | GET | Yes | Load session notes (line 1957) |
| `/api/sessions/{id}/notes` | POST | Yes | Save session notes (line 1973) |
| `/api/corpus_files` | GET | Yes | Corpus file list in settings panel (line 2349) |
| `/api/corpus_files/remove` | POST | Yes | Remove corpus source (line 2361) |
| `/api/upload` | POST | Yes | File upload (lines 2129, 2216, 2234) |
| `/api/export_notebook` | POST | Yes | Notebook export button (line 2316) |
| `/api/export_csv` | GET | Yes | CSV export button (line 2408) |
| `/api/export_df` | POST | Yes | DataFrame CSV download (line 1194) |
| `/api/export_json` | GET | Yes | JSON export button (line 2424) |
| `/api/reset_namespace` | POST | No | Backend-only |
| `/api/settings` | POST | Yes | Settings updates (lines 2146, 2151, 2164, 2171, 2175, 2179, 2183, 2339) |
| `/api/model_mode` | POST | No | Backend-only |
| `/api/model_mode` | GET | No | Backend-only |
| `/api/index` | POST | Yes | Index local path (line 2188) |
| `/api/finetune` | POST | Yes | Fine-tune button in settings (line 2198) |
| `/api/mem_status` | GET | Yes | Memory status in settings panel (line 2208) |
| `/api/reinforce` | POST | Yes | Thumbs up/down buttons (lines 1316, 1322) |
| `/api/trace` | GET | No | Backend-only (accessed via `/trace` slash command) |
| `/api/spectrum` | GET | No | Backend-only (accessed via `/spectrum` slash command) |
| `/api/speak` | POST | Yes | Text-to-speech button on messages (line 1293) |
| `/api/pin` | POST | Yes | Pin from context menu and chat (lines 1302, 2977) |
| `/api/unpin` | POST | Yes | Unpin button in pins panel (lines 1993, 2059) |
| `/api/pins` | GET | Yes | Load pinned items (lines 1984, 2048) |
| `/api/react` | POST | Yes | Reaction rating on messages (line 1515) |
| `/api/export_all` | GET | Yes | Export all sessions button (line 2072) |
| `/api/export` | GET | Yes | Export data button (line 718) |
| `/api/experiment` | POST | No | Backend-only (accessed via `/experiment` slash command) |
| `/api/adapter/save` | POST | No | Backend-only (accessed via `/adapter` slash command) |
| `/api/adapter/load` | POST | No | Backend-only (accessed via `/adapter` slash command) |
| `/api/adapter/list` | GET | No | Backend-only (accessed via `/adapter` slash command) |
| `/api/fetch_url` | POST | Yes | URL content extraction (line 3371) |
| `/api/sessions/{id}/tags` | POST | Yes | Tag management (line 3600) |
| `/api/search_sessions` | GET | Yes | Session search (line 4120) |
| `/api/backup` | GET | No | Backend-only (accessed via `/backup` slash command) |
| `/api/backup/download` | GET | Yes | Download link injected in chat (line 4325) |
| `/api/analytics` | GET | Yes | Analytics dashboard (line 4155) |
| `/api/knowledge_graph` | GET | Yes | Knowledge graph visualization (line 3835) |
| `/api/named_memory` | GET | Yes | Memory panel (line 4068) |
| `/api/named_memory/{key}` | DELETE | Yes | Delete named memory (line 4082) |
| `/api/stats_json` | GET | No | Backend-only |
| `/api/_test/trace_inject` | POST | No | Test-only (conditional on GRACEFUL_TEST_MODE=1) |
