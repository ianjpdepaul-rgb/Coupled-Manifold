# Manifold UI Inventory

Complete catalog of every interactive element in `manifold_data/static/index.html`.

---

## Hotbar / Toolbar (`#topbar`)

### Buttons

| ID | Label / Icon | Description |
|----|-------------|-------------|
| `#sb-tog` | `☰` | Toggle sidebar visibility |
| `#export-btn` | `↑ md` | Export chat as Markdown file |
| `#export-nb-btn` | `↑ ipynb` | Export as Jupyter Notebook |
| `#export-csv-btn` | `↑ csv` | Export stats as CSV spreadsheet |
| `#export-json-btn` | `↑ json` | Export session as JSON |
| `#think-btn` | `Think` | Toggle deep reasoning (think mode) |
| `#kg-btn` | `◉ Map` | Open/close Borgesian Map (knowledge graph) panel |
| `#mem-browser-btn` | `Memory` | Open/close memory browser panel |
| `#tts-btn` | `🔊` | Toggle TTS read-aloud on responses |
| `#sound-btn` | `🔇` / `🔊` | Toggle sound effects (chime on response complete) |
| `#theme-btn` | `⚙ Theme` | Open/close theme editor panel |

### Response Length Buttons (`#resp-len-btns`)

| Class / data-len | Label | Description |
|-------------------|-------|-------------|
| `.rl-btn[data-len="S"]` | `S` | Short response mode |
| `.rl-btn[data-len="M"]` | `M` | Medium response mode (default) |
| `.rl-btn[data-len="L"]` | `L` | Long response mode |

### Non-interactive Display Elements

| ID | Description |
|----|-------------|
| `#model-badge` | Shows model name: "Graceful-I1 . Gemma 4 E4B" |
| `#temp-badge` | Temperature badge (e.g. "τ 0.0"), click does nothing |
| `#trace-pill` | Live trace metrics (turn, temp, mode, tokens, corpus, archive); hover shows tooltip |
| `#status-strip` | Dynamic status pills (generating, think mode, focus, offline, etc.) |
| `#activity-bar` | Thin animated bar; click toggles trace detail panel |

---

## Left Sidebar (`#sidebar`)

### Top Section (`#sb-top`)

| ID | Label / Icon | Description |
|----|-------------|-------------|
| `#logo` | `manifold` | App logo text (non-interactive) |
| `#new-chat-btn` | `+ new` | Create a new chat session |
| `#sb-filter` | Search input | Filters session list by text match |
| `#ss-trigger-btn` | `⌕ all` | Open session search modal (Cmd+Shift+S) |
| `#tag-bar` | Tag chips | Dynamically rendered tag filter chips; click filters sessions by tag |

### Session List (`#session-list`)

| Element | Description |
|---------|-------------|
| `.si` | Session item row; click loads that session |
| `.si-date` | Session name/date label; double-click to rename |
| `.si-ren` | `✎` Rename button (per session) |
| `.si-del` | `✕` Delete button (per session) |
| `.tag-chip` | Clickable tag chips to filter sessions |

### Notes Panel (`#notes-panel`)

| ID | Description |
|----|-------------|
| `#session-notes` | `<textarea>` for per-session notes; auto-saves on input |

### Bottom Section (`#sb-bot`)

| ID | Label | Description |
|----|-------|-------------|
| `#settings-btn` | `⚙ Settings` | Open settings panel |
| `#medulla-btn` | `◈ Trace` | Toggle trace detail bar |
| `#analytics-btn` | `📊 Stats` | Toggle analytics dashboard |
| `#analytics-panel` | (hidden panel) | Shows session/trace stats inline |
| `#export-all-btn` | `↓ Export All` | Download all sessions as one markdown file |

---

## Right Panels

### Settings Panel (`#settings-panel`)

Slide-in panel from right, opened by `#settings-btn`.

#### Buttons

| ID | Label | Description |
|----|-------|-------------|
| `#s-close` | `✕` | Close settings panel |
| `#name-set` | `Set` | Save user name |
| `#name-clear` | `↺ Clear` | Clear user name |
| `#sys-set` | `Set` | Save system prompt |
| `#sys-clear` | `↺ Clear` | Clear system prompt |
| `#index-btn` | `Index` | Index a corpus file/folder |
| `#ft-btn` | `Fine-tune` | Fine-tune LoRA on corpus |
| `#corpus-refresh-btn` | `↻ Refresh` | Reload corpus file list |
| `#learn-toggle-btn` | `On` / `Off` | Toggle online LoRA learning |
| `#mem-btn` | `Refresh` | Refresh memory status |
| `#export-data-btn` | `Export` | Export session data as zip |

#### Inputs

| ID | Type | Description |
|----|------|-------------|
| `#temp-sl` | `range` (0-2, step 0.05) | Temperature slider |
| `#user-name` | `text` | User's display name |
| `#sys-prompt` | `textarea` | System prompt |
| `#corpus-path` | `text` | Path to file/folder for indexing |
| `#ft-steps` | `number` (1-200) | Fine-tuning step count |
| `#think-budget-sl` | `range` (100-1500, step 50) | Think mode token budget |

#### Display

| ID | Description |
|----|-------------|
| `#temp-disp` | Temperature mood text |
| `#think-budget-disp` | Think budget label |
| `#index-st` | Index status text |
| `#ft-st` | Fine-tune status text |
| `#mem-out` | Memory status JSON output |
| `#corpus-file-list` | Listed corpus files with per-file `✕` remove buttons |

### Theme Panel (`#theme-panel`)

Slides in from right, opened by `#theme-btn`.

#### Preset Theme Buttons

| data-preset | Label | Colors |
|-------------|-------|--------|
| `graceful` | Graceful | Pink / Blue / Dark gray |
| `walden` | Walden | Green / Light green / Dark green |
| `tyrian` | Tyrian | Purple / Orange / Deep purple |

#### Controls

| ID | Type | Description |
|----|------|-------------|
| `#t-density` | `<select>` | Density: Compact / Comfortable / Spacious |
| `#t-fontsize` | `range` (12-20) | Font size slider |
| `#t-accent` | `color` | Accent color picker |
| `#t-user-clr` | `color` | User bubble color picker |
| `#t-asst-clr` | `color` | Assistant bubble color picker |
| `#t-bg-clr` | `color` | Background color picker |

#### Accessibility Buttons

| ID | Label | Description |
|----|-------|-------------|
| `#t-hc` | `High contrast` | Toggle high contrast mode |
| `#t-large` | `Large text` | Toggle large text mode |
| `#t-motion` | `Reduced motion` | Toggle reduced motion |
| `#t-reset` | `Reset to Graceful` | Reset all theme settings to defaults |
| `#theme-close` | `✕` | Close theme panel |

### Knowledge Graph Panel (`#kg-panel`)

Floating panel, opened by `#kg-btn` or `/knowledge` command.

| Element | Description |
|---------|-------------|
| `#kg-panel-header` | Header with "Borgesian Map" title |
| Close button (inline onclick) | `✕` closes the panel |
| `.kg-reset-btn` | `⌖ Reset` resets zoom/pan (injected dynamically) |
| `#kg-svg` | SVG canvas with D3 force-directed graph; supports zoom, pan, drag, click-to-select, hover |

### Memory Browser Panel (`#mem-panel`)

Slides in from right, opened by `#mem-browser-btn`.

| Element | Description |
|---------|-------------|
| Refresh button (inline onclick `↻`) | Refresh memory list |
| `#mem-close` | `✕` close panel |
| `.mem-item` | Individual memory entry; click inserts `/recall name` into input |
| `.mem-item` delete button (`✕`) | Delete a named memory |

---

## Input Area (`#input-area`)

### Main Controls

| ID | Label / Icon | Description |
|----|-------------|-------------|
| `#attach-btn` | `📎` | Open file picker to attach files |
| `#msg` | `<textarea>` | Main message input; placeholder "Message manifold..." |
| `#send-btn` | `↑` (arrow) / spinner | Send message or stop generation |
| `#mic-btn` | `🎙` | Voice input via Speech Recognition (created dynamically, only if browser supports it) |

### Supplementary

| ID | Description |
|----|-------------|
| `#file-inp` | Hidden `<input type="file">` triggered by `#attach-btn`; accepts .txt, .md, .pdf, .docx, .pptx, .xlsx, .csv, .py, .html, .json, images, audio, video |
| `#char-ct` | Character count display (shown when >40 chars) |
| `#token-counter` | Approximate token count (shown when >10 chars) |
| `#pins-bar` | Pinned context chips with `×` unpin buttons |
| `#url-offer` | URL paste detection banner with "Add to context", "Paste as text", and dismiss buttons |

### Empty State (`#empty`)

| Element | Description |
|---------|-------------|
| `#empty-title` | "manifold" title |
| `#resume-last-btn` | `↑ resume last chat` button (shown when previous sessions exist) |
| `.sug` suggestions | Four clickable suggestion chips: `/who`, `/stats`, `/recap`, `/search` |
| `.welcome-greeting` | "Ready. / for commands, ⌘K for the palette." (injected dynamically) |

### In-Chat Controls

| Element | Description |
|---------|-------------|
| `#scroll-btn` | `↓` Scroll to bottom (appears when scrolled up) |
| `#load-older-btn` | `↑ load older` Load older messages (appears when history has more) |

---

## Per-Message Action Buttons (built by `buildRow()`)

These appear on hover over each message row.

### User Messages

| Button | Label | Description |
|--------|-------|-------------|
| `.ma-btn` | `copy` | Copy message text |
| `.ma-btn` | `fork` | Fork session from this point |
| `.ma-btn` | `✎` (pencil) | Edit and regenerate |

### Assistant Messages

| Button | Label | Description |
|--------|-------|-------------|
| `.ma-btn` | `copy` | Copy rendered text |
| `.ma-btn` | `copy md` | Copy raw markdown source |
| `.ma-btn` | `fork` | Fork session from this point |
| `.ma-btn` | `regen` | Regenerate this response |
| `.ma-btn` | `🔊` | Read aloud via TTS API |
| `.ma-btn` | `📌` | Pin to context |
| `.ma-btn` | `▲` | Upvote (reinforcement) |
| `.ma-btn` | `▽` | Downvote (penalize + auto-regen) |
| `.ma-btn` | `continue` | Resume a truncated response (hidden until last assistant msg) |
| `.expand-btn` | `▼ show more` / `▲ show less` | Collapse/expand long responses (>35 lines) |

### Reaction Buttons (added by `finalizeStream`)

| Button | Label | Description |
|--------|-------|-------------|
| `.react-btn.react-up` | `👍` | Good response |
| `.react-btn.react-down` | `👎` | Poor response |

### Code Block Buttons

| Class | Label | Description |
|-------|-------|-------------|
| `.cbtn.run` | `▶ run` | Execute Python code block |
| `.cbtn.copy-code` | `copy` | Copy code block content |
| `.copy-code-btn` | `copy` | Alternate copy button injected on `<pre>` blocks |
| `.tbl-dl-btn` | `↓ csv` | Download markdown table as CSV |
| `.df-dl-btn` | `↓ csv` | Download DataFrame table as CSV |

### Follow-up Chips (shown on high-drift responses)

| Class | Description |
|-------|-------------|
| `.fc-chip` | Clickable follow-up question chip; sends the question |
| `.fc-dismiss` | `×` dismiss all follow-up chips |

---

## Modal Overlays

### Command Palette (`#cmd-ov` / `#cmd-pal`)

Opened by `Cmd+K`. Full-screen overlay with search.

| ID | Description |
|----|-------------|
| `#cmd-s` | Search input for filtering commands |
| `#cmd-esc` | `esc` badge |
| `#cmd-list` | Filterable list of all commands (from `CMDS` array) |
| `.ci` items | Clickable command items |

### Slash Popup (`#slash-pop`)

Appears above input when typing `/`. Uses the `CMDS` array.

| Element | Description |
|---------|-------------|
| `.sp-item` | Clickable slash command items |
| Footer | Keyboard hints: ↑↓ navigate, Tab select, Esc close |

### Cmd Dropdown (`#cmd-dropdown`)

Inline autocomplete above input when typing `/`. Uses the `CMD_DROPDOWN_LIST` array.

| Element | Description |
|---------|-------------|
| `.cd-item` | Clickable autocomplete command items |

### Chat Search Bar (`#chat-search`)

Opened by `Cmd+F`. Inline bar below topbar.

| ID | Description |
|----|-------------|
| `#chat-search-inp` | Search input for in-chat text search |
| `#chat-search-info` | Match count display (e.g. "3/7") |
| `#chat-search-close` | `✕` close search bar |

### Shortcuts Modal (`#shortcuts-modal`)

Opened by `Cmd+/`. Lists all keyboard shortcuts.

| ID | Description |
|----|-------------|
| `#shortcuts-box` | Modal content with shortcut rows |
| `#shortcuts-close-btn` | `Close` button |

### Session Search Modal (`#session-search-modal`)

Opened by `Cmd+Shift+S` or `#ss-trigger-btn`.

| ID | Description |
|----|-------------|
| `#ssm-input` | Search input for searching across all sessions |
| `#ssm-results` | Clickable result list; loads matching session on click |
| `#ssm-backdrop` | Click to close |

### Context Menu (`#ctx-menu`)

Right-click on any message row.

| Item | Description |
|------|-------------|
| Copy message text | Copies bubble inner text |
| Pin to session | Pins message text to context |
| Export as .txt | Downloads message as text file |
| Re-generate | Regenerate response (assistant messages only) |
| Copy as HTML | Copies styled HTML of the message |
| Fork from here | Forks conversation at this message |

### Selection Mini-Menu (`#sel-menu`)

Appears when text is selected inside a message bubble.

| ID | Label | Description |
|----|-------|-------------|
| `#sel-explain` | `Explain` | Sends "Explain: [selected text]" |
| `#sel-note` | `Save note` | Appends selection to session notes |
| `#sel-search` | `Search corpus` | Sends "/find [selected text]" |

### Other Overlays

| ID | Description |
|----|-------------|
| `#settings-wrap` + `#settings-backdrop` | Settings panel wrapper with backdrop click-to-close |
| `#drag-ov` | "Drop to upload" overlay during file drag |
| `#toasts` | Toast notification container (bottom right) |
| `#reconnect-toast` | "Connection lost. Reconnecting..." banner |
| `#offline-banner` | "You are offline" banner |
| `#timer-bar` | Countdown timer bar (top of screen) |
| `#focus-exit-btn` | `✕ Focus` exit focus mode button (top right, visible only in focus mode) |
| `#mob-backdrop` | Mobile sidebar backdrop |
| `#pin-strip` | Pinned messages strip above message area |

---

## Dropdowns / Select Elements

| ID | Options | Location |
|----|---------|----------|
| `#t-density` | Compact, Comfortable (default), Spacious | Theme panel |

---

## Slash-Command Arrays

### `CMDS` (44 entries) -- used by slash popup and command palette

| Command | Description | Group |
|---------|-------------|-------|
| `/who` | Identity model -- who I am | Memory |
| `/iam ` | Append fact to identity model | Memory |
| `/forget ` | Remove a fact from identity | Memory |
| `/remember ` | Store a persistent note | Memory |
| `/recall ` | Retrieve a stored note | Memory |
| `/knowledge` | Borgesian Map -- knowledge graph summary | Memory |
| `/export` | Export session as markdown | Session |
| `/save ` | Snapshot session with label | Session |
| `/load` | Load / list session snapshots | Session |
| `/clear` | Clear current session | Session |
| `/summarize` | Summarize this conversation | Session |
| `/pin ` | Pin a message to context | Session |
| `/backup` | Backup all manifold data | Session |
| `/search ` | Force web search | Search |
| `/find ` | Semantic corpus search | Corpus |
| `/run ` | Execute Python snippet | Code |
| `/plot ` | Plot math instantly | Code |
| `/calc ` | Symbolic math | Code |
| `/trace` | Cross-session trace analytics | Analysis |
| `/spectrum` | LoRA adapter spectral analysis | Analysis |
| `/swot ` | SWOT analysis of a topic | Analysis |
| `/risk ` | Risk analysis of a topic | Analysis |
| `/hypothesis ` | Generate and stress-test a hypothesis | Analysis |
| `/experiment` | Run adversarial self-test | Research |
| `/debate ` | Steel-man debate on a topic | Research |
| `/devil ` | Devil's advocate on a claim | Research |
| `/peer` | Peer-review mode for text | Research |
| `/data` | Export research data bundle | Research |
| `/scaffold ` | Structural scaffold for a topic | Writing |
| `/elaborate` | Elaborate on the last response | Writing |
| `/rhetorical` | Rhetorical analysis of text | Writing |
| `/evolve` | Evolve / sharpen the last argument | Writing |
| `/thread` | Thread-form breakdown of topic | Writing |
| `/translate ` | Translate to another language | Writing |
| `/eli5 ` | Explain like I'm 5 | Teaching |
| `/teacher ` | Socratic teacher mode on topic | Teaching |
| `/brainstorm ` | Freeform brainstorm | Teaching |
| `/antagonist` | Antagonist mode -- challenge next response | Modes |
| `/socratic` | Socratic mode toggle | Modes |
| `/compress` | Compression mode toggle | Modes |
| `/dream` | Dream mode -- lateral generative response | Modes |
| `/reading` | Personalized reading list | Modes |
| `/finetune` | Fine-tune LoRA on corpus | Training |
| `/adapter ` | Save/load/list LoRA adapter profiles | Training |
| `/learn ` | Toggle online LoRA updates (on/off) | Training |
| `/visionquality ` | Vision token budget (low/medium/high/ultra) | Vision |
| `/timer ` | Set a countdown timer | Utility |
| `/stats` | System status | Utility |
| `/?` | All commands | Utility |

### `CMD_DROPDOWN_LIST` (21 entries) -- used by inline input autocomplete

| Command | Description |
|---------|-------------|
| `/who` | What the model knows about you |
| `/recap` | Summarize recent conversation |
| `/find` | Semantic corpus search |
| `/iam` | Append fact to identity |
| `/search` | Force web search |
| `/save` | Snapshot session with label |
| `/load` | Load / list snapshots |
| `/export` | Export session as markdown |
| `/run` | Execute Python code |
| `/plot` | Plot math |
| `/calc` | Symbolic math |
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

## Keyboard Shortcuts

### Wired in JS (functional)

| Shortcut | Handler | Description |
|----------|---------|-------------|
| `Enter` | `#msg` keydown | Send message (or stop generation if generating) |
| `Shift+Enter` | `#msg` keydown | Newline in input (not intercepted) |
| `Cmd/Ctrl+K` | global keydown | Toggle command palette |
| `Escape` | global keydown | Stop generation; close command palette; close shortcuts modal; close context menu; close chat search; close session search; exit focus mode |
| `Cmd/Ctrl+F` | global keydown (capture) | Open/close in-chat search |
| `Cmd/Ctrl+/` | global keydown | Toggle keyboard shortcuts modal |
| `Cmd/Ctrl+N` | global keydown | New chat session (when focus is in app) |
| `Cmd/Ctrl+Shift+F` | global keydown | Toggle focus mode |
| `Cmd/Ctrl+Shift+V` | global keydown | Toggle voice input (click mic button) |
| `Cmd/Ctrl+Shift+S` | global keydown | Open session search modal |
| `ArrowUp` | `#msg` keydown | Input history recall (previous); also slash popup / cmd dropdown navigation |
| `ArrowDown` | `#msg` keydown | Input history recall (next); also slash popup / cmd dropdown navigation |
| `Tab` | `#msg` keydown | Select highlighted slash command / dropdown item |
| `Cmd/Ctrl+Enter` | edit textarea keydown | Save edited message and regenerate |
| `Enter` | `#chat-search-inp` keydown | Next search match (Shift+Enter for previous) |
| `ArrowUp/ArrowDown` | command palette open | Navigate palette items |
| `Enter` | command palette open | Select highlighted palette command |

### Documented but NOT wired in JS

| Shortcut | Listed Description | Note |
|----------|--------------------|------|
| `Cmd+S` | Save snapshot (/save) | Appears in shortcuts modal HTML but no keydown handler found |
| `Cmd+Enter` | Send message | Listed in shortcuts modal; actual send is plain `Enter` |

---

## Other Interactive Behaviors

| Feature | Description |
|---------|-------------|
| Drag and drop | File drop anywhere triggers upload (drag overlay `#drag-ov`) |
| Image paste | Pasting an image from clipboard uploads it |
| URL paste | Pasting a URL shows the `#url-offer` banner |
| Mobile swipe | Swipe right from left edge opens sidebar; swipe left closes it |
| Haptic feedback | `navigator.vibrate(30)` on send button click (mobile) |
| Double-click session name | Opens rename prompt |
| Right-click message | Opens context menu |
| Text selection in bubble | Shows selection mini-menu (Explain / Save note / Search corpus) |
| `beforeunload` | Auto-saves session via beacon |
| Offline/online events | Shows/hides offline banner |
| Browser notifications | Requests permission on first send; notifies when response ready (tab hidden) |
| Service worker | Registered from `/static/sw.js` on load |
| Draft persistence | Input text saved to `localStorage` (`manifold_draft`); restored on reload |
| Session auto-save | Saves current session before switching to another |
| Auto-title | Sessions auto-titled from first user message after first reply |
| Hashtag extraction | `#tags` in messages are extracted and attached to the session |
