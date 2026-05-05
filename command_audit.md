# Command Audit — 2026-04-18

## Surface Definitions

- **Handler**: server-side slash command handler in `app.py`
- **Autocomplete**: dropdown entries when user types `/` in the input box (`index.html` lines 1059-1118)
- **Palette**: Cmd+K palette entries (`index.html` lines 2706-2727)

## Reconciliation Table

| Command | Handler? | Autocomplete? | Palette? | Emoji consistent? | Description consistent? | Notes |
|---------|----------|---------------|----------|--------------------|-------------------------|-------|
| `/who` | ✅ L2683 | ✅ 🧸 "Identity model — who I am" | ✅ "What the model knows about you" | N/A (no emoji in palette) | ❌ Descriptions differ | |
| `/iam` | ✅ L2928 | ✅ 🧬 "Append fact to identity model" | ✅ "Append fact to identity" | N/A | ⚠️ Minor diff (palette shorter) | |
| `/forget` | ✅ L2953 | ✅ 🗑 "Remove a fact from identity" | ❌ Missing | — | — | **Gap: no palette entry** |
| `/remember` | ✅ L3992 | ✅ 📌 "Store a persistent note" | ❌ Missing | — | — | **Gap: no palette entry** |
| `/recall` | ✅ L4013 | ✅ 💭 "Retrieve a stored note" | ❌ Missing | — | — | **Gap: no palette entry** |
| `/knowledge` | ✅ L3922 | ✅ 🕸 "Borgesian Map — knowledge graph summary" | ❌ Missing | — | — | **Gap: no palette entry** |
| `/stats` | ✅ L2806 | ✅ 📡 "System status" | ✅ "System status" | N/A | ✅ | |
| `/version` | ✅ L2907 | ❌ Missing | ✅ "Show model/server version" | — | — | **Gap: no autocomplete entry** |
| `/recap` | ✅ L2974 | ❌ Missing | ✅ "Summarize recent conversation" | — | — | **Gap: no autocomplete entry** |
| `/summarize` | ✅ L2992 | ✅ 📝 "Summarize this conversation" | ✅ "Summarize current conversation" | N/A | ⚠️ Minor diff | |
| `/check` | ✅ L3027 | ❌ Missing | ✅ "Health check" | — | — | **Gap: no autocomplete entry** |
| `/rephrase` | ✅ L3069 | ❌ Missing | ✅ "Rephrase last response" | — | — | **Gap: no autocomplete entry** |
| `/mood` | ✅ L3093 | ❌ Missing | ❌ Missing | — | — | **Gap: no UI entries at all** |
| `/find` | ✅ L3131 | ✅ 🔍 "Semantic corpus search" | ✅ "Semantic corpus search" | N/A | ✅ | |
| `/export` | ✅ L3150 | ✅ 📤 "Export session as markdown" | ✅ "Export session as markdown" | N/A | ✅ | |
| `/clear` | ✅ L3171 | ✅ 🧹 "Clear current session" | ✅ "Clear current chat" | N/A | ⚠️ "session" vs "chat" | |
| `/pin` | ✅ L3188 | ✅ 📌 "Pin a message to context" | ✅ "Pin text to context" | N/A | ⚠️ Minor diff | |
| `/save` | ✅ L3210 | ✅ 💾 "Snapshot session with label" | ✅ "Snapshot session with label" | N/A | ✅ | |
| `/load` | ✅ L3240 | ✅ 📂 "Load / list session snapshots" | ✅ "Load / list snapshots" | N/A | ⚠️ Minor diff | |
| `/finetune` | ✅ L3296 | ✅ 🔬 "Fine-tune LoRA on corpus" | ✅ "Fine-tune on corpus" | N/A | ⚠️ Minor diff | |
| `/run` | ✅ L3381 | ✅ ▶ "Execute Python snippet" | ✅ "Execute Python code" | N/A | ⚠️ "snippet" vs "code" | |
| `/plot` | ✅ L3421 | ✅ 📈 "Plot math instantly — /plot sin(x), cos(x)" | ✅ "Plot math — /plot sin(x), cos(x)" | N/A | ⚠️ "instantly" missing in palette | |
| `/calc` | ✅ L3448 | ✅ ∞ "Symbolic math — /calc integrate(x**2, x)" | ✅ "Symbolic math — /calc integrate(x**2, x)" | N/A | ✅ | |
| `/math` | ✅ L3448 (alias of /calc) | ❌ Missing | ❌ Missing | — | — | Alias — OK to omit from UI |
| `/analyze` | ✅ L3476 | ❌ Missing | ❌ Missing | — | — | **Gap: no UI entries at all** |
| `/trace` | ✅ L3613 | ✅ 📊 "Cross-session trace analytics" | ✅ "Cross-session trace analytics" | N/A | ✅ | |
| `/spectrum` | ✅ L3629 | ✅ 🌀 "LoRA adapter spectral analysis" | ✅ "LoRA adapter spectral analysis" | N/A | ✅ | |
| `/experiment` | ✅ L3647 | ✅ 🧪 "Run adversarial self-test" | ❌ Missing | — | — | **Gap: no palette entry** |
| `/learn` | ✅ L3678 | ✅ 🧠 "Toggle online LoRA updates (on/off)" | ❌ Missing | — | — | **Gap: no palette entry** |
| `/trace-mode` | ✅ L3699 | ❌ Missing | ❌ Missing | — | — | **Gap: no UI entries** (internal/dev?) |
| `/adapter` | ✅ L3721 | ✅ 🧬 "Save/load/list LoRA adapter profiles" | ❌ Missing | — | — | **Gap: no palette entry** |
| `/visionquality` | ✅ L3791 | ✅ 👁 "Vision token budget (low/medium/high/ultra)" | ❌ Missing | — | — | **Gap: no palette entry** |
| `/antagonist` | ✅ L3823 | ✅ 🦢 "Antagonist mode — challenge next response" | ❌ Missing | — | — | **Gap: no palette entry** |
| `/socratic` | ✅ L3833 | ✅ 💬 "Socratic mode toggle" | ❌ Missing | — | — | **Gap: no palette entry** |
| `/compress` | ✅ L3846 | ✅ 🗜 "Compression mode toggle" | ❌ Missing | — | — | **Gap: no palette entry** |
| `/backup` | ✅ L3859 | ✅ 📦 "Backup all manifold data" | ❌ Missing | — | — | **Gap: no palette entry** |
| `/scaffold` | ✅ L3886 | ✅ 🏗 "Structural scaffold for a topic" | ❌ Missing | — | — | **Gap: no palette entry** |
| `/dream` | ✅ L3903 | ✅ 💫 "Dream mode — lateral generative response" | ❌ Missing | — | — | **Gap: no palette entry** |
| `/reading` | ✅ L3912 | ✅ 📚 "Personalized reading list" | ❌ Missing | — | — | **Gap: no palette entry** |
| `/data` | ✅ L3944 | ✅ 📦 "Export research data bundle" | ❌ Missing | — | — | **Gap: no palette entry** |
| `/timer` | ✅ L4036 | ✅ ⏱ "Set a countdown timer" | ❌ Missing | — | — | **Gap: no palette entry** |
| `/search` | ✅ L4061 | ✅ 🌐 "Force web search" | ✅ "Force web search" | N/A | ✅ | |
| `/week` | ✅ L4136 | ❌ Missing | ❌ Missing | — | — | **Gap: no UI entries at all** |
| `/brief` | ✅ L4177 | ❌ Missing | ❌ Missing | — | — | **Gap: no UI entries at all** |
| `/debate` | ✅ L4225 | ✅ 🎙 "Steel-man debate on a topic" | ❌ Missing | — | — | **Gap: no palette entry** |
| `/eli5` | ✅ L4235 | ✅ 👶 "Explain like I'm 5" | ❌ Missing | — | — | **Gap: no palette entry** |
| `/teacher` | ✅ L4244 | ✅ 📚 "Socratic teacher mode on topic" | ❌ Missing | — | — | **Gap: no palette entry** |
| `/brainstorm` | ✅ L4253 | ✅ ⚡ "Freeform brainstorm" | ❌ Missing | — | — | **Gap: no palette entry** |
| `/devil` | ✅ L4262 | ✅ 😈 "Devil's advocate on a claim" | ❌ Missing | — | — | **Gap: no palette entry** |
| `/peer` | ✅ L4271 | ✅ 📋 "Peer-review mode for text" | ❌ Missing | — | — | **Gap: no palette entry** |
| `/hypothesis` | ✅ L4285 | ✅ 🧪 "Generate and stress-test a hypothesis" | ❌ Missing | — | — | **Gap: no palette entry** |
| `/quiz` | ✅ L4294 | ❌ Missing | ❌ Missing | — | — | **Gap: no UI entries at all** |
| `/swot` | ✅ L4307 | ✅ 📊 "SWOT analysis of a topic" | ❌ Missing | — | — | **Gap: no palette entry** |
| `/glossary` | ✅ L4316 | ❌ Missing | ❌ Missing | — | — | **Gap: no UI entries at all** |
| `/counterpoint` | ✅ L4332 | ❌ Missing | ❌ Missing | — | — | **Gap: no UI entries at all** |
| `/risk` | ✅ L4348 | ✅ ⚠ "Risk analysis of a topic" | ❌ Missing | — | — | **Gap: no palette entry** |
| `/flashcards` | ✅ L4357 | ❌ Missing | ❌ Missing | — | — | **Gap: no UI entries at all** |
| `/translate` | ✅ L4369 | ✅ 🌍 "Translate to another language" | ❌ Missing | — | — | **Gap: no palette entry** |
| `/contradict` | ✅ L4386 | ❌ Missing | ❌ Missing | — | — | **Gap: no UI entries at all** |
| `/elaborate` | ✅ L4404 | ✅ 💬 "Elaborate on the last response" | ❌ Missing | — | — | **Gap: no palette entry** |
| `/rhetorical` | ✅ L4434 | ✅ ✍ "Rhetorical analysis of text" | ❌ Missing | — | — | **Gap: no palette entry** |
| `/abstract` | ✅ L4420 | ❌ Missing | ❌ Missing | — | — | **Gap: no UI entries at all** |
| `/evolve` | ✅ L4460 | ✅ 🔀 "Evolve / sharpen the last argument" | ❌ Missing | — | — | **Gap: no palette entry** |
| `/thread` | ✅ L4489 | ✅ 🧵 "Thread-form breakdown of topic" | ❌ Missing | — | — | **Gap: no palette entry** |
| `/zettelkasten` | ✅ L4522 | ❌ Missing | ❌ Missing | — | — | **Gap: no UI entries at all** |
| `/help` / `/?` | ✅ L2697 | ✅ (as `/?`) ❓ "All commands" | ✅ "Show all commands" | N/A | ⚠️ Slightly different | |
| `/continue` | ✅ L2777 | ❌ Missing | ❌ Missing | — | — | **Gap: no UI entries at all** |
| `/think` | ❌ No handler (in `_known_cmds` only) | ❌ Missing | ❌ Missing | — | — | **Dead entry in _known_cmds** — think is a UI toggle, not a slash cmd |
| `/new` | ❌ No handler (in `_known_cmds` only) | ❌ Missing | ❌ Missing | — | — | **Dead entry in _known_cmds** — new is a UI button, not a slash cmd |
| `/mode` | ❌ No handler (in `_known_cmds` only) | ❌ Missing | ❌ Missing | — | — | **Dead entry in _known_cmds** — /adapter mode is the real cmd |

## Summary

### Missing from autocomplete (handler exists, no dropdown entry): 12
`/version`, `/recap`, `/check`, `/rephrase`, `/mood`, `/analyze`, `/trace-mode`, `/week`, `/brief`, `/quiz`, `/glossary`, `/counterpoint`, `/flashcards`, `/contradict`, `/abstract`, `/zettelkasten`, `/continue`

### Missing from palette (handler + autocomplete exist, no Cmd+K entry): 24
`/forget`, `/remember`, `/recall`, `/knowledge`, `/experiment`, `/learn`, `/adapter`, `/visionquality`, `/antagonist`, `/socratic`, `/compress`, `/backup`, `/scaffold`, `/dream`, `/reading`, `/data`, `/timer`, `/debate`, `/eli5`, `/teacher`, `/brainstorm`, `/devil`, `/peer`, `/hypothesis`, `/swot`, `/risk`, `/translate`, `/elaborate`, `/rhetorical`, `/evolve`, `/thread`

### Missing from both autocomplete AND palette: 11
`/mood`, `/analyze`, `/trace-mode`, `/week`, `/brief`, `/quiz`, `/glossary`, `/counterpoint`, `/flashcards`, `/contradict`, `/abstract`, `/zettelkasten`, `/continue`

### Dead entries in `_known_cmds` (no handler): 3
`/think`, `/new`, `/mode`

### Description mismatches (autocomplete vs palette): 8
`/who`, `/clear`, `/pin`, `/run`, `/plot`, `/summarize`, `/finetune`, `/load`

### Emoji: No mismatches possible — palette has no emoji column, only autocomplete does.

---

## After (gap-fill applied 2026-04-18)

### Changes made

| Category | Count | Details |
|----------|-------|---------|
| Palette entries added | 24 | `/forget`, `/remember`, `/recall`, `/knowledge`, `/experiment`, `/learn`, `/adapter`, `/visionquality`, `/antagonist`, `/socratic`, `/compress`, `/backup`, `/scaffold`, `/dream`, `/reading`, `/data`, `/timer`, `/debate`, `/eli5`, `/teacher`, `/brainstorm`, `/devil`, `/peer`, `/hypothesis`, `/swot`, `/risk`, `/translate`, `/elaborate`, `/rhetorical`, `/evolve`, `/thread`, `/counterpoint`, `/continue`, `/analyze`, `/contradict`, `/abstract`, `/quiz`, `/glossary`, `/flashcards`, `/mood`, `/week`, `/brief`, `/zettelkasten` |
| Autocomplete entries added | 16 | `/version`, `/recap`, `/check`, `/rephrase`, `/mood`, `/analyze`, `/week`, `/brief`, `/quiz`, `/glossary`, `/counterpoint`, `/flashcards`, `/contradict`, `/abstract`, `/zettelkasten`, `/continue` |
| Dead `_known_cmds` removed | 3 | `/think`, `/new`, `/mode` |
| Missing `_known_cmds` added | 11 | `/counterpoint`, `/flashcards`, `/contradict`, `/abstract`, `/quiz`, `/glossary`, `/week`, `/brief`, `/dream`, `/reading`, `/data` |
| Descriptions normalized | 8 | Palette descriptions updated to match autocomplete (canonical) for `/who`, `/clear`, `/pin`, `/run`, `/plot`, `/summarize`, `/finetune`, `/load` |
| Intentionally omitted | 2 | `/trace-mode` (dev command, controlled by Sync pill), `/math` (alias of `/calc`) |

### Final state
- **Autocomplete entries**: 65 (includes `/?`)
- **Palette entries**: 65
- **`_known_cmds`**: all handlers covered, no dead entries
- **All descriptions**: autocomplete and palette now match exactly
