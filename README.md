# Graceful — Coupled Manifold

A private, local AI companion that runs entirely on your Mac. No internet required. No data leaves your machine.

---

## What it is

Graceful runs a small language model (Gemma 4 E4B, 4-bit quantized) directly on your Apple Silicon chip using MLX — Apple's native machine learning framework. It is fast, private, and fully offline after the one-time setup.

Underneath the chat interface, the system measures the curvature of its own reasoning in real time — a number called the Hessian trace. When the geometry of the model's attention drifts toward narrow, repetitive patterns, the system detects it and adapts. You are not just talking to a model; the model is monitoring itself as it talks to you.

This is a research deployment of the Coupled Manifold Safety Framework.

---

## Requirements

- **Mac with Apple Silicon** (M1, M2, M3, or M4) — required for MLX
- **Python 3.9 or later** — get it at [python.org](https://python.org)
- **~6 GB RAM** available during use
- **~5 GB free disk space** for the model (one-time download)
- **No HuggingFace account required** — the model is public

---

## Setup (one time only)

Open Terminal, drag `setup.sh` into it, and press Enter. Or run:

```
bash setup.sh
```

The script will:

1. Check Python and confirm Apple Silicon
2. Install Homebrew dependencies (if needed)
3. Create an isolated Python environment
4. Install all required libraries
5. Download the Gemma 4 E4B model (~4 GB, one time only)
6. Build `Graceful.app` with the correct icon

**No HuggingFace token is needed.** The model (`mlx-community/gemma-4-e4b-it-4bit`) is publicly available. You can press Enter to skip the token prompt.

---

## First-time launch

The first time you open the app:

1. A Terminal window appears showing boot progress
2. The model loads into memory — this takes **15-30 seconds** and the terminal will show loading bars
3. Once you see `Ready.` and a URL, the browser opens automatically to `http://localhost:7860`
4. The chat interface appears. Type anything to start

The model download and loading only happens once per session. After the first load, responses are fast.

---

## Every launch after setup

Double-click **`Graceful.app`**.

That's it. The app opens a browser window at `http://localhost:7860`.

---

## Commands

Type these in the chat input:

| Command | What it does |
|---|---|
| `/who` | Show identity profile |
| `/find [query]` | Search memory and corpus |
| `/iam [statement]` | Add a personal statement to identity |
| `/recap` | Summarize the current session |
| `/save` | Save session manually |
| `/load` | Load a previous session |
| `/export` | Export session as CSV or JSON |
| `/stats` | Show session statistics and trace history |
| `/help` | List all commands |
| `/clear` | Clear the chat display |
| `/trace` | Show current Hessian trace value |
| `/spectrum` | Show curvature spectrum chart |
| `/adapter` | Show adapter weight status |
| `/experiment` | Run adversarial self-test |
| `/think` | Toggle step-by-step reasoning mode |
| `/help` | List all commands (50+) |

The table above shows the most common commands. Type `/help` in the chat for the full list, including research tools (`/analyze`, `/hypothesis`, `/zettelkasten`), teaching modes (`/socratic`, `/eli5`, `/teacher`), and session management (`/backup`, `/compress`, `/scaffold`).

---

## Features

- **Geometric self-monitoring** — real-time Hessian trace measurement of the model's reasoning. The medulla bar above the chat shows the live geometric state: trace value, mode, drift, pathology indicators.
- **Online learning with safety gate (LoRA + anti-LoRA)** — adapter weights update from your conversations, but only when the geometric monitor confirms healthy learning conditions. Anti-LoRA adapter applies roughening when the system detects pathological convergence (basin lock, shortcut formation).
- **SnobLine controller** — adaptive switching between learning and intervention modes based on trace dynamics. The system can refuse to learn from a conversation if the geometry indicates the conversation is unhealthy.
- **Interoceptive feedback** — the model receives a structured self-report of its own geometric state in context each turn (heartbeat, momentum, anxiety, drift, stability instructions).
- **Streaming chat** — responses appear word by word
- **Think mode** — model reasons step by step before answering
- **Memory & corpus RAG** — retrieves relevant context from saved notes and documents
- **Knowledge graph (Borgesian Map)** — visual map of concepts and thinkers from your conversations and corpus
- **CSV / JSON / Markdown export** — download full session transcripts including geometric trace data
- **Voice TTS** — text-to-speech readback of responses
- **Drag-and-drop files** — drop PDFs, documents, or images into the chat
- **Personas and modes** — configurable personality, plus session-scoped mode shifts (`/socratic`, `/antagonist`, `/compress`, `/eli5`, etc.)

---

## How it works

Underneath the chat interface, the system measures the curvature of its own reasoning in real time — a number called the Hessian trace. When the geometry of the model's attention drifts toward narrow, repetitive patterns (basin collapse, shortcut formation), the system detects it and adapts:

1. **Detection** — The Hessian trace is computed periodically during inference. The SnobLine controller watches the trace trajectory.
2. **Gate** — When the geometry indicates pathological convergence, online learning is suppressed. The system refuses to absorb patterns from unhealthy conversations.
3. **Intervention** — The anti-LoRA adapter applies counter-roughening to push the loss landscape back toward healthy curvature.
4. **Feedback** — The model receives a structured description of its own geometric state in context each turn, allowing it to be aware of and respond to its own internal dynamics.

This is a research deployment of the Coupled Manifold Safety Framework. The framework's claim is that AI safety is best understood as a geometric property of the human-AI coupled system, not as an external constraint imposed on the model. The full theoretical paper is on ResearchGate (linked below).

---

## Files in this folder

| File | Purpose |
|---|---|
| `app.py` | The application |
| `setup.sh` | Run once to set everything up |
| `launch.py` | App launcher helper |
| `icon.png` | App icon |
| `Graceful.app` | Created by setup.sh — your launcher |
| `manifold_data/` | Your sessions, memory, and model checkpoints |

---

## Troubleshooting

**"app is damaged" or won't open** — macOS quarantine flag. Run this in Terminal:
```
xattr -cr Graceful.app
```
Then try opening it again.

**Slow first response** — Normal. The model takes 15–30 seconds to load on first use each session. After that, responses are fast.

**"Python 3 not found"** — Install via Homebrew: `brew install python@3.11`. If you don't have Homebrew, install it first from [brew.sh](https://brew.sh), or download Python from [python.org](https://python.org) and re-run `setup.sh`.

**Model download fails** — Check your internet connection. The model is ~4 GB and downloads from HuggingFace. No token is required.

**App opens but browser stays blank** — Wait 10–15 seconds for the server to start, then refresh the page.

---

## Data & privacy

Everything is stored locally in `manifold_data/`:

- `sessions/` — full conversation logs with trace values
- `checkpoints/` — adapter weights (your personalized model state)
- `corpus/` — documents you've added to memory
- `logs/` — continuous trace history
- `identity/` — your `/iam` statements and identity profile

Nothing is sent anywhere. There are no analytics, no telemetry, no external calls except the optional search APIs you configure yourself.

---

## Research context

This application is a deployment of the Coupled Manifold Safety Framework. The theoretical foundation, experimental results, and full paper are available at:

[ResearchGate — Ian J. Preston-Campbell](https://www.researchgate.net/profile/Ian-Preston-Campbell)

[Substack — ijpc43.substack.com](https://ijpc43.substack.com)

---

## Known limitations

- First response per session takes 15-30 seconds while the model loads into memory
- Hessian trace computation runs one turn behind (current turn's trace reflects the previous response)
- Search APIs are optional and require manual configuration during setup
- The app requires Apple Silicon — Intel Macs are not supported
- `/run` executes Python in a restricted namespace (no `__builtins__`) but without a filesystem sandbox — code can access local files
- This is a research deployment; expect rough edges

---

*MIT License — Ian J. Preston-Campbell (Ian De Paul), 2026*  
*Model: Gemma 4 E4B by Google DeepMind — subject to [Gemma Terms of Use](https://ai.google.dev/gemma/terms)*
