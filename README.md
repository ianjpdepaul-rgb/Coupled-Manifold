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

---

## Features

- **Streaming chat** — responses appear word by word
- **Think mode** — model reasons step by step before answering
- **Memory & corpus RAG** — retrieves relevant context from saved notes and documents
- **LoRA online learning** — adapter weights update from your conversations
- **CSV / JSON export** — download full session transcripts
- **Voice TTS** — text-to-speech readback of responses
- **Drag-and-drop files** — drop PDFs, documents, or images into the chat

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

**"Python 3 not found"** — Install from [python.org](https://python.org) and re-run `setup.sh`.

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

*MIT License — Ian J. Preston-Campbell (Ian De Paul), 2026*  
*Model: Gemma 4 E4B by Google DeepMind — subject to [Gemma Terms of Use](https://ai.google.dev/gemma/terms)*
