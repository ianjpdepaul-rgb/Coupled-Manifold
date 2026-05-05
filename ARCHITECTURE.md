# Graceful -- Architecture Map

## Current State

All core logic lives in `app.py` (~8,193 lines). Supporting modules exist but were
extracted ad hoc. Test suite: 136 tests across 9 files (`tests/test_*.py`).

### Source Files

| File | Lines | Role |
|---|---|---|
| `app.py` | ~8,193 | Startup, model, trace, chat, UI, server |
| `graceful/config.py` | 64 | Constants, LR schedule, keys helper (Phase 2) |
| `graceful/flattery.py` | 57 | Sycophancy scoring, greeting detection (Phase 2) |
| `graceful/snobline.py` | 169 | Trace-based LoRA/Anti-LoRA controller (Phase 2) |
| `graceful/dual_adapter.py` | 39 | LoRA + Anti-LoRA nn.Module (Phase 2) |
| `memory.py` | 1,043 | Corpus, history, identity model (already extracted) |
| `search_stack.py` | 732 | Web search pipeline (already extracted) |
| `model_router.py` | 335 | Query routing heuristics (already extracted) |
| `trace_analytics.py` | 407 | Cross-session trace analysis (already extracted) |
| `context_budget.py` | 148 | Token budget tracking (already extracted) |
| `search_sanitizer.py` | 595 | Search query sanitization (already extracted) |
| `ingest_concordance.py` | 197 | Corpus ingestion helper (already extracted) |

### Subsystems Inside app.py

| Subsystem | Lines | Description |
|---|---|---|
| Startup Guards | 24-145 | Python version, dependency, disk, port checks |
| Config | 147-383 | Constants, LR schedules, lockfile, PWA icons |
| DualAdapter | 384-418 | LoRA + Anti-LoRA nn.Module |
| SnobLine | 419-585 | Trace-based learning controller |
| Flattery Detection | 586-663 | Sycophancy scoring, diversity gate for learning |
| Model Pair | 664-705 | Model wrapper (single-model, legacy dual API) |
| Hessian Trace | 706-996 | MLX-based trace computation (sampled + exact) |
| Search Stack | 997-1041 | Thin wrapper around search_stack.py |
| Helpers | 1042-1209 | Adapters, checkpoints, mode switching, history I/O |
| Python Executor | 1210-1679 | Safe code execution sandbox, /plot, /calc |
| Load Model | 1680-1930 | Model loading, boot, warmup, auto-summarize |
| Temporal Awareness | 1931-1962 | Session timing, boot context |
| Session Weight Fusion | 1963-2104 | Cross-session LoRA weight accumulation |
| Semantic Drift + Interoception | 2105-2374 | Corpus centroid, diversity check, drift, redirect |
| Voice Output | 2375-2392 | TTS stub |
| Pinned Context | 2393-2399 | User-pinned context management |
| Auto Code Execution | 2400-2454 | Inline code detection and execution |
| Tool Dispatch | 2455-2565 | Tool tag extraction and self-correction |
| Self-Experiment | 2566-2664 | Adversarial self-test harness |
| Chat | 2665-5804 | Command handlers, generation, medulla, post-processing |
| PWA Head JS | 5805-5844 | Service worker injection |
| FastAPI + UI | 5845-7921 | All API routes, SSE streaming, middleware |
| HTML | 7922-7942 | Static HTML loader |
| Test Injection | 7943-8122 | Test-mode-only endpoints |
| Launch | 8123-8156 | Uvicorn startup |

### Thread Safety (as of 2026-05-05)

| Lock | Protects | Added |
|------|----------|-------|
| `_session_lock` | `_session_history`, `_active_session_ts` | original |
| `_model_lock` | model forward pass, `_user_request_pending` | original |
| `_pending_trace_lock` | `_pending_trace` | original |
| `_pinned_context_lock` | `_pinned_context` list | fix pass |

Key patterns:
- Lock-and-snapshot: acquire lock, `list(data)`, release, iterate snapshot
- `_user_request_pending` counter: incremented at API entry, decremented at
  `_model_lock.acquire()` or via `finally` block for early-return commands
- Session-scoped state (`_recent_response_vecs`, `_socratic_mode`, `_compress_mode`,
  `ctrl.*`) cleared on `/api/new` and `/api/reset`

### Security Boundaries

| Location | Mechanism | Note |
|----------|-----------|------|
| `app.py` `/run` exec() | Restricted namespace (no `__builtins__`) | Local-only; no filesystem sandbox |
| `memory.py` pickle.load | Trusted local files only | Files in `manifold_data/` |
| `launch.py` subprocess | `shell=False` with list args | No shell injection |
| `search_sanitizer.py` MD5 | `usedforsecurity=False` | Content dedup keys only |

## Proposed Module Structure

```
graceful/
  __init__.py
  __main__.py          # startup, signal handlers, lock, uvicorn.run
  config.py            # constants, LR schedules, feature flags
  dual_adapter.py      # DualAdapter nn.Module
  snobline.py          # SnobLine controller class
  flattery.py          # compute_flattery_score, helpers
  diversity.py         # DiversityTracker (wraps check_response_diversity + learn gate)
  hessian.py           # Trace computation (sampled + exact)
  model_pair.py        # ModelPair wrapper
  temporal.py          # Temporal awareness, session weight fusion
  interoception.py     # Semantic drift, interoceptive block builder
  experiment.py        # Self-experiment harness
  python_executor.py   # Safe code sandbox, /plot, /calc
  chat.py              # Chat generator, command dispatch, medulla
  server.py            # FastAPI app, all API routes, SSE, middleware
  helpers.py           # Adapter utils, checkpoint I/O, mode switching

app.py                 # 3-line shim: from graceful.__main__ import main; main()
memory.py              # (unchanged)
search_stack.py        # (unchanged)
model_router.py        # (unchanged)
trace_analytics.py     # (unchanged)
context_budget.py      # (unchanged)
search_sanitizer.py    # (unchanged)
```

## Extraction Order (Phases 2-4)

**Phase 2 -- Pure logic (no model dependencies):**
1. ~~`config.py` -- constants only, no imports of mlx~~ ✅ done
2. ~~`flattery.py` -- string matching, no external deps~~ ✅ done
3. ~~`snobline.py` -- numpy only~~ ✅ done
4. ~~`dual_adapter.py` -- mlx.nn but no model instance needed~~ ✅ done
5. `diversity.py` -- numpy + sentence-transformers

**Phase 3 -- Model-dependent modules:**
1. `hessian.py` -- needs mlx model
2. `temporal.py` -- needs memory + model
3. `interoception.py` -- needs corpus embedder
4. `experiment.py` -- needs chat function
5. `python_executor.py` -- self-contained sandbox

**Phase 4 -- Chat orchestration and server:**
1. `chat.py` -- the big one, depends on everything above
2. `server.py` -- FastAPI routes, depends on chat
3. `__main__.py` -- startup glue

## Phase 5 -- Mutable State Audit

Module-level mutable state to wrap in classes:
- `trace_history_live` -> `TraceHistory`
- `_session_history`, `_session_lock`, `_active_session_ts` -> `SessionState`
- `ctrl_active`, `pair`, `mem` -> owned by `__main__`, passed explicitly

## Testing Strategy

- **Unit tests** (`tests/test_*.py`): import extracted modules directly
  - `test_config.py` (13), `test_context_budget.py` (12), `test_flattery.py` (11),
    `test_ingest_concordance.py` (6), `test_model_router.py` (16),
    `test_personality.py` (39), `test_search_sanitizer.py` (18),
    `test_snobline.py` (15), `test_trace_analytics.py` (6)
  - Total: 136 passing, 3 deselected (require MLX hardware)
- **Integration tests** (`tests/test_integration.py`): HTTP against running server
- **Existing harnesses** (`test_smoke.py`, `test_full_ux.py`, etc.): kept as-is
- All tests run via `python3 -m pytest tests/ -x -q`
