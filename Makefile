.PHONY: test test-quick test-full test-manifold test-harness test-harness-managed smoke ux start start-test unit integration lint format clean agent-test agent-baseline agent-regression

# ── pytest-based targets ────────────────────────────────

# Unit tests only (no server needed)
unit:
	python3 -m pytest tests/ -m "not integration" -v

# Integration tests (requires running server on port 7860)
integration:
	python3 -m pytest tests/ -m integration -v

# All pytest tests (unit + integration if server is up)
test: unit

# ── Legacy test harnesses ───────────────────────────────

# Quick: smoke tests only (no server needed)
test-quick: smoke

# Full: smoke + full adversarial UX suite (server must be running)
test-full: smoke
	python3 test_full_ux.py --port 7860

# Pure-logic unit tests (no server needed) — legacy harness
smoke:
	python3 test_smoke.py

# Basic UX integration tests (requires running server on port 7860)
ux:
	python3 test_ux.py --port 7860

# Coupled Manifold experimental validation (requires running server)
test-manifold:
	python3 test_manifold.py --port 7860

# Trace replay harness (requires server running with GRACEFUL_TEST_MODE=1)
test-harness:
	python3 trace_replay_harness.py --verbose

# Trace replay harness with managed server (starts/stops automatically)
test-harness-managed:
	python3 trace_replay_harness.py --managed --verbose

# ── Synthetic user agent ───────────────────────────────

# Run all agent personas (requires running server)
agent-test:
	python3 -m tests.chat_agent.runner --all

# Run all personas and save as baseline
agent-baseline:
	python3 -m tests.chat_agent.runner --all --save-baseline

# Run all personas and compare against saved baseline
agent-regression:
	@LATEST=$$(ls -t tests/chat_agent/logs/run_*.jsonl 2>/dev/null | head -1); \
	if [ -z "$$LATEST" ]; then \
		echo "No run logs found. Run 'make agent-test' first."; exit 1; \
	fi; \
	if [ ! -f tests/chat_agent/baselines/baseline.jsonl ]; then \
		echo "No baseline found. Run 'make agent-baseline' first."; exit 1; \
	fi; \
	python3 -m tests.chat_agent.runner --compare-baseline tests/chat_agent/baselines/baseline.jsonl "$$LATEST"

# ── Dev utilities ───────────────────────────────────────

# Start the server
start:
	python3 app.py

# Start the server in test mode
start-test:
	GRACEFUL_TEST_MODE=1 python3 app.py

# Lint (ruff if installed, else flake8)
lint:
	@which ruff >/dev/null 2>&1 && ruff check . || python3 -m flake8 --max-line-length=120 --exclude=graceful_env .

# Format (ruff if installed, else black)
format:
	@which ruff >/dev/null 2>&1 && ruff format . || python3 -m black --line-length=120 --exclude=graceful_env .

# Clean build artifacts
clean:
	find . -path ./graceful_env -prune -o -name '__pycache__' -print -exec rm -rf {} +
	find . -path ./graceful_env -prune -o -name '*.pyc' -print -exec rm -f {} +
	rm -rf .pytest_cache
