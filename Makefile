.PHONY: test test-quick test-full test-manifold test-harness test-harness-managed smoke ux start start-test unit integration lint format clean

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
