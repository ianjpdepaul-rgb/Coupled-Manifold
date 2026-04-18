.PHONY: test test-quick test-full test-manifold smoke ux start

# Quick: smoke tests only (no server needed)
test-quick: smoke

# Full: smoke + full adversarial UX suite (server must be running)
test-full: smoke
	python3 test_full_ux.py --port 7860

# Legacy: smoke + basic UX
test: smoke ux

# Pure-logic unit tests (no server needed)
smoke:
	python3 test_smoke.py

# Basic UX integration tests (requires running server on port 7860)
ux:
	python3 test_ux.py --port 7860

# Coupled Manifold experimental validation (requires running server)
test-manifold:
	python3 test_manifold.py --port 7860

# Start the server
start:
	python3 app.py
