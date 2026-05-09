#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  COUPLED MANIFOLD — Uninstall
#  Removes everything: app, venv, model weights, user data.
#  Run: bash uninstall.sh
# ═══════════════════════════════════════════════════════════════

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  COUPLED MANIFOLD — Uninstall"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  This will remove:"
echo ""
echo "  1. Virtual environment (graceful_env)"
echo "  2. Graceful.app bundle"
echo "  3. All model weights from HuggingFace cache:"

HF_CACHE="$HOME/.cache/huggingface/hub"
FOUND_MODELS=0
for d in "$HF_CACHE"/models--mlx-community--gemma-* "$HF_CACHE"/models--sentence-transformers--*; do
    if [ -d "$d" ]; then
        SIZE=$(du -sh "$d" 2>/dev/null | cut -f1)
        echo "     - $(basename "$d") ($SIZE)"
        FOUND_MODELS=1
    fi
done
if [ "$FOUND_MODELS" -eq 0 ]; then
    echo "     (none found)"
fi

echo "  4. All user data (sessions, memory, checkpoints, corpus)"
echo ""

if [ -t 0 ]; then
    read -p "  Continue? This cannot be undone. (y/N): " CONFIRM
    if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
        echo "  Cancelled."
        exit 0
    fi
else
    echo "  Non-interactive — aborting for safety."
    echo "  Run interactively: bash uninstall.sh"
    exit 1
fi

echo ""

# 1. Virtual environment
if [ -d "$DIR/graceful_env" ]; then
    echo "  Removing virtual environment..."
    rm -rf "$DIR/graceful_env"
    echo "  ✓ Removed graceful_env"
fi

# 2. Graceful.app
if [ -d "$DIR/Graceful.app" ]; then
    echo "  Removing Graceful.app..."
    rm -rf "$DIR/Graceful.app"
    echo "  ✓ Removed Graceful.app"
fi

# 3. Model caches
for d in "$HF_CACHE"/models--mlx-community--gemma-*; do
    if [ -d "$d" ]; then
        echo "  Removing $(basename "$d")..."
        rm -rf "$d"
        echo "  ✓ Removed"
    fi
done
for d in "$HF_CACHE"/models--sentence-transformers--*; do
    if [ -d "$d" ]; then
        echo "  Removing $(basename "$d")..."
        rm -rf "$d"
        echo "  ✓ Removed"
    fi
done

# 4. User data
if [ -d "$DIR/manifold_data" ]; then
    echo "  Removing all user data..."
    rm -rf "$DIR/manifold_data"
    echo "  ✓ Removed manifold_data"
fi

# 5. Build artifacts
rm -rf "$DIR"/__pycache__ "$DIR"/.pytest_cache
find "$DIR" -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
find "$DIR" -name '*.pyc' -delete 2>/dev/null || true

echo ""
echo "  ✓ Uninstall complete."
echo ""
echo "  To fully remove Graceful, delete this folder:"
echo "  rm -rf \"$DIR\""
echo ""
