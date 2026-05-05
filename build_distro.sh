#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  COUPLED MANIFOLD — Distribution Builder
#  No Python required on Grace's machine. uv handles everything.
#  Usage: bash build_distro.sh
# ═══════════════════════════════════════════════════════════════

set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
DIST_DIR="/tmp/graceful_dist"
DEST="$DIST_DIR/Graceful_App"
APP="$DEST/Graceful.app"
MACOS="$APP/Contents/MacOS"
RES="$APP/Contents/Resources"
OUT="$HOME/Desktop/Graceful_App.zip"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Graceful App — Distribution Build"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── Clean slate ──────────────────────────────────────────────────
rm -rf "$DIST_DIR"
mkdir -p "$DEST" "$MACOS" "$RES"
echo "✅  Build directory ready."

# ── Download uv binary (Apple Silicon) ───────────────────────────
echo ""
echo "  Downloading uv (Python bootstrapper)..."
UV_URL="https://github.com/astral-sh/uv/releases/latest/download/uv-aarch64-apple-darwin.tar.gz"
UV_TMP="$DIST_DIR/uv.tar.gz"
curl -L --progress-bar "$UV_URL" -o "$UV_TMP"
tar -xzf "$UV_TMP" -C "$DIST_DIR"
# uv extracts to uv-aarch64-apple-darwin/uv
UV_BIN=$(find "$DIST_DIR" -name "uv" -type f | head -1)
cp "$UV_BIN" "$MACOS/uv"
chmod +x "$MACOS/uv"
rm -f "$UV_TMP"
echo "✅  uv bundled ($(du -sh "$MACOS/uv" | cut -f1))."

# ── Copy Python source files ─────────────────────────────────────
echo ""
echo "  Copying source files..."
# Copy all Python source files (exclude test files and build scripts)
for f in "$DIR"/*.py; do
    fname=$(basename "$f")
    # Skip test files and build artifacts
    case "$fname" in
        test_*.py|*_test.py|build_*.py) continue ;;
    esac
    cp "$f" "$DEST/$fname" && echo "    + $fname"
done
# Copy non-Python required files
for f in requirements.txt README.md LICENSE icon.png setup.sh default_system_prompt.txt; do
    [ -f "$DIR/$f" ] && cp "$DIR/$f" "$DEST/$f" && echo "    + $f"
done

# ── Copy UI ──────────────────────────────────────────────────────
mkdir -p "$DEST/manifold_data/static"
[ -d "$DIR/manifold_data/static" ] && cp -R "$DIR/manifold_data/static/." "$DEST/manifold_data/static/"
[ -d "$DIR/static" ] && { mkdir -p "$DEST/static"; cp -R "$DIR/static/." "$DEST/static/"; }
echo "✅  UI copied."

# ── Clean data skeleton ──────────────────────────────────────────
mkdir -p "$DEST/manifold_data/"{checkpoints,corpus,sessions,logs,identity,backups}
cat > "$DEST/manifold_data/identity.json" << 'EOF'
{"thinkers":[],"concepts":[],"raw_notes":[],"iam_statements":[]}
EOF

# Verify zero leaks
LEAK=$(find "$DEST/manifold_data" \( -name "*.json" ! -name "identity.json" \) \
    -o -name "*.npz" -o -name "*.jsonl" 2>/dev/null | wc -l | tr -d ' ')
[ "$LEAK" -gt "0" ] && {
    find "$DEST/manifold_data" \( -name "*.json" ! -name "identity.json" \) -delete 2>/dev/null || true
    find "$DEST/manifold_data" \( -name "*.npz" -o -name "*.jsonl" \) -delete 2>/dev/null || true
}
# Also clean any stray session files
find "$DEST/manifold_data/sessions" -name "*.md" -delete 2>/dev/null || true
find "$DEST/manifold_data/sessions" -name "*.txt" -delete 2>/dev/null || true
echo "✅  Clean data skeleton — zero personal files."

# ── Build Graceful.app ────────────────────────────────────────────
echo ""
echo "  Building Graceful.app..."

cat << 'LAUNCHER' > "$MACOS/Graceful"
#!/bin/bash
# Graceful.app — GUI-only launcher
# Finds Python, runs launch.py. Never opens Terminal.

# ── Translocation guard ──────────────────────────────────────────
case "$(cd "$(dirname "$0")" && pwd)" in
  */AppTranslocation/*)
    osascript -e 'display dialog "macOS is running Graceful from a temporary quarantine folder.\n\nPlease move Graceful.app to your Applications folder or Desktop, then double-click it again." buttons {"OK"} default button 1 with title "Graceful — Move Required" with icon caution'
    exit 0
    ;;
esac

APP_DIR="$(cd "$(dirname "$0")" && pwd)"   # .app/Contents/MacOS
ROOT="$(cd "$APP_DIR/../../.." && pwd)"    # the Graceful_App folder
UV="$APP_DIR/uv"
VENV="$ROOT/graceful_env"

cd "$ROOT"
mkdir -p "$ROOT/manifold_data/logs"
xattr -cr "$ROOT/Graceful.app" 2>/dev/null || true
export MANIFOLD_DIR="$ROOT"

# ── Find Python ──────────────────────────────────────────────────
# Tier 1: venv already exists from a previous setup
if [ -f "$VENV/bin/python3" ]; then
    PYTHON="$VENV/bin/python3"

# Tier 2: system Python 3.9+ with tkinter
elif command -v python3 &>/dev/null; then
    SYS_OK=$(python3 -c '
import sys
if sys.version_info >= (3, 9):
    try:
        import tkinter
        print("yes")
    except ImportError:
        print("no-tk")
else:
    print("no-ver")
' 2>/dev/null)
    if [ "$SYS_OK" = "yes" ]; then
        PYTHON="$(command -v python3)"
    else
        PYTHON=""
    fi
else
    PYTHON=""
fi

# Tier 3 fallback: use bundled uv to install Python
if [ -z "$PYTHON" ] && [ -x "$UV" ]; then
    "$UV" python install 3.12 --quiet 2>/dev/null
    UV_PY=$("$UV" python find 3.12 2>/dev/null)
    if [ -n "$UV_PY" ] && [ -x "$UV_PY" ]; then
        PYTHON="$UV_PY"
    fi
fi

# ── Fatal: no Python found ───────────────────────────────────────
if [ -z "$PYTHON" ]; then
    osascript -e 'display dialog "Python 3.9 or later is required but was not found.\n\nInstall Python from python.org and try again." buttons {"OK"} default button 1 with title "Graceful — Python Required" with icon stop'
    exit 1
fi

# ── Launch ───────────────────────────────────────────────────────
exec "$PYTHON" "$ROOT/launch.py"
LAUNCHER
chmod +x "$MACOS/Graceful"

# ── Info.plist ────────────────────────────────────────────────────
cat << 'PLIST' > "$APP/Contents/Info.plist"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
    <key>CFBundleName</key>              <string>Graceful</string>
    <key>CFBundleDisplayName</key>       <string>Graceful</string>
    <key>CFBundleIdentifier</key>        <string>com.coupledmanifold.graceful</string>
    <key>CFBundleVersion</key>           <string>3.0</string>
    <key>CFBundleShortVersionString</key><string>3.0</string>
    <key>CFBundleExecutable</key>        <string>Graceful</string>
    <key>CFBundlePackageType</key>       <string>APPL</string>
    <key>CFBundleIconFile</key>          <string>AppIcon</string>
    <key>LSMinimumSystemVersion</key>    <string>12.0</string>
    <key>NSHighResolutionCapable</key>   <true/>
</dict></plist>
PLIST

# ── App icon ──────────────────────────────────────────────────────
ICON_SRC=""
for c in "$DIR/icon.png" "$DIR/slash_icon_v4.png" "$DIR/app_icon.png"; do
    [ -f "$c" ] && ICON_SRC="$c" && break
done
if [ -n "$ICON_SRC" ] && command -v sips &>/dev/null && command -v iconutil &>/dev/null; then
    ICONSET="$DIST_DIR/icon.iconset"
    mkdir -p "$ICONSET"
    for sz in 16 32 128 256 512; do
        sips -z $sz $sz "$ICON_SRC" --out "$ICONSET/icon_${sz}x${sz}.png" &>/dev/null
        d=$((sz*2))
        sips -z $d $d "$ICON_SRC" --out "$ICONSET/icon_${sz}x${sz}@2x.png" &>/dev/null
    done
    iconutil -c icns "$ICONSET" -o "$RES/AppIcon.icns" 2>/dev/null || true
    rm -rf "$ICONSET"
    echo "✅  Icon set."
fi

xattr -cr "$APP" 2>/dev/null || true
echo "✅  Graceful.app built."

# ── Zip ───────────────────────────────────────────────────────────
echo ""
echo "  Zipping..."
rm -f "$OUT"
cd "$DIST_DIR"
zip -r "$OUT" "Graceful_App" \
    -x "*.DS_Store" -x "__pycache__/*" -x "*.pyc" > /dev/null

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✅  Graceful_App.zip on your Desktop"
echo "    Path: $OUT"
SIZE=$(du -sh "$OUT" | cut -f1)
echo "    Size: $SIZE"
echo ""
echo "  Grace's flow:"
echo "  1. Unzip → Graceful_App folder"
echo "  2. Double-click Graceful.app"
echo "  3. Click Setup  (nothing to pre-install)"
echo "  4. Wait ~5 min  (deps + 4GB model, one time)"
echo "  5. Browser opens → done forever"
echo "═══════════════════════════════════════════════════════════"
echo ""

rm -rf "$DIST_DIR"
