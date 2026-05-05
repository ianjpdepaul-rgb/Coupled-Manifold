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
for f in requirements.txt README.md LICENSE icon.png setup.sh grace_system_prompt.txt; do
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
# Graceful.app — self-bootstrapping launcher
# uv is bundled: no Python needed on the host machine.

# ── Translocation guard ──────────────────────────────────────────
# macOS quarantines apps opened from a zip. The OS copies the .app
# to a random /private/var/folders/.../AppTranslocation/ path, which
# breaks every relative path the launcher uses.
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
PY="$VENV/bin/python3"
LOG="$ROOT/manifold_data/logs/launch.log"

cd "$ROOT"
mkdir -p "$ROOT/manifold_data/logs"
xattr -cr "$ROOT/Graceful.app" 2>/dev/null || true
export MANIFOLD_DIR="$ROOT"

# ── First run ─────────────────────────────────────────────────────
if [ ! -f "$PY" ]; then

    REPLY=$(osascript << 'DIALOG'
display dialog "Welcome to Graceful.

First launch will:
  • Install everything automatically  (~3 min)
  • Download the AI model one time    (~4 GB)
  • Open your browser when ready

No Python installation needed — everything is bundled.

Click Setup to begin." \
        buttons {"Cancel", "Setup"} default button 2 \
        with title "Graceful — First Launch" \
        with icon note
return button returned of result
DIALOG
    )

    [ "$REPLY" = "Cancel" ] && exit 0

    # Write the auto-setup script
    SETUP_SCRIPT="$ROOT/manifold_data/logs/auto_setup.sh"
    cat > "$SETUP_SCRIPT" << AUTOSETUP
#!/bin/bash
set -e
cd "$ROOT"
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Graceful — First-time Setup"
echo "═══════════════════════════════════════════════════════"
echo ""

# 1. Install Python 3.12 via uv (no system Python needed)
echo "  Installing Python 3.12..."
"$UV" python install 3.12 --quiet
echo "✅  Python 3.12 ready."

# 2. Create virtualenv
echo "  Creating virtual environment..."
"$UV" venv "$VENV" --python 3.12 --quiet
echo "✅  Environment ready."

# 3. Bootstrap pip inside venv
"$PY" -m ensurepip --upgrade 2>/dev/null || true
if [ ! -f "$VENV/bin/pip" ]; then
    osascript -e 'display dialog "pip could not be installed in the virtual environment.\n\nPlease install Python from python.org and try again." buttons {"OK"} default button 1 with title "Graceful — Setup Error" with icon stop'
    exit 1
fi

# 4. Install dependencies
echo "  Installing dependencies (this takes ~2 min)..."
"$UV" pip install -r "$ROOT/requirements.txt" --python "$PY" --quiet
echo "✅  Dependencies installed."

# 5. Homebrew + tesseract (optional — for OCR on image PDFs)
if ! command -v brew &>/dev/null; then
    BREW_REPLY=\$(osascript -e 'display dialog "Install Homebrew?\n\n(Needed for OCR on image PDFs — app works without it.)" buttons {"No thanks", "Install"} default button 1 with title "Graceful — Optional" with icon note' 2>/dev/null) || BREW_REPLY=""
    case "\$BREW_REPLY" in
        *Install*)
            echo "  Installing Homebrew..."
            /bin/bash -c "\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" < /dev/null || true
            if [ -f "/opt/homebrew/bin/brew" ]; then
                eval "\$(/opt/homebrew/bin/brew shellenv)"
            fi
            ;;
        *)
            echo "  Homebrew skipped — OCR on image PDFs won't be available."
            ;;
    esac
fi
if command -v brew &>/dev/null && ! brew list tesseract &>/dev/null 2>&1; then
    echo "  Installing tesseract (OCR)..."
    brew install tesseract -q 2>/dev/null || true
fi
echo ""

# 6. Download the AI model
echo "═══════════════════════════════════════════════════════"
echo "  Downloading AI model (~4 GB — this is the slow part)"
echo "═══════════════════════════════════════════════════════"
echo ""
"$PY" -c "
from huggingface_hub import snapshot_download
print('  Downloading mlx-community/gemma-4-e4b-it-4bit ...')
snapshot_download(repo_id='mlx-community/gemma-4-e4b-it-4bit')
print('  Model downloaded.')
"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "✅  Setup complete! Opening Graceful..."
echo "═══════════════════════════════════════════════════════"
echo ""

# Launch server + browser
source "$VENV/bin/activate"
nohup python3 "$ROOT/app.py" >> "$LOG" 2>&1 &
i=0; while [ \$i -lt 60 ]; do
    nc -z 127.0.0.1 7860 2>/dev/null && open http://localhost:7860 && exit
    sleep 2; i=\$((i+1))
done
AUTOSETUP
    chmod +x "$SETUP_SCRIPT"

    osascript -e "tell application \"Terminal\"
        activate
        do script \"bash '$SETUP_SCRIPT'\"
    end tell"
    exit 0
fi

# ── Normal launch ─────────────────────────────────────────────────
# Use tkinter GUI launcher if available
if "$PY" -c "import tkinter" 2>/dev/null && [ -f "$ROOT/launch.py" ]; then
    exec "$PY" "$ROOT/launch.py"
fi

# Fallback: headless server + open browser
source "$VENV/bin/activate"
nohup python3 "$ROOT/app.py" >> "$LOG" 2>&1 &
SRV=$!
( i=0; while [ $i -lt 60 ]; do
    nc -z 127.0.0.1 7860 2>/dev/null && open http://localhost:7860 && exit
    sleep 2; i=$((i+1))
done ) &
wait $SRV
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
