#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  COUPLED MANIFOLD — Universal Setup
#  Run once: bash setup.sh
#  Every launch after: double-click Graceful.app
# ═══════════════════════════════════════════════════════════════

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  COUPLED MANIFOLD — Setup"
echo "  Gemma 3 12B  ·  MLX  ·  Hessian trace  ·  online learning"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── 1. Python check ─────────────────────────────────────────────
if ! command -v python3 &> /dev/null; then
    echo "❌  Python 3 not found."
    echo "    Install via Homebrew: brew install python@3.11"
    echo "    If you don't have Homebrew, install it first from brew.sh,"
    echo "    or download Python from https://python.org and re-run setup.sh."
    read -p "    Press Enter to exit." _
    exit 1
fi

# Check Python version (MLX requires 3.9+, some deps require 3.10)
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    echo "❌  Python $PYTHON_VERSION found, but Python 3.9 or later is required."
    echo "    Install a newer Python: brew install python@3.11"
    echo "    Or download from https://python.org and re-run setup.sh."
    read -p "    Press Enter to exit." _
    exit 1
fi

echo "✅  Python $PYTHON_VERSION OK"

# ── 2. Apple Silicon check ───────────────────────────────────────
ARCH=$(uname -m)
if [ "$ARCH" != "arm64" ]; then
    echo "⚠️   MLX requires Apple Silicon (M1/M2/M3/M4)."
    echo "    Detected: $ARCH"
    echo "    This app will not run on Intel Macs."
    read -p "    Continue anyway? (y/n): " CONT
    [[ "$CONT" =~ ^[Yy]$ ]] || exit 1
fi
echo "✅  Apple Silicon ($ARCH) confirmed."

# ── 3. Homebrew check + dependencies (optional) ────────────────
echo ""
echo "  Checking Homebrew dependencies..."
if ! command -v brew &> /dev/null; then
    echo ""
    echo "  Homebrew is not installed."
    echo "  It's needed only for OCR on image PDFs — the app works without it."
    read -p "  Install Homebrew? (y/N): " BREW_ANS
    if [[ "$BREW_ANS" =~ ^[Yy]$ ]]; then
        echo "  Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        if [ -f "/opt/homebrew/bin/brew" ]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
        fi
        echo "✅  Homebrew installed."
    else
        echo "✅  Homebrew skipped — OCR on image PDFs won't be available."
    fi
else
    echo "✅  Homebrew found."
fi

if command -v brew &> /dev/null; then
    for pkg in tesseract; do
        if brew list "$pkg" &> /dev/null; then
            echo "✅  $pkg already installed."
        else
            echo "  Installing $pkg..."
            brew install "$pkg" -q
            echo "✅  $pkg installed."
        fi
    done
fi

# ── 4. Virtual environment ───────────────────────────────────────
echo ""
if [ ! -d "$DIR/graceful_env" ]; then
    echo "📦  Creating virtual environment..."
    python3 -m venv "$DIR/graceful_env"
    echo "✅  Environment created."
else
    echo "✅  Virtual environment already exists."
fi

PY="$DIR/graceful_env/bin/python3"

# Bootstrap pip inside the venv (some macOS Python installs omit it)
"$PY" -m ensurepip --upgrade 2>/dev/null || true
if [ ! -f "$DIR/graceful_env/bin/pip" ]; then
    echo "❌  pip could not be installed in the virtual environment."
    echo "    Please install Python from https://python.org and re-run."
    read -p "    Press Enter to exit." _
    exit 1
fi

source "$DIR/graceful_env/bin/activate"

# ── 5. Python dependencies ───────────────────────────────────────
echo ""
echo "📦  Installing Python dependencies..."
pip install --upgrade pip -q

echo "  MLX framework (Apple Silicon native)..."
pip install mlx mlx-vlm -q

echo "  Data science: numpy, pandas, matplotlib, scipy..."
pip install numpy pandas matplotlib scipy -q

echo "  Search & networking: httpx, duckduckgo-search..."
pip install httpx duckduckgo-search -q

echo "  Document parsing: pymupdf, pdfplumber, docx, pptx..."
pip install pymupdf pdfplumber python-docx python-pptx openpyxl -q

echo "  Embeddings & memory: sentence-transformers..."
pip install sentence-transformers -q

echo "  Web server: fastapi, uvicorn, python-multipart..."
pip install fastapi uvicorn python-multipart -q

echo "  Media: pillow, opencv-python, soundfile, miniaudio..."
pip install pillow opencv-python soundfile miniaudio -q

echo "  Desktop app: pywebview..."
pip install pywebview -q

echo "  Extras: beautifulsoup4, huggingface_hub, sympy, scikit-learn..."
pip install beautifulsoup4 huggingface_hub sympy seaborn statsmodels scikit-learn ebooklib pytesseract -q

echo "✅  All dependencies installed."

# ── 6. Hugging Face token ────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Hugging Face Token  (OPTIONAL)"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  The model used by this app (mlx-community/gemma-3-12b-it-4bit)"
echo "  is publicly available — no token is required."
echo ""
if [ -t 0 ]; then
    echo "  If you have a HuggingFace token and want to save it for"
    echo "  other models, paste it below. Otherwise, just press Enter."
    echo ""
    read -p "  HF token (press Enter to skip — not required): " HF_TOKEN
    if [ -n "$HF_TOKEN" ]; then
        "$PY" -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
        echo "✅  Token saved."
    else
        echo "✅  Skipped — no token needed for Gemma 3 12B."
    fi
else
    echo "✅  Skipped (non-interactive) — no token needed for Gemma 3 12B."
fi

# ── 7. API Keys ──────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  API Keys (all optional — press Enter to skip any)"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  Brave Search (brave.com/search/api — free 2000/mo)"
read -p "  Brave API key: " BRAVE_KEY

echo "  YouTube Data API (console.cloud.google.com — free 10k/day)"
read -p "  YouTube API key: " YOUTUBE_KEY

echo "  Google Custom Search (programmablesearchengine.google.com)"
read -p "  Google Search API key: " GOOGLE_KEY
read -p "  Google Custom Search CX ID: " GOOGLE_CX

echo "  Email for Unpaywall free PDF finder (any email works)"
read -p "  Email: " UNPAYWALL_EMAIL

mkdir -p "$DIR/manifold_data/"{static,corpus,checkpoints,sessions,logs,identity,backups,personalities,consolidation}

# Extract UI from app.py into static/index.html (app.py loads it at runtime)
if [ ! -f "$DIR/manifold_data/static/index.html" ]; then
    "$PY" -c "
import re
with open('$DIR/app.py') as f:
    src = f.read()
# If HTML is still inline (old install), extract it
m = re.search(r'MANIFEST_HTML = r\"\"\"(.*?)^\"\"\"', src, re.DOTALL | re.MULTILINE)
if m:
    with open('$DIR/manifold_data/static/index.html', 'w') as f:
        f.write(m.group(1))
    print('  UI extracted to static/index.html')
elif open('$DIR/manifold_data/static/index.html').read():
    print('  UI already extracted')
" 2>/dev/null || true
fi
"$PY" -c "
import json, os
path = '$DIR/manifold_data/keys.json'
keys = {}
if os.path.exists(path):
    try: keys = json.load(open(path))
    except: pass
if '$BRAVE_KEY':         keys['brave']            = '$BRAVE_KEY'
if '$YOUTUBE_KEY':       keys['youtube']          = '$YOUTUBE_KEY'
if '$GOOGLE_KEY':        keys['google_search']    = '$GOOGLE_KEY'
if '$GOOGLE_CX':         keys['google_cx']        = '$GOOGLE_CX'
if '$UNPAYWALL_EMAIL':   keys['unpaywall_email']  = '$UNPAYWALL_EMAIL'
json.dump(keys, open(path,'w'), indent=2)
print('  Keys saved.')
"
echo "✅  API keys saved."

# ── 8. Download model ────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Downloading Model (~7 GB, one-time only)"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  Model: mlx-community/gemma-3-12b-it-4bit"
echo "  MLX-native 4-bit quantization  ·  128K context  ·  vision + text"
echo ""

"$PY" -c "
from huggingface_hub import snapshot_download
print('  Downloading mlx-community/gemma-3-12b-it-4bit ...')
snapshot_download(repo_id='mlx-community/gemma-3-12b-it-4bit')
print('  Done.')
"
echo "✅  Model downloaded."

# ── 9. Build Graceful.app ────────────────────────────────────────
echo ""
echo "  Building Graceful.app..."

APP="$DIR/Graceful.app"
MACOS="$APP/Contents/MacOS"
RES="$APP/Contents/Resources"
mkdir -p "$MACOS" "$RES"

cat << 'LAUNCHER' > "$MACOS/Graceful"
#!/bin/bash
DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$DIR"
xattr -cr "$DIR/Graceful.app" 2>/dev/null || true
export MANIFOLD_DIR="$DIR"
VENV="$DIR/graceful_env"
# Prefer venv python, fall back to system python3
if [ -f "$VENV/bin/python3" ]; then
    exec "$VENV/bin/python3" "$DIR/launch.py"
elif python3 -c "import tkinter" 2>/dev/null; then
    exec python3 "$DIR/launch.py"
else
    # No tkinter — fall back to headless server
    if [ -d "$VENV" ]; then
        source "$VENV/bin/activate"
        nohup python3 "$DIR/app.py" >> "$DIR/manifold_data/logs/launch.log" 2>&1 &
        ( i=0; while [ $i -lt 60 ]; do nc -z 127.0.0.1 7860 2>/dev/null && open http://localhost:7860 && exit; sleep 2; i=$((i+1)); done ) &
    else
        osascript -e "tell app \"Terminal\" to do script \"cd \\\"$DIR\\\" && bash setup.sh\""
        osascript -e 'tell app "Terminal" to activate'
    fi
fi
LAUNCHER
chmod +x "$MACOS/Graceful"

cat << 'PLIST' > "$APP/Contents/Info.plist"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>        <string>Coupled Manifold</string>
    <key>CFBundleDisplayName</key> <string>Coupled Manifold</string>
    <key>CFBundleIdentifier</key>  <string>com.coupledmanifold.graceful</string>
    <key>CFBundleVersion</key>     <string>2.0</string>
    <key>CFBundleExecutable</key>  <string>Graceful</string>
    <key>CFBundlePackageType</key> <string>APPL</string>
    <key>CFBundleIconFile</key>    <string>AppIcon</string>
</dict>
</plist>
PLIST

# Icon
ICON_SRC=""
for candidate in "$DIR/icon.png" "$DIR/slash_icon_v4.png" "$DIR/app_icon.png"; do
    [ -f "$candidate" ] && ICON_SRC="$candidate" && break
done

if [ ! -z "$ICON_SRC" ] && command -v sips &> /dev/null && command -v iconutil &> /dev/null; then
    ICONSET="$DIR/icon.iconset"
    mkdir -p "$ICONSET"
    for size in 16 32 128 256 512; do
        sips -z $size $size "$ICON_SRC" --out "$ICONSET/icon_${size}x${size}.png" &>/dev/null
        dbl=$((size*2))
        sips -z $dbl $dbl "$ICON_SRC" --out "$ICONSET/icon_${size}x${size}@2x.png" &>/dev/null
    done
    iconutil -c icns "$ICONSET" -o "$RES/AppIcon.icns" 2>/dev/null && echo "✅  Icon set." || true
    rm -rf "$ICONSET"
fi

xattr -cr "$APP" 2>/dev/null || true
echo "✅  Graceful.app built."

# ── Done ─────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✅  Setup Complete!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  Launch:  double-click Graceful.app"
echo "  Or:      source graceful_env/bin/activate && python3 app.py"
echo "  Data:    $DIR/manifold_data/"
echo ""
read -p "  Launch now? (Y/n): " LAUNCH
LAUNCH="${LAUNCH:-Y}"   # default to Y if user just presses Enter
if [[ "$LAUNCH" =~ ^[Yy]$ ]]; then
    source "$DIR/graceful_env/bin/activate"
    python3 "$DIR/app.py"
fi
