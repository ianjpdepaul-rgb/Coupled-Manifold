"""
COUPLED MANIFOLD — GUI Launcher & Installer
Handles first-run setup wizard and normal app launching.
Uses only stdlib tkinter so it works before the venv exists.
"""
import os, sys, json, time, threading, subprocess, webbrowser, socket
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

# ── Paths ──────────────────────────────────────────────────────────────────
DIR      = os.environ.get("MANIFOLD_DIR", os.path.dirname(os.path.abspath(__file__)))
VENV     = os.path.join(DIR, "graceful_env")
PY       = os.path.join(VENV, "bin", "python3")
APP_PY   = os.path.join(DIR, "app.py")
DATA_DIR = os.path.join(DIR, "manifold_data")
KEYS_PATH= os.path.join(DATA_DIR, "keys.json")

# ── Theme ──────────────────────────────────────────────────────────────────
BG    = "#0a0a0a"
BG2   = "#111111"
BG3   = "#1c1c1c"
FG    = "#e8e8e8"
FG2   = "#888888"
GREEN = "#7ee8a2"
RED   = "#f47174"
FONT  = ("SF Pro Display", 13)
MONO  = ("SF Mono", 11)


def is_setup_done():
    sentinel = os.path.join(DATA_DIR, ".setup_complete")
    return os.path.isdir(VENV) and os.path.isfile(PY) and os.path.isfile(sentinel)


def get_ram_gb() -> float:
    """Detect physical RAM on macOS via sysctl."""
    try:
        r = subprocess.run(["sysctl", "-n", "hw.memsize"],
                           capture_output=True, text=True, timeout=3)
        return int(r.stdout.strip()) / (1024 ** 3)
    except Exception:
        return 0.0


def get_free_gb(path: str) -> float:
    """Free disk space at the given path."""
    import shutil
    return shutil.disk_usage(path).free / (1024 ** 3)


def mem_summary() -> str:
    """One-line corpus + session count for the launcher."""
    try:
        chunks_f = os.path.join(DATA_DIR, "corpus", "chunks.json")
        n_chunks = len(json.load(open(chunks_f))) if os.path.exists(chunks_f) else 0
        sess_dir = os.path.join(DATA_DIR, "sessions")
        n_sess   = len([f for f in os.listdir(sess_dir)
                        if f.startswith("session_") and f.endswith(".json")
                        and os.path.isfile(os.path.join(sess_dir, f))]) \
                   if os.path.isdir(sess_dir) else 0
        return f"{n_chunks} corpus chunks  ·  {n_sess} sessions"
    except Exception:
        return ""


def _find_server_pid(port=7860):
    """Return PID of the process listening on the given port, or None."""
    try:
        r = subprocess.run(["lsof", "-ti", f"tcp:{port}"],
                           capture_output=True, text=True, timeout=3)
        for pid in r.stdout.strip().split():
            if pid.isdigit():
                return int(pid)
    except Exception:
        pass
    return None


def _kill_port_server(port=7860):
    """Kill whatever is listening on the port. Returns True if something was killed."""
    import signal as _sig
    pid = _find_server_pid(port)
    if pid is None:
        return False
    try:
        os.kill(pid, _sig.SIGTERM)
        # Wait up to 4s for clean exit
        for _ in range(20):
            time.sleep(0.2)
            try:
                os.kill(pid, 0)  # check if still alive
            except ProcessLookupError:
                return True
        # Still alive — force kill
        os.kill(pid, _sig.SIGKILL)
        return True
    except ProcessLookupError:
        return True
    except Exception:
        return False


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Coupled Manifold")
        self.configure(bg=BG)
        self.resizable(True, True)
        self._server_proc = None

        # ttk style — darken progressbar
        s = ttk.Style(self)
        s.theme_use("default")
        s.configure("green.Horizontal.TProgressbar",
                    troughcolor=BG3, background=GREEN, thickness=4)
        s.configure("TScrollbar", background=BG3, troughcolor=BG2,
                    arrowcolor=FG2, bordercolor=BG)

        self._load_icon()

        if is_setup_done():
            self._build_launcher()
        else:
            self._build_setup()

        self.update_idletasks()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        ww, wh = self.winfo_width(), self.winfo_height()
        self.geometry(f"+{(sw-ww)//2}+{(sh-wh)//2}")
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Command-q>", lambda e: self._on_close())
        self.bind("<Command-w>", lambda e: self._on_close())
        # macOS application menu Quit item
        try:
            self.createcommand("tk::mac::Quit", self._on_close)
        except Exception:
            pass

    # ── Icon ──────────────────────────────────────────────────────────────
    def _load_icon(self):
        """Load icon.png — sets window icon and returns a PhotoImage for display."""
        self._icon_photo = None
        path = os.path.join(DIR, "icon.png")
        if not os.path.exists(path):
            return
        try:
            from PIL import Image, ImageTk
            img = Image.open(path).convert("RGBA")
            sm  = img.resize((64, 64),  Image.LANCZOS)
            self._icon_photo   = ImageTk.PhotoImage(sm)
            self._icon_photo_lg = ImageTk.PhotoImage(img.resize((120, 120), Image.LANCZOS))
            self.iconphoto(True, self._icon_photo)
        except Exception:
            self._icon_photo   = None
            self._icon_photo_lg = None

    def _icon_label(self, parent, large=False):
        photo = getattr(self, "_icon_photo_lg" if large else "_icon_photo", None)
        if photo:
            lbl = tk.Label(parent, image=photo, bg=BG)
            lbl.pack(pady=(20, 6))
        else:
            tk.Label(parent, text="🧠", font=("SF Pro Display", 48), bg=BG).pack(pady=(20, 6))

    # ══════════════════════════════════════════════════════════════════════
    # LAUNCHER  (normal run — venv exists)
    # ══════════════════════════════════════════════════════════════════════
    def _build_launcher(self):
        self.geometry("520x500")

        self._icon_label(self, large=True)
        tk.Label(self, text="Coupled Manifold",
                 font=("SF Pro Display", 22, "bold"), bg=BG, fg=FG).pack()
        tk.Label(self, text="Gemma 4 E4B  ·  MLX  ·  Hessian trace  ·  online learning",
                 font=("SF Pro Display", 11), bg=BG, fg=FG2).pack(pady=(2, 14))

        tk.Frame(self, bg=BG3, height=1).pack(fill=tk.X, padx=40, pady=(0, 14))

        # Check if server is already running from a previous session
        self._orphan_pid = _find_server_pid()
        _init_status = "Ready."
        _init_fg = FG2
        if self._orphan_pid:
            _init_status = f"Server already running (PID {self._orphan_pid})"
            _init_fg = GREEN
        self._status = tk.Label(self, text=_init_status, font=FONT, bg=BG, fg=_init_fg)
        self._status.pack()

        self._pbar = ttk.Progressbar(self, style="green.Horizontal.TProgressbar",
                                     mode="indeterminate", length=380)
        self._pbar.pack(pady=(8, 6))

        # Embedded console
        self._console = scrolledtext.ScrolledText(
            self, height=9, font=MONO,
            bg="#050505", fg="#7dd9a0", insertbackground=GREEN,
            relief=tk.FLAT, borderwidth=0, state=tk.DISABLED,
        )
        self._console.pack(fill=tk.X, padx=30, pady=(0, 12))

        btn_row = tk.Frame(self, bg=BG)
        btn_row.pack(pady=(0, 16))

        self._launch_btn = tk.Button(
            btn_row, text="Launch  →",
            font=("SF Pro Display", 13, "bold"),
            bg=GREEN, fg="#000", relief=tk.FLAT,
            padx=26, pady=9, cursor="hand2",
            command=self._do_launch,
        )
        self._launch_btn.pack(side=tk.LEFT, padx=(0, 10))

        self._stop_btn = tk.Button(
            btn_row, text="Stop",
            font=("SF Pro Display", 12),
            bg=BG3, fg=FG2, relief=tk.FLAT,
            padx=16, pady=9, cursor="hand2",
            state=tk.NORMAL if self._orphan_pid else tk.DISABLED,
            command=self._do_stop,
        )
        self._stop_btn.pack(side=tk.LEFT)

        # If server is already running, switch Launch to Open Browser
        if self._orphan_pid:
            self._launch_btn.config(
                text="Open Browser",
                command=lambda: webbrowser.open("http://localhost:7860"),
            )

        # Utility row — data folder, settings, memory stats
        tk.Frame(self, bg=BG3, height=1).pack(fill=tk.X, padx=40, pady=(12, 4))
        util = tk.Frame(self, bg=BG)
        util.pack(pady=(0, 10))
        tk.Button(util, text="📁  Data",
                  font=("SF Pro Display", 11), bg=BG3, fg=FG2,
                  relief=tk.FLAT, padx=14, pady=5, cursor="hand2",
                  command=lambda: subprocess.run(["open", DATA_DIR])).pack(side=tk.LEFT, padx=4)
        tk.Button(util, text="⚙  Settings",
                  font=("SF Pro Display", 11), bg=BG3, fg=FG2,
                  relief=tk.FLAT, padx=14, pady=5, cursor="hand2",
                  command=self._open_settings).pack(side=tk.LEFT, padx=4)
        self._mem_lbl = tk.Label(util, text=mem_summary(),
                                 font=("SF Pro Display", 10), bg=BG, fg=FG2)
        self._mem_lbl.pack(side=tk.LEFT, padx=10)

    def _cwrite(self, text):
        """Thread-safe write to console widget."""
        def _do():
            self._console.config(state=tk.NORMAL)
            self._console.insert(tk.END, text)
            self._console.see(tk.END)
            self._console.config(state=tk.DISABLED)
        self.after(0, _do)

    def _do_launch(self):
        self._launch_btn.config(state=tk.DISABLED, text="Starting…")
        self._pbar.start(10)
        self._status.config(
            text="Starting… (first run may download ~4GB of model weights)",
            fg=FG2,
        )
        self._dl_done = threading.Event()   # signals download monitor to stop

        def _dl_monitor():
            """Watch HuggingFace cache and drive the progress bar."""
            import glob
            hf_base = os.path.expanduser(
                "~/.cache/huggingface/hub/models--mlx-community--gemma-4-e4b-it-4bit"
            )
            # Approximate total: ~4.2 GB
            total_bytes = 4_200_000_000

            blobs_dir = f"{hf_base}/blobs"

            # Check if model is already fully downloaded before doing anything
            try:
                downloaded = sum(
                    os.path.getsize(os.path.join(blobs_dir, f))
                    for f in os.listdir(blobs_dir)
                    if not f.endswith(".incomplete")
                )
            except Exception:
                downloaded = 0

            already_done = downloaded >= total_bytes

            if already_done:
                # Model already cached — just update status text, keep bar indeterminate
                def _loading():
                    self._status.config(text="Loading model into memory…", fg=FG2)
                self.after(0, _loading)
                return

            # Model is downloading — switch bar to determinate mode
            def _det():
                self._pbar.config(mode="determinate", maximum=100, value=0)
            self.after(0, _det)

            while not self._dl_done.is_set():
                time.sleep(2)
                if self._dl_done.is_set():
                    break
                try:
                    downloaded = sum(
                        os.path.getsize(os.path.join(blobs_dir, f))
                        for f in os.listdir(blobs_dir)
                        if not f.endswith(".incomplete")
                    )
                    pct = min(99, int(downloaded / total_bytes * 100))
                    def _upd(p=pct, d=downloaded):
                        self._pbar["value"] = p
                        self._status.config(
                            text=f"Downloading model  {d/1e9:.1f} / {total_bytes/1e9:.1f} GB  ({p}%)",
                            fg=FG2,
                        )
                    self.after(0, _upd)
                    if downloaded >= total_bytes:
                        def _loading():
                            self._pbar.config(mode="indeterminate")
                            self._pbar.start(10)
                            self._status.config(text="Loading model into memory…", fg=FG2)
                        self.after(0, _loading)
                        break
                except Exception:
                    pass

        threading.Thread(target=_dl_monitor, daemon=True).start()

        def _run():
            self._server_proc = subprocess.Popen(
                [PY, APP_PY],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=DIR,
            )

            # Stream output to console
            def _stream():
                for line in self._server_proc.stdout:
                    if line.strip():
                        self._cwrite(line)
            threading.Thread(target=_stream, daemon=True).start()

            # Poll until port 7860 is open — up to 30 min for slow connections
            ready = False
            for _ in range(900):   # 900 × 2s = 30 minutes
                time.sleep(2)
                try:
                    c = socket.create_connection(("127.0.0.1", 7860), timeout=1)
                    c.close()
                    ready = True
                    break
                except OSError:
                    pass

            self._dl_done.set()   # stop download monitor

            def _up():
                self._pbar.stop()
                if ready:
                    self._status.config(text="Running at localhost:7860", fg=GREEN)
                    self._launch_btn.config(
                        text="Open Browser", state=tk.NORMAL,
                        command=lambda: webbrowser.open("http://localhost:7860"),
                    )
                    self._stop_btn.config(state=tk.NORMAL)
                    webbrowser.open("http://localhost:7860")
                else:
                    self._status.config(text="Server didn't start — check console", fg=RED)
                    self._launch_btn.config(text="Retry", state=tk.NORMAL,
                                            command=self._do_launch)
            self.after(0, _up)

        threading.Thread(target=_run, daemon=True).start()

    def _do_stop(self):
        if self._server_proc:
            try:
                self._server_proc.terminate()   # SIGTERM → triggers _shutdown_save
                self._server_proc.wait(timeout=4)
            except Exception:
                try:
                    self._server_proc.kill()
                except Exception:
                    pass
            self._server_proc = None
        else:
            # No subprocess handle — kill whatever's on the port (orphaned server)
            _kill_port_server()
        self._orphan_pid = None
        self._pbar.stop()
        self._stop_btn.config(state=tk.DISABLED)
        self._launch_btn.config(text="Launch  →", state=tk.NORMAL,
                                command=self._do_launch)
        self._status.config(text="Stopped.", fg=FG2)

    # ══════════════════════════════════════════════════════════════════════
    # SETUP WIZARD  (first run)
    # ══════════════════════════════════════════════════════════════════════
    STEP_NAMES = ["Install", "Keys", "Model", "Done"]

    def _build_setup(self):
        self.geometry("720x800")

        # Header
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill=tk.X)
        self._icon_label(hdr, large=False)
        tk.Label(hdr, text="Coupled Manifold",
                 font=("SF Pro Display", 20, "bold"), bg=BG, fg=FG).pack()
        tk.Label(hdr, text="First-run setup  —  takes about 5 minutes",
                 font=("SF Pro Display", 11), bg=BG, fg=FG2).pack(pady=(2, 10))

        # Step strip
        self._step_strip = tk.Frame(self, bg=BG2, pady=6)
        self._step_strip.pack(fill=tk.X)
        self._step_lbls = []
        for name in self.STEP_NAMES:
            lbl = tk.Label(self._step_strip, text=f"○  {name}",
                           font=MONO, bg=BG2, fg=FG2, padx=18)
            lbl.pack(side=tk.LEFT)
            self._step_lbls.append(lbl)

        tk.Frame(self, bg=BG3, height=1).pack(fill=tk.X)

        # Swappable content area
        self._area = tk.Frame(self, bg=BG)
        self._area.pack(fill=tk.BOTH, expand=True, padx=32, pady=16)

        self._goto(0)

    def _set_step(self, active):
        for i, lbl in enumerate(self._step_lbls):
            name = self.STEP_NAMES[i]
            if i < active:
                lbl.config(fg=GREEN, text=f"✓  {name}")
            elif i == active:
                lbl.config(fg=GREEN, text=f"●  {name}")
            else:
                lbl.config(fg=FG2,   text=f"○  {name}")

    def _clear(self):
        for w in self._area.winfo_children():
            w.destroy()

    def _goto(self, step):
        self._clear()
        self._set_step(step)
        [self._step_install, self._step_keys,
         self._step_model,   self._step_done][step]()

    # ── Step 0: Install ───────────────────────────────────────────────────
    def _step_install(self):
        tk.Label(self._area, text="Installing Python dependencies",
                 font=("SF Pro Display", 15, "bold"), bg=BG, fg=FG).pack(anchor=tk.W)
        tk.Label(self._area,
                 text="Installs MLX, mlx-vlm, FastAPI, and all required packages "
                      "into an isolated virtual environment.",
                 font=("SF Pro Display", 11), bg=BG, fg=FG2,
                 wraplength=640, justify=tk.LEFT).pack(anchor=tk.W, pady=(2, 10))

        self._install_console = scrolledtext.ScrolledText(
            self._area, height=18, font=MONO,
            bg="#050505", fg="#7dd9a0", relief=tk.FLAT, borderwidth=0,
            state=tk.DISABLED,
        )
        self._install_console.pack(fill=tk.BOTH, expand=True)

        self._install_pbar = ttk.Progressbar(
            self._area, style="green.Horizontal.TProgressbar",
            mode="indeterminate")
        self._install_pbar.pack(fill=tk.X, pady=(8, 4))

        self._install_lbl = tk.Label(self._area, text="Press Install to begin.",
                                     font=MONO, bg=BG, fg=FG2)
        self._install_lbl.pack(anchor=tk.W)

        self._install_btn = tk.Button(
            self._area, text="Install  →",
            font=("SF Pro Display", 13, "bold"),
            bg=GREEN, fg="#000", relief=tk.FLAT, padx=22, pady=8,
            cursor="hand2", command=self._do_install,
        )
        self._install_btn.pack(anchor=tk.E, pady=(8, 0))

    def _iwrite(self, text):
        def _do():
            self._install_console.config(state=tk.NORMAL)
            self._install_console.insert(tk.END, text)
            self._install_console.see(tk.END)
            self._install_console.config(state=tk.DISABLED)
        self.after(0, _do)

    def _do_install(self):
        self._install_btn.config(state=tk.DISABLED, text="Installing…")
        self._install_pbar.start(10)

        req_file = os.path.join(DIR, "requirements.txt")
        steps = [
            ("Creating virtual environment…",
             f"'{sys.executable}' -m venv '{VENV}'"),
            ("Bootstrapping pip…",
             f"'{PY}' -m ensurepip --upgrade"),
            ("Upgrading pip…",
             f"'{PY}' -m pip install --upgrade pip -q"),
            ("Installing dependencies…",
             f"'{PY}' -m pip install -r '{req_file}' -q"),
        ]

        def _run():
            ok = True
            for msg, cmd in steps:
                self._iwrite(f"\n▸  {msg}\n")
                self.after(0, lambda m=msg: self._install_lbl.config(text=m))
                proc = subprocess.Popen(
                    cmd, shell=True,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, cwd=DIR,
                )
                for line in proc.stdout:
                    if line.strip():
                        self._iwrite(line)
                proc.wait()
                if proc.returncode != 0:
                    self._iwrite(f"\n✗  Failed (exit {proc.returncode})\n")
                    ok = False
                    break
                self._iwrite("✓  Done\n")

            def _done():
                self._install_pbar.stop()
                if ok:
                    self._install_lbl.config(text="✓  Installation complete.", fg=GREEN)
                    self._install_btn.config(
                        text="Next: API Keys  →", state=tk.NORMAL,
                        command=lambda: self._goto(1),
                    )
                else:
                    self._install_lbl.config(text="✗  Failed — see output above.", fg=RED)
                    self._install_btn.config(text="Retry", state=tk.NORMAL,
                                             command=self._do_install)
            self.after(0, _done)

        threading.Thread(target=_run, daemon=True).start()

    # ── Step 1: Keys ──────────────────────────────────────────────────────
    def _step_keys(self):
        tk.Label(self._area, text="API Keys",
                 font=("SF Pro Display", 15, "bold"), bg=BG, fg=FG).pack(anchor=tk.W)
        tk.Label(self._area,
                 text="All optional. Without keys, search falls back to DuckDuckGo (always free).",
                 font=("SF Pro Display", 11), bg=BG, fg=FG2,
                 wraplength=640, justify=tk.LEFT).pack(anchor=tk.W, pady=(2, 14))

        FIELDS = [
            ("HuggingFace Token",     "hf_token",        "hf_...  (huggingface.co/settings/tokens)"),
            ("Brave Search API Key",  "brave",            "brave.com/search/api  —  free 2k/mo"),
            ("YouTube API Key",       "youtube",          "console.cloud.google.com  —  free 10k/day"),
            ("Google Search API Key", "google_search",    "programmablesearchengine.google.com"),
            ("Google CX ID",          "google_cx",        "Custom Search Engine ID"),
            ("Unpaywall Email",       "unpaywall_email",  "any email for free PDF links"),
        ]

        self._kvars = {}
        existing = {}
        if os.path.exists(KEYS_PATH):
            try:
                existing = json.load(open(KEYS_PATH))
            except Exception:
                pass

        for label, key, hint in FIELDS:
            row = tk.Frame(self._area, bg=BG)
            row.pack(fill=tk.X, pady=4)
            tk.Label(row, text=label, font=("SF Pro Display", 11, "bold"),
                     bg=BG, fg=FG, width=22, anchor=tk.W).pack(side=tk.LEFT)
            var = tk.StringVar(value=existing.get(key, ""))
            ent = tk.Entry(row, textvariable=var, font=MONO,
                           bg=BG3, fg=FG, insertbackground=GREEN,
                           relief=tk.FLAT, bd=6, width=34)
            ent.pack(side=tk.LEFT, padx=(6, 0))
            tk.Label(row, text=hint, font=("SF Pro Display", 9),
                     bg=BG, fg=FG2).pack(side=tk.LEFT, padx=(8, 0))
            self._kvars[key] = var

        brow = tk.Frame(self._area, bg=BG)
        brow.pack(fill=tk.X, pady=(18, 0))
        tk.Button(brow, text="Skip  →", font=("SF Pro Display", 12),
                  bg=BG3, fg=FG2, relief=tk.FLAT, padx=16, pady=8,
                  cursor="hand2", command=lambda: self._goto(2)).pack(side=tk.RIGHT, padx=(8, 0))
        tk.Button(brow, text="Save & Continue  →",
                  font=("SF Pro Display", 13, "bold"),
                  bg=GREEN, fg="#000", relief=tk.FLAT, padx=22, pady=8,
                  cursor="hand2", command=self._save_keys).pack(side=tk.RIGHT)

    def _save_keys(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        keys = {}
        if os.path.exists(KEYS_PATH):
            try:
                keys = json.load(open(KEYS_PATH))
            except Exception:
                pass
        hf_token = None
        for k, var in self._kvars.items():
            v = var.get().strip()
            if not v:
                continue
            if k == "hf_token":
                hf_token = v
            else:
                keys[k] = v
        json.dump(keys, open(KEYS_PATH, "w"), indent=2)
        if hf_token:
            threading.Thread(
                target=lambda: subprocess.run(
                    [PY, "-c",
                     f"from huggingface_hub import login; login(token='{hf_token}')"],
                    cwd=DIR,
                ),
                daemon=True,
            ).start()
        self._goto(2)

    # ── Step 2: Model ─────────────────────────────────────────────────────
    def _step_model(self):
        tk.Label(self._area, text="Download model",
                 font=("SF Pro Display", 15, "bold"), bg=BG, fg=FG).pack(anchor=tk.W)

        ram_gb = get_ram_gb()
        ram_note = (f"Detected {ram_gb:.0f} GB RAM  →  Gemma 4 E4B 4-bit (~4 GB) — fits comfortably"
                    if ram_gb >= 8 else
                    f"Detected {ram_gb:.0f} GB RAM  →  may be tight, but should work")
        ram_color = GREEN if ram_gb >= 8 else "#ffd740"
        tk.Label(self._area,
                 text="Gemma 4 E4B 4-bit  ·  ~4 GB download  ·  MLX native  ·  128K context  ·  vision + audio",
                 font=("SF Pro Display", 11), bg=BG, fg=FG2,
                 wraplength=640).pack(anchor=tk.W, pady=(2, 4))
        tk.Label(self._area, text=ram_note, font=("SF Pro Display", 10),
                 bg=BG, fg=ram_color).pack(anchor=tk.W, pady=(0, 10))

        tk.Frame(self._area, bg=BG3, height=1).pack(fill=tk.X, pady=12)

        self._model_console = scrolledtext.ScrolledText(
            self._area, height=11, font=MONO,
            bg="#050505", fg="#7dd9a0", relief=tk.FLAT, borderwidth=0,
            state=tk.DISABLED,
        )
        self._model_console.pack(fill=tk.BOTH, expand=True)

        self._model_pbar = ttk.Progressbar(
            self._area, style="green.Horizontal.TProgressbar",
            mode="indeterminate")
        self._model_pbar.pack(fill=tk.X, pady=(8, 4))

        self._model_lbl = tk.Label(self._area, text="",
                                   font=MONO, bg=BG, fg=FG2)
        self._model_lbl.pack(anchor=tk.W)

        self._model_btn = tk.Button(
            self._area, text="Download & Continue  →",
            font=("SF Pro Display", 13, "bold"),
            bg=GREEN, fg="#000", relief=tk.FLAT, padx=22, pady=8,
            cursor="hand2", command=self._do_download,
        )
        self._model_btn.pack(anchor=tk.E, pady=(8, 0))

    def _do_download(self):
        model_id = "mlx-community/gemma-4-e4b-it-4bit"
        needed_gb = 4.5
        free_gb   = get_free_gb(os.path.expanduser("~"))
        if free_gb < needed_gb:
            messagebox.showerror(
                "Not Enough Disk Space",
                f"Need ~{needed_gb:.0f} GB free to download the model.\n"
                f"Only {free_gb:.1f} GB available.\n\nFree up space and try again.",
            )
            return

        self._model_btn.config(state=tk.DISABLED, text="Downloading…")
        self._model_pbar.start(10)
        self._model_lbl.config(text=f"Downloading {model_id}…")

        def _mwrite(text):
            def _do():
                self._model_console.config(state=tk.NORMAL)
                self._model_console.insert(tk.END, text)
                self._model_console.see(tk.END)
                self._model_console.config(state=tk.DISABLED)
            self.after(0, _do)

        def _run():
            script = (
                "from huggingface_hub import snapshot_download;"
                f"print('Downloading {model_id} (~4 GB, be patient)...');"
                f"snapshot_download(repo_id='{model_id}');"
                "print('Done.')"
            )
            proc = subprocess.Popen(
                [PY, "-c", script],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=DIR,
            )
            for line in proc.stdout:
                if line.strip():
                    _mwrite(line)
            proc.wait()

            def _done():
                self._model_pbar.stop()
                if proc.returncode == 0:
                    self._model_lbl.config(text="✓  Model ready.", fg=GREEN)
                    self._model_btn.config(text="Next  →", state=tk.NORMAL,
                                           command=lambda: self._goto(3))
                else:
                    self._model_lbl.config(text="✗  Download failed.", fg=RED)
                    self._model_btn.config(text="Retry", state=tk.NORMAL,
                                           command=self._do_download)
            self.after(0, _done)

        threading.Thread(target=_run, daemon=True).start()

    # ── Step 3: Done ──────────────────────────────────────────────────────
    def _step_done(self):
        tk.Label(self._area, text="Setup complete.",
                 font=("SF Pro Display", 22, "bold"), bg=BG, fg=GREEN).pack(pady=(40, 8))
        tk.Label(self._area,
                 text="Double-click Graceful.app any time to launch.\n"
                      "The browser opens automatically once the model loads.",
                 font=("SF Pro Display", 13), bg=BG, fg=FG2,
                 justify=tk.CENTER).pack(pady=(0, 32))
        tk.Button(
            self._area, text="Launch Coupled Manifold  →",
            font=("SF Pro Display", 14, "bold"),
            bg=GREEN, fg="#000", relief=tk.FLAT, padx=28, pady=12,
            cursor="hand2",
            command=self._setup_then_launch,
        ).pack()

    def _setup_then_launch(self):
        # Mark setup as complete
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(os.path.join(DATA_DIR, ".setup_complete"), "w") as f:
            f.write(time.strftime("%Y-%m-%dT%H:%M:%S"))
        # Re-build as launcher UI, then auto-launch
        for w in self.winfo_children():
            w.destroy()
        self.geometry("520x500")
        self._build_launcher()
        self.update_idletasks()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        ww, wh = self.winfo_width(), self.winfo_height()
        self.geometry(f"+{(sw-ww)//2}+{(sh-wh)//2}")
        self.after(300, self._do_launch)

    # ── Settings overlay ──────────────────────────────────────────────────
    def _open_settings(self):
        win = tk.Toplevel(self)
        win.title("Settings — Coupled Manifold")
        win.configure(bg=BG)
        win.geometry("620x480")
        win.resizable(False, False)
        win.transient(self)

        tk.Label(win, text="API Keys", font=("SF Pro Display", 16, "bold"),
                 bg=BG, fg=FG).pack(anchor=tk.W, padx=28, pady=(22, 2))
        tk.Label(win, text="Changes saved immediately. Leave blank to keep existing value.",
                 font=("SF Pro Display", 11), bg=BG, fg=FG2).pack(anchor=tk.W, padx=28)

        FIELDS = [
            ("Brave Search API Key",  "brave",           "brave.com/search/api"),
            ("YouTube API Key",       "youtube",         "console.cloud.google.com"),
            ("Google Search API Key", "google_search",   "programmablesearchengine.google.com"),
            ("Google CX ID",          "google_cx",       "Custom Search Engine ID"),
            ("Unpaywall Email",       "unpaywall_email", "any email"),
        ]

        existing = {}
        if os.path.exists(KEYS_PATH):
            try:
                existing = json.load(open(KEYS_PATH))
            except Exception:
                pass

        frame = tk.Frame(win, bg=BG)
        frame.pack(fill=tk.X, padx=28, pady=14)
        kvars = {}
        for label, key, hint in FIELDS:
            row = tk.Frame(frame, bg=BG)
            row.pack(fill=tk.X, pady=3)
            tk.Label(row, text=label, font=("SF Pro Display", 11, "bold"),
                     bg=BG, fg=FG, width=22, anchor=tk.W).pack(side=tk.LEFT)
            var = tk.StringVar(value=existing.get(key, ""))
            ent = tk.Entry(row, textvariable=var, font=MONO,
                           bg=BG3, fg=FG, insertbackground=GREEN,
                           relief=tk.FLAT, bd=6, width=28)
            ent.pack(side=tk.LEFT, padx=(6, 0))
            tk.Label(row, text=hint, font=("SF Pro Display", 9),
                     bg=BG, fg=FG2).pack(side=tk.LEFT, padx=(6, 0))
            kvars[key] = var

        def _save():
            try:
                saved = json.load(open(KEYS_PATH)) if os.path.exists(KEYS_PATH) else {}
            except Exception:
                saved = {}
            for k, var in kvars.items():
                v = var.get().strip()
                if v:
                    saved[k] = v
            os.makedirs(DATA_DIR, exist_ok=True)
            json.dump(saved, open(KEYS_PATH, "w"), indent=2)
            win.destroy()

        brow = tk.Frame(win, bg=BG)
        brow.pack(fill=tk.X, padx=28, pady=(6, 20))
        tk.Button(brow, text="Cancel", font=("SF Pro Display", 12),
                  bg=BG3, fg=FG2, relief=tk.FLAT, padx=16, pady=7,
                  cursor="hand2", command=win.destroy).pack(side=tk.RIGHT, padx=(6, 0))
        tk.Button(brow, text="Save", font=("SF Pro Display", 13, "bold"),
                  bg=GREEN, fg="#000", relief=tk.FLAT, padx=22, pady=7,
                  cursor="hand2", command=_save).pack(side=tk.RIGHT)

    # ── Close ─────────────────────────────────────────────────────────────
    def _on_close(self):
        _running = self._server_proc or _find_server_pid()
        if _running:
            if messagebox.askokcancel("Quit",
                                      "This will stop the Coupled Manifold server. Quit?"):
                self._do_stop()
                self.destroy()
        else:
            self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
