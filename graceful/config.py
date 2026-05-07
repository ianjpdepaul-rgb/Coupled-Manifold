"""
Coupled Manifold — configuration constants and pure helpers.
No side effects, no heavy imports.
"""
import os, json, math
import numpy as np

# ── Model ────────────────────────────────────────────────────────
MODEL       = "mlx-community/gemma-3-12b-it-4bit"
MODEL_SMALL = MODEL   # single model — Gemma 3 12B handles all complexity tiers
RANK        = 8
HUTCH_N     = 1
MAX_CTX     = 2048   # tokens for online-learning backprop
TRACE_CTX   = 32     # Hessian trace window
TRACE_LAYERS = 8     # adapter layers to subsample per trace call
MAX_NEW     = 1024   # max generated tokens
LR          = 1e-5
GRAD_ACCUM  = 4      # gradient accumulation steps
_VISION_TOKENS = 256 # Gemma 3 vision default (256 tokens/image)

# ── Paths ────────────────────────────────────────────────────────
DATA_DIR    = "./manifold_data"
_KEYS_PATH  = f"{DATA_DIR}/keys.json"
_SETTINGS_PATH = f"{DATA_DIR}/user_settings.json"

# ── Limits ───────────────────────────────────────────────────────
MAX_PROMPT_CHARS = 8000
DEV              = "mlx"
CONSEC_PATHO_LIMIT = 3
TREND_WINDOW       = 5
TREND_THRESHOLD    = -15
EXEC_TIMEOUT       = 25

# ── Keys template ────────────────────────────────────────────────
_KEYS_TEMPLATE = {
    "brave":            "",
    "google_search":    "",
    "google_cx":        "",
    "youtube":          "",
    "unpaywall_email":  "",
}


def _cosine_lr(step: int, total: int, lr_max: float = LR, lr_min: float = 1e-7) -> float:
    """Cosine annealing learning rate schedule."""
    if total <= 1:
        return lr_max
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * step / total))


def load_keys() -> dict:
    """Load API keys / config from keys.json."""
    if os.path.exists(_KEYS_PATH):
        try:
            with open(_KEYS_PATH) as _kf:
                return json.load(_kf)
        except Exception:
            pass
    return {}


def _trace_valid(t):
    """True if t is a real finite trace value (not None, not NaN, not Inf)."""
    return t is not None and isinstance(t, (int, float)) and not math.isnan(t) and not math.isinf(t)
