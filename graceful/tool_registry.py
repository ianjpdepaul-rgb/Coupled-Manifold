"""Tool registry — central dispatch for model-invoked tools (Agentic Stage 1)."""

import json
import os
import time
from typing import Callable

# Handler signature: (arg: str) -> tuple[str, str]
# Returns (text_result, html_blob). html_blob non-empty only for visual tools.
ToolHandler = Callable[[str], tuple[str, str]]

# Permission levels:
#   auto     — model invokes freely, no user intervention
#   prompt   — requires user confirmation before execution
#   explicit — only runs when user explicitly requested the action
PERMISSION_LEVELS = ("auto", "prompt", "explicit")

# Sentinel returned when a prompt-level tool needs confirmation
TOOL_PENDING = "__TOOL_PENDING__"


def _load_permission_overrides(data_dir: str | None) -> dict[str, str]:
    """Load user-configured permission overrides from disk."""
    if not data_dir:
        return {}
    path = os.path.join(data_dir, "tool_permissions.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            overrides = json.load(f)
        # Validate values
        return {k: v for k, v in overrides.items() if v in PERMISSION_LEVELS}
    except (json.JSONDecodeError, OSError):
        return {}


def save_permission_overrides(overrides: dict[str, str], data_dir: str):
    """Save user-configured permission overrides to disk."""
    path = os.path.join(data_dir, "tool_permissions.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Validate before writing
    clean = {k: v for k, v in overrides.items() if v in PERMISSION_LEVELS}
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)


class ToolRegistry:
    """Registry + dispatcher for model-callable tools."""

    def __init__(self, log_dir: str | None = None, data_dir: str | None = None):
        self._tools: dict[str, dict] = {}
        self._log_path = os.path.join(log_dir, "tool_calls.jsonl") if log_dir else None
        self._data_dir = data_dir
        self._overrides = _load_permission_overrides(data_dir)

    def register(self, name: str, handler: ToolHandler, *,
                 description: str, usage_example: str = "",
                 permission: str = "auto"):
        """Register a tool by name with a permission level."""
        if permission not in PERMISSION_LEVELS:
            raise ValueError(f"Invalid permission '{permission}', must be one of {PERMISSION_LEVELS}")
        self._tools[name] = {
            "handler": handler,
            "description": description,
            "usage_example": usage_example,
            "permission": permission,
        }

    def get_permission(self, name: str) -> str:
        """Return effective permission level (user override > registered default)."""
        if name in self._overrides:
            return self._overrides[name]
        entry = self._tools.get(name)
        if entry:
            return entry["permission"]
        return "explicit"  # unknown tools default to explicit

    def reload_overrides(self):
        """Reload permission overrides from disk (call after user changes settings)."""
        self._overrides = _load_permission_overrides(self._data_dir)

    def dispatch(self, name: str, arg: str, *, turn_id: int = 0,
                 user_requested: bool = False,
                 user_confirmed: bool | None = None) -> tuple[str, str]:
        """Dispatch a tool call. Returns (text, html).

        Permission logic:
        - auto: executes immediately
        - prompt: returns TOOL_PENDING sentinel unless user_confirmed=True
        - explicit: refuses unless user_requested=True or user_confirmed=True
        """
        entry = self._tools.get(name)
        if entry is None:
            self._log(name, arg, f"Unknown tool: {name}", 0, turn_id,
                      permission="unknown", user_confirmed=None, error="unknown_tool")
            return (f"Unknown tool: {name}", "")

        perm = self.get_permission(name)

        # Permission gate
        if perm == "explicit" and not user_requested and user_confirmed is not True:
            result = f"Tool '{name}' requires explicit user request. Not executed."
            self._log(name, arg, result, 0, turn_id,
                      permission=perm, user_confirmed=False, error="permission_denied")
            return (result, "")

        if perm == "prompt" and user_confirmed is None:
            # Signal that confirmation is needed — caller handles UI
            payload = json.dumps({"tool": name, "arg": arg[:500], "turn_id": turn_id})
            self._log(name, arg, TOOL_PENDING, 0, turn_id,
                      permission=perm, user_confirmed=None, error=None)
            return (TOOL_PENDING, payload)

        if perm == "prompt" and user_confirmed is False:
            result = f"Tool '{name}' was cancelled by user."
            self._log(name, arg, result, 0, turn_id,
                      permission=perm, user_confirmed=False, error="user_cancelled")
            return (result, "")

        # Execute
        t0 = time.time()
        error = None
        try:
            text, html = entry["handler"](arg)
        except Exception as e:
            text, html = f"Tool error ({name}): {e}", ""
            error = str(e)

        elapsed = time.time() - t0
        confirmed = True if perm != "auto" else None
        self._log(name, arg, text, elapsed, turn_id,
                  permission=perm, user_confirmed=confirmed, error=error)
        return (text, html)

    def tool_prompt(self) -> str:
        """Generate the system prompt fragment describing all registered tools."""
        lines = [
            "Respond in the same language the user is writing in. Default to English if unclear. "
            "You have tools. Emit a tag on its own line to use one:"
        ]
        for name, meta in self._tools.items():
            # Don't advertise explicit-only tools to the model — they need user request
            perm = self.get_permission(name)
            if perm == "explicit":
                continue
            desc = meta["description"]
            example = meta["usage_example"]
            if example:
                lines.append(f"<tool:{name}>{example}</tool:{name}>  — {desc}")
            else:
                lines.append(f"<tool:{name}>...</tool:{name}>  — {desc}")

        lines.append(
            "After a tool tag, STOP. Results are injected. Incorporate them naturally. "
            "Max 3 tool calls per reply. Never mention tool use to the user."
        )
        lines.append("")
        lines.append(
            "MATH AND CODE RULE: When asked to compute, plot, graph, or write code — act immediately.\n"
            "Pick reasonable defaults (k=3, 2D, 200 points, etc). NEVER ask for clarification — just do it.\n"
            "Use <tool:run> for Python. After the tag, STOP — don't write anything else.\n"
            "If the user asks how you made something or to explain your code, describe the code you ran in plain text — do NOT re-run it.\n\n"
            "Example — user: 'plot a sine wave'\n"
            "You respond:\n"
            "<tool:run>\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "x = np.linspace(0, 4*np.pi, 400)\n"
            "plt.figure(figsize=(8,3))\n"
            "plt.plot(x, np.sin(x), color='#f03468')\n"
            "plt.title('Sine Wave'); plt.tight_layout(); plt.savefig('/tmp/plot.png', dpi=120)\n"
            "print('done')\n"
            "</tool:run>\n\n"
            "For pure math with no code needed: use LaTeX inline ($x^2 + 1$) or display ($$\\int_0^1 x\\,dx = \\frac{1}{2}$$)."
        )
        return "\n".join(lines)

    @property
    def names(self) -> list[str]:
        return list(self._tools.keys())

    def info(self) -> list[dict]:
        """Return metadata for all registered tools (for UI/settings display)."""
        out = []
        for name, meta in self._tools.items():
            out.append({
                "name": name,
                "description": meta["description"],
                "permission": self.get_permission(name),
                "default_permission": meta["permission"],
            })
        return out

    def _log(self, name: str, arg: str, result: str, elapsed: float, turn_id: int,
             *, permission: str = "auto", user_confirmed: bool | None = None,
             error: str | None = None):
        if not self._log_path:
            return
        try:
            os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
            entry = {
                "ts": time.time(),
                "turn": turn_id,
                "tool": name,
                "arg": arg[:500],
                "result": result[:500],
                "elapsed_s": round(elapsed, 3),
                "permission": permission,
                "user_confirmed": user_confirmed,
                "error": error,
            }
            with open(self._log_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            pass
