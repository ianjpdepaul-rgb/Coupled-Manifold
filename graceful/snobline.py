"""
Coupled Manifold — SnobLine controller.
Trace-based learning gate: routes between LoRA and Anti-LoRA
based on Hessian trace percentile thresholds.

Citational regulation: the model can write to its own thresholds
via set_thresholds(). Adjustments are logged and can be deferred
when the geometry is in crisis (anti-LoRA active or patho >= 2).
"""
import time

import numpy as np

from .config import CONSEC_PATHO_LIMIT, TREND_WINDOW, TREND_THRESHOLD

_ABSOLUTE_FLOOR  = -150  # hard anti-mode trigger regardless of percentiles
_SUSTAINED_FLOOR = -100  # softer floor requiring N consecutive turns
_SUSTAINED_COUNT = 3
_ANCHOR_WINDOW   = 5     # turns to establish session baseline
_ANCHOR_DRIFT_LIMIT = -100  # rolling mean vs session anchor before tightening thresholds
_ANCHOR_TERMINATE   = -150  # rolling mean vs session anchor before termination


class SnobLine:
    def __init__(self, window=10, low_pct=30, high_pct=50, max_anti=2):
        self.window, self.low_pct, self.high_pct = window, low_pct, high_pct
        self.max_anti   = max_anti
        self.mode       = "lora"
        self.history    = []
        self.all_traces = []
        self.anti_count = 0
        self.consec_patho = 0
        self.log        = []
        self.session_anchor = None   # mean of first 5 traces — fixed session baseline
        self.consec_flattery = 0     # consecutive turns with flattery_score > 0.8
        self.manual_mode = None      # set by /adapter mode — overrides automatic routing
        # Citational regulation — model-writable thresholds
        self.model_low  = None       # absolute trace threshold for pathology (None = use percentile)
        self.model_high = None       # absolute trace threshold for healthy (None = use percentile)
        self.thresholds_history = []  # list of dicts logging every set_thresholds call
        self._threshold_queue = []   # deferred adjustments when geometry is in crisis

    def set_thresholds(self, low: int, high: int, reason: str) -> dict:
        """Model-callable: adjust routing thresholds.

        low:    pathology threshold (range -200 to 0)
        high:   healthy threshold (range 0 to 50)
        reason: required explanation string

        Returns a dict with applied status and details.
        """
        # Validate ranges
        low = max(-200, min(0, int(low)))
        high = max(0, min(50, int(high)))
        if low >= high:
            return {"applied": False, "defer_reason": "low must be less than high",
                    "low": low, "high": high}

        old_low = self.model_low
        old_high = self.model_high
        record = {
            "timestamp": time.time(),
            "old_low": old_low, "old_high": old_high,
            "new_low": low, "new_high": high,
            "reason": reason,
        }

        # Defer if geometry is in crisis
        if self.mode == "anti" or self.consec_patho >= 2:
            defer_reason = ("anti-LoRA active" if self.mode == "anti"
                            else f"patho {self.consec_patho}/3")
            record["applied"] = False
            record["defer_reason"] = defer_reason
            self._threshold_queue.append(record)
            self.thresholds_history.append(record)
            return record

        # Apply immediately
        self.model_low = low
        self.model_high = high
        record["applied"] = True
        record["defer_reason"] = None
        self.thresholds_history.append(record)
        return record

    def reset_thresholds(self):
        """Clear model-set thresholds, revert to percentile-based routing."""
        if self.model_low is not None or self.model_high is not None:
            self.thresholds_history.append({
                "timestamp": time.time(),
                "old_low": self.model_low, "old_high": self.model_high,
                "new_low": None, "new_high": None,
                "reason": "user reset", "applied": True, "defer_reason": None,
            })
        self.model_low = None
        self.model_high = None
        self._threshold_queue.clear()

    def _drain_threshold_queue(self):
        """Apply the most recent queued threshold adjustment (if any)."""
        if not self._threshold_queue:
            return
        # Only apply the last one — earlier ones are superseded
        last = self._threshold_queue[-1]
        self.model_low = last["new_low"]
        self.model_high = last["new_high"]
        last["applied"] = True
        last["defer_reason"] = "applied after recovery"
        self._threshold_queue.clear()

    def step(self, trace, turn):
        """Step the SnobLine controller with a new trace measurement.

        trace: Optional[float] — a real Hessian trace value, or None if no
               valid measurement was obtained this turn. When None, the turn
               is a no-op: no internal state changes, current mode returned
               unchanged. A measured 0.0 is a real value and IS processed.
        turn:  int — current turn number.
        Returns: str — current mode ("lora" or "anti").
        """
        # Manual override — user explicitly set a mode, don't auto-route until cleared
        if self.manual_mode is not None:
            return self.manual_mode

        # No measurement → no-op, return current mode unchanged
        if trace is None:
            return self.mode

        self.all_traces.append(trace)

        # ── Set session anchor after first N turns ──────────────────────────
        if len(self.all_traces) == _ANCHOR_WINDOW and self.session_anchor is None:
            self.session_anchor = float(np.mean(self.all_traces))
            self.log.append({"turn": turn, "session_anchor": self.session_anchor})

        # ── Circuit breaker 1: absolute floor ──────────────────────────────
        if trace < _ABSOLUTE_FLOOR:
            if self.mode != "anti":
                self.mode, self.anti_count = "anti", 0
                self.consec_patho += 1
                self.log.append({"turn": turn, "to": "anti", "trace": trace,
                                 "reason": "absolute_floor",
                                 "absolute_floor_triggered": True})
            return self.mode

        # ── Circuit breaker 2: sustained negative ──────────────────────────
        if len(self.all_traces) >= _SUSTAINED_COUNT:
            recent_n = self.all_traces[-_SUSTAINED_COUNT:]
            if all(t < _SUSTAINED_FLOOR for t in recent_n):
                if self.mode != "anti":
                    self.mode, self.anti_count = "anti", 0
                    self.consec_patho += 1
                    self.log.append({"turn": turn, "to": "anti",
                                     "reason": "sustained_negative",
                                     "sustained_negative_triggered": True})
                return self.mode

        # ── Anchor drift: tighten thresholds if session is declining ───────
        if self.session_anchor is not None and len(self.all_traces) >= 8:
            current_mean = float(np.mean(self.all_traces[-5:]))
            anchor_drift = current_mean - self.session_anchor
            if anchor_drift < _ANCHOR_DRIFT_LIMIT:
                new_low = min(self.low_pct + 10, 50)
                if new_low != self.low_pct:
                    self.low_pct = new_low
                    self.log.append({"turn": turn, "anchor_drift": anchor_drift,
                                     "new_low_pct": self.low_pct})
            elif anchor_drift > -50 and self.low_pct > 30:
                # Session recovering — ease thresholds back toward baseline
                self.low_pct = max(30, self.low_pct - 5)

        # ── Normal percentile routing ───────────────────────────────────────
        if self.mode == "anti":
            self.anti_count += 1
            if self.anti_count >= self.max_anti:
                self.mode, self.anti_count = "lora", 0
                self._drain_threshold_queue()
                self.log.append({"turn": turn, "to": "lora", "reason": "max_duration"})
                return self.mode
        else:
            self.history.append(trace)
        if len(self.history) < 3:
            return self.mode
        recent = self.history[-self.window:]
        # Use model-set thresholds if available, else percentile-computed
        if self.model_low is not None:
            low = float(self.model_low)
        else:
            low = float(np.percentile(recent, self.low_pct))
        if self.model_high is not None:
            high = float(self.model_high)
        else:
            high = float(np.percentile(recent, self.high_pct))
        if self.mode == "lora" and trace < low:
            self.mode, self.anti_count = "anti", 0
            self.consec_patho += 1
            self.log.append({"turn": turn, "to": "anti",
                             "trace": trace, "low": low, "high": high})
        elif self.mode == "anti" and trace > high:
            self.mode, self.anti_count = "lora", 0
            self.consec_patho = 0
            self._drain_threshold_queue()
            self.log.append({"turn": turn, "to": "lora", "reason": "recovered"})
        else:
            if self.mode == "lora":
                self.consec_patho = 0
        return self.mode

    def trend(self):
        if len(self.all_traces) < 3:
            return "building", 0, 0
        r = self.all_traces[-TREND_WINDOW:] if len(self.all_traces) >= TREND_WINDOW else self.all_traces
        slope = (r[-1] - r[0]) / len(r) if len(r) > 1 else 0
        avg   = np.mean(r)
        return ("declining" if slope < -10 else "rising" if slope > 10 else "stable"), avg, slope

    def should_terminate(self):
        if self.consec_patho >= CONSEC_PATHO_LIMIT:
            return True, "sustained_pathological"
        if len(self.all_traces) >= TREND_WINDOW:
            r = self.all_traces[-TREND_WINDOW:]
            if (r[-1] - r[0]) / len(r) < TREND_THRESHOLD:
                return True, "drift_detected"
        # Anchor drift termination — session mean too far below baseline
        if self.session_anchor is not None and len(self.all_traces) >= 8:
            current_mean = float(np.mean(self.all_traces[-5:]))
            if current_mean - self.session_anchor < _ANCHOR_TERMINATE:
                return True, "anchor_drift"
        return False, None

    def get_thresholds(self):
        if self.model_low is not None and self.model_high is not None:
            return float(self.model_low), float(self.model_high)
        if len(self.history) < 3:
            return 0, 0
        recent = self.history[-self.window:]
        low = float(self.model_low) if self.model_low is not None else float(np.percentile(recent, self.low_pct))
        high = float(self.model_high) if self.model_high is not None else float(np.percentile(recent, self.high_pct))
        return low, high

    def get_anti_strength(self, trace) -> float:
        """
        Scale roughening strength with pathology depth.
        trace: Optional[float]. If None, returns a safe default (0.1).
        Returns a_str in [0.05, 0.3]. Tapers during the anti window.
        """
        if trace is None:
            return 0.1
        if len(self.history) < 3:
            strength = 0.1
        else:
            low, _ = self.get_thresholds()
            if trace >= low:
                strength = 0.05
            else:
                depth = abs(trace - low)
                strength = min(0.3, 0.05 + (depth / 500))
        # Taper across anti window: 100% → 80% → 60% → 40% minimum
        if self.anti_count > 0:
            strength *= max(0.4, 1.0 - (self.anti_count * 0.2))
        return round(strength, 3)
