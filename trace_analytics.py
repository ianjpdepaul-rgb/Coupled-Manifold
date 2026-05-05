"""
COUPLED MANIFOLD — Trace Analytics & Adapter Spectral Analysis
================================================================
Cross-session Hessian trace analysis and LoRA adapter SVD profiling.

Usage:
    from trace_analytics import TraceAnalytics, analyze_adapters, save_spectral_snapshot

    ta     = TraceAnalytics(DATA_DIR)
    report = ta.format_report()          # /trace command

    analysis = analyze_adapters(model)   # /spectrum command
    report   = format_spectrum_report(analysis)

    save_spectral_snapshot(model, DATA_DIR)   # called every 25 turns
"""

import json, time
import numpy as np
from collections import defaultdict
from pathlib import Path


# ── Cross-session trace analytics ───────────────────────────────────────────

class TraceAnalytics:

    def __init__(self, data_dir: str):
        self.traces_path  = Path(data_dir) / "logs" / "traces.jsonl"
        self.sessions_dir = Path(data_dir) / "sessions"

    def load_all_traces(self) -> list:
        if not self.traces_path.exists():
            return []
        traces = []
        try:
            for line in self.traces_path.read_text().strip().split("\n"):
                if line.strip():
                    try:
                        traces.append(json.loads(line))
                    except Exception:
                        pass
        except Exception:
            pass
        return traces

    def session_summary(self) -> dict:
        """Aggregate stats across all recorded sessions."""
        traces = self.load_all_traces()
        if not traces:
            return {"sessions": 0, "total_turns": 0}

        by_session = defaultdict(list)
        for t in traces:
            by_session[t.get("session", "?")].append(t)

        session_stats = []
        for sid, turns in by_session.items():
            vals = [t["trace"] for t in turns if "trace" in t]
            if not vals:
                continue
            session_stats.append({
                "session":      sid,
                "turns":        len(turns),
                "mean_trace":   float(np.mean(vals)),
                "min_trace":    float(np.min(vals)),
                "max_trace":    float(np.max(vals)),
                "std_trace":    float(np.std(vals)),
                "n_patho":      sum(1 for t in turns if t.get("mode") == "anti"),
                "final_trend":  turns[-1].get("trend", "?"),
            })

        all_vals = [t["trace"] for t in traces if "trace" in t]
        if not all_vals:
            return {"sessions": len(by_session), "total_turns": len(traces)}

        healthiest = max(session_stats, key=lambda s: s["mean_trace"]) if session_stats else None
        worst      = min(session_stats, key=lambda s: s["mean_trace"]) if session_stats else None

        return {
            "sessions":          len(by_session),
            "total_turns":       len(traces),
            "global_mean":       float(np.mean(all_vals)),
            "global_std":        float(np.std(all_vals)),
            "healthiest_session": healthiest,
            "worst_session":     worst,
            "patho_rate":        sum(s["n_patho"] for s in session_stats) / max(len(traces), 1),
            "session_stats":     session_stats[-10:],
        }

    def collapse_patterns(self) -> list:
        """Find three-turn sequences where trace collapses sharply."""
        traces    = self.load_all_traces()
        collapses = []
        for i in range(2, len(traces)):
            curr  = traces[i].get("trace", 0)
            prev  = traces[i-1].get("trace", 0)
            prev2 = traces[i-2].get("trace", 0)
            if prev2 > 50 and prev > 0 and curr < -50:
                session_id = traces[i].get("session")
                turn       = traces[i].get("turn")
                user_msg   = self._find_user_msg(session_id, turn)
                collapses.append({
                    "session":          session_id,
                    "turn":             turn,
                    "trace_sequence":   [prev2, prev, curr],
                    "user_msg_preview": (user_msg or "")[:100],
                })
        return collapses

    def _find_user_msg(self, session_id, turn) -> str:
        if not self.sessions_dir.exists():
            return ""
        try:
            for f in self.sessions_dir.glob("*.json"):
                if str(session_id) in f.name:
                    try:
                        data = json.loads(f.read_text())
                        for t in data.get("turns", []):
                            if t.get("turn") == turn:
                                return t.get("user", "")
                    except Exception:
                        pass
        except Exception:
            pass
        return ""

    def spectral_profile(self) -> dict:
        """FFT-based periodicity analysis of the trace signal."""
        traces = self.load_all_traces()
        vals   = np.array([t["trace"] for t in traces if "trace" in t])
        if len(vals) < 10:
            return {"status": "insufficient_data", "n": len(vals)}

        try:
            from numpy.polynomial import polynomial as P
            x         = np.arange(len(vals))
            coeffs    = P.polyfit(x, vals, 2)
            trend     = P.polyval(x, coeffs)
            detrended = vals - trend
            # Determine direction from linear coefficient
            linear_coeff = float(coeffs[1]) if len(coeffs) > 1 else 0.0
            direction = ("declining" if linear_coeff < -0.5
                         else "rising" if linear_coeff > 0.5 else "flat")
        except Exception:
            detrended = vals
            direction = "unknown"

        fft   = np.fft.rfft(detrended)
        freqs = np.fft.rfftfreq(len(detrended))
        power = np.abs(fft) ** 2

        top_k   = min(5, max(1, len(power) - 1))
        # Sort ALL non-DC frequencies (skip index 0 = DC component) and take top-k
        top_idx = np.argsort(power[1:])[::-1][:top_k] + 1

        return {
            "n_turns":          int(len(vals)),
            "trend_direction":  direction,
            "dominant_periods": [
                {
                    "period": round(1 / freqs[i], 1) if freqs[i] > 0 else float('inf'),
                    "power":  round(float(power[i]), 1),
                }
                for i in top_idx
            ],
            "total_variance":    float(np.var(vals)),
            "detrended_variance": float(np.var(detrended)),
        }

    def format_report(self) -> str:
        """Human-readable report for the /trace command."""
        summary   = self.session_summary()
        collapses = self.collapse_patterns()
        spectral  = self.spectral_profile()

        lines = ["**Cross-Session Trace Analytics**\n"]
        lines.append(
            f"**Sessions:** {summary.get('sessions', 0)} | "
            f"**Turns:** {summary.get('total_turns', 0)}"
        )

        if "global_mean" in summary:
            lines.append(
                f"**Global mean trace:** {summary['global_mean']:.1f} "
                f"± {summary.get('global_std', 0):.1f}"
            )
            lines.append(
                f"**Pathological rate:** {summary.get('patho_rate', 0)*100:.1f}% "
                f"of turns in anti-LoRA"
            )

        h = summary.get("healthiest_session")
        if h:
            lines.append(
                f"**Healthiest session:** `{h['session']}` "
                f"(mean {h['mean_trace']:.0f}, {h['turns']} turns)"
            )

        w = summary.get("worst_session")
        if w and (not h or w["session"] != h["session"]):
            lines.append(
                f"**Most pathological:** `{w['session']}` "
                f"(mean {w['mean_trace']:.0f}, {w['turns']} turns)"
            )

        if collapses:
            lines.append(f"\n**Collapse events detected:** {len(collapses)}")
            for c in collapses[:3]:
                seq = " → ".join(f"{v:.0f}" for v in c["trace_sequence"])
                preview = c["user_msg_preview"] or "(no message found)"
                lines.append(f"  • t{c['turn']}: {seq}")
                lines.append(f"    *\"{preview}\"*")

        if spectral.get("dominant_periods"):
            lines.append(f"\n**Trace trend:** {spectral.get('trend_direction', 'unknown')}")
            lines.append(
                f"**Variance:** total {spectral.get('total_variance', 0):.0f} | "
                f"detrended {spectral.get('detrended_variance', 0):.0f}"
            )
            lines.append("**Dominant cycles:**")
            for p in spectral["dominant_periods"][:3]:
                period_str = f"{p['period']:.0f}-turn" if p['period'] != float('inf') else "aperiodic"
                lines.append(f"  • {period_str} cycle (power {p['power']:.0f})")
        elif spectral.get("status") == "insufficient_data":
            lines.append(f"\n*Spectral analysis: need ≥10 turns (have {spectral.get('n', 0)})*")

        recent = summary.get("session_stats", [])
        if recent:
            lines.append("\n**Recent sessions (last 5):**")
            for s in recent[-5:]:
                lines.append(
                    f"  • `{s['session']}`: {s['turns']} turns, "
                    f"mean {s['mean_trace']:.0f}, "
                    f"{s['n_patho']} anti-LoRA, trend: {s['final_trend']}"
                )

        return "\n".join(lines)


# ── Adapter spectral analysis ────────────────────────────────────────────────

def analyze_adapters(model) -> dict:
    """
    SVD analysis of all LoRA adapter matrices.
    Returns spectral profile: effective rank, concentration, norms.
    """
    profiles = []
    # Walk model tree looking for DualAdapter-like modules
    def _walk(mod, prefix=""):
        for attr_name in list(vars(mod).keys()) if not hasattr(mod, 'named_modules') else []:
            pass
        if hasattr(mod, 'named_modules'):
            for name, child in mod.named_modules():
                if (hasattr(child, 'lA') and hasattr(child, 'lB') and
                        hasattr(child, 'aA') and hasattr(child, 'aB')):
                    yield (prefix + name) if prefix else name, child
    for name, mod in _walk(model):
        try:
            # Convert MLX arrays to numpy for SVD
            import mlx.core as _mx
            lA = np.array(mod.lA).astype(np.float32)
            lB = np.array(mod.lB).astype(np.float32)
            aA = np.array(mod.aA).astype(np.float32)
            aB = np.array(mod.aB).astype(np.float32)
            W_lora = lB @ lA
            W_anti = aB @ aA

            _, s_lora, _ = np.linalg.svd(W_lora, full_matrices=False)
            _, s_anti, _ = np.linalg.svd(W_anti, full_matrices=False)

            def effective_rank(s):
                total = s.sum() + 1e-10
                s_n   = s / total
                s_n   = s_n[s_n > 1e-10]
                ent   = -np.sum(s_n * np.log(s_n + 1e-30))
                return float(np.exp(ent))

            def concentration(s):
                total = float((s**2).sum())
                return float(s[0]**2 / total) if total > 1e-10 else 0.0

            profiles.append({
                "layer": name,
                "lora": {
                    "singular_values": s_lora.tolist()[:8],
                    "effective_rank":  effective_rank(s_lora),
                    "concentration":   concentration(s_lora),
                    "frobenius_norm":  float(np.sqrt((s_lora**2).sum())),
                    "top3_energy":     float((s_lora[:3]**2).sum() / max((s_lora**2).sum(), 1e-10)),
                },
                "anti": {
                    "singular_values": s_anti.tolist()[:8],
                    "effective_rank":  effective_rank(s_anti),
                    "concentration":   concentration(s_anti),
                    "frobenius_norm":  float(np.sqrt((s_anti**2).sum())),
                    "top3_energy":     float((s_anti[:3]**2).sum() / max((s_anti**2).sum(), 1e-10)),
                },
            })
        except Exception:
            pass

    if not profiles:
        return {"error": "no adapters found", "n_adapters": 0}

    lora_ranks = [p["lora"]["effective_rank"] for p in profiles]
    anti_ranks = [p["anti"]["effective_rank"]  for p in profiles]
    lora_conc  = [p["lora"]["concentration"]   for p in profiles]

    return {
        "n_adapters":               len(profiles),
        "lora_mean_effective_rank": float(np.mean(lora_ranks)),
        "anti_mean_effective_rank": float(np.mean(anti_ranks)),
        "lora_mean_concentration":  float(np.mean(lora_conc)),
        "diagnosis":                _diagnose_adapters(profiles),
        "profiles":                 profiles[:5],
    }


def _diagnose_adapters(profiles: list) -> str:
    lora_conc = np.mean([p["lora"]["concentration"]  for p in profiles])
    lora_rank = np.mean([p["lora"]["effective_rank"]  for p in profiles])
    anti_norm = np.mean([p["anti"]["frobenius_norm"] for p in profiles])
    notes = []

    if lora_conc > 0.8:
        notes.append(
            "HIGH CONCENTRATION — learning collapsed to a single direction. "
            "The adapter is essentially rank-1. "
            "Consider resetting adapters or introducing learning rate diversity."
        )
    elif lora_conc > 0.5:
        notes.append(
            "MODERATE CONCENTRATION — learning dominated by 1–2 directions. "
            "Healthy if intentional (strong user style), concerning otherwise."
        )
    else:
        notes.append("DISTRIBUTED LEARNING — multiple directions active. Healthy spectral profile.")

    if lora_rank < 2.0:
        notes.append(
            f"EFFECTIVE RANK {lora_rank:.1f} — functionally rank-1. "
            "Full rank-8 adapter is being underused."
        )

    if anti_norm < 0.001:
        notes.append(
            "ANTI-LORA DORMANT — roughening adapters near-zero. "
            "Not activated enough, or learning rate too slow."
        )

    return " ".join(notes)


def format_spectrum_report(analysis: dict) -> str:
    """Human-readable adapter spectral analysis for the /spectrum command."""
    if "error" in analysis:
        return f"**Adapter Spectral Analysis**\n\n⚠ {analysis['error']}"

    rank = analysis.get("lora_mean_effective_rank", 0)
    conc = analysis.get("lora_mean_concentration", 0)
    n    = analysis.get("n_adapters", 0)

    lines = [
        "**Adapter Spectral Analysis**\n",
        f"**Adapter pairs:** {n}",
        f"**LoRA effective rank:** {rank:.2f}  *(1 = collapsed, ~8 = full)*",
        f"**Anti-LoRA effective rank:** {analysis.get('anti_mean_effective_rank', 0):.2f}",
        f"**LoRA concentration:** {conc:.3f}  *(1.0 = all energy in one direction)*",
        f"\n**Diagnosis:**\n{analysis.get('diagnosis', 'N/A')}",
    ]

    profiles = analysis.get("profiles", [])
    if profiles:
        lines.append("\n**First 3 adapter layers:**")
        for p in profiles[:3]:
            lr = p["lora"]
            ar = p["anti"]
            lines.append(
                f"  • `{p['layer']}`\n"
                f"    LoRA  — rank {lr['effective_rank']:.1f}, "
                f"conc {lr['concentration']:.2f}, ‖W‖ {lr['frobenius_norm']:.4f}\n"
                f"    Anti  — rank {ar['effective_rank']:.1f}, "
                f"conc {ar['concentration']:.2f}, ‖W‖ {ar['frobenius_norm']:.4f}"
            )

    return "\n".join(lines)


def save_spectral_snapshot(model, data_dir: str):
    """Save current spectral profile to logs/spectral.jsonl for longitudinal tracking."""
    try:
        analysis = analyze_adapters(model)
        if "error" in analysis:
            return
        snapshot = {
            "ts":        time.time(),
            "mean_rank": analysis.get("lora_mean_effective_rank", 0),
            "mean_conc": analysis.get("lora_mean_concentration", 0),
            "diagnosis": analysis.get("diagnosis", ""),
        }
        path = Path(data_dir) / "logs" / "spectral.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(snapshot) + "\n")
    except Exception as e:
        print(f"  [spectral snapshot failed: {e}]")
