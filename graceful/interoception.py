"""
Coupled Manifold — interoceptive apparatus state.
Surfaces sampling parameters, thresholds, anti-LoRA status,
framework confidence, and channel agreement to the model each turn.

Change annotations: when any value differs from the previous turn,
a CHANGED line is prepended so the model sees what shifted.

Pure function — no side effects, no state.
"""


def build_apparatus_block(state, prev_state=None):
    """Build the APPARATUS STATE section for the interoceptive block.

    state: dict with keys:
        temperature     — float, current sampling temperature
        top_p           — float, current top_p
        threshold_low   — float, low threshold value
        threshold_high  — float, high threshold value
        threshold_source — str, "model-set" or "percentile"
        threshold_set_turn — int or None, turn when model last set thresholds
        current_turn    — int, current turn number
        anti_lora_active — bool, whether anti-LoRA is currently engaged
        anti_lora_last_turn — int or None, turn when anti-LoRA last fired
        fw_confidence   — float 0-1, framework confidence score
        fw_provisional  — bool, whether measurement is provisional
        fw_disagreements — list of str, active disagreements
        fw_channels     — dict, per-channel assessments

    prev_state: dict or None — previous turn's state dict for change detection.

    Returns: (block_str, state_snapshot) where state_snapshot should be
             passed as prev_state on the next turn.
    """
    temp = state.get("temperature", 0.7)
    top_p = state.get("top_p", 0.95)
    t_low = state.get("threshold_low", 0)
    t_high = state.get("threshold_high", 0)
    t_source = state.get("threshold_source", "percentile")
    t_set_turn = state.get("threshold_set_turn")
    cur_turn = state.get("current_turn", 0)
    anti_active = state.get("anti_lora_active", False)
    anti_last = state.get("anti_lora_last_turn")
    fw_conf = state.get("fw_confidence", 1.0)
    fw_prov = state.get("fw_provisional", False)
    fw_disagree = state.get("fw_disagreements", [])
    fw_channels = state.get("fw_channels", {})

    # Build snapshot for change detection
    snapshot = {
        "temperature": round(temp, 2),
        "top_p": round(top_p, 2),
        "threshold_low": round(t_low, 0),
        "threshold_high": round(t_high, 0),
        "anti_lora_active": anti_active,
        "fw_confidence": round(fw_conf, 2),
        "fw_provisional": fw_prov,
    }

    # Detect changes
    changes = []
    if prev_state:
        _labels = {
            "temperature": "temperature",
            "top_p": "top_p",
            "threshold_low": "threshold_low",
            "threshold_high": "threshold_high",
            "anti_lora_active": "anti_lora",
            "fw_confidence": "framework_confidence",
            "fw_provisional": "provisional",
        }
        for key, label in _labels.items():
            old = prev_state.get(key)
            new = snapshot[key]
            if old is not None and old != new:
                changes.append(f"{label} {old} → {new}")

    # Threshold age
    if t_set_turn is not None and cur_turn > t_set_turn:
        thresh_age = f"{t_source} {cur_turn - t_set_turn} turns ago"
    else:
        thresh_age = t_source

    # Anti-LoRA status
    if anti_active:
        anti_str = "ACTIVE"
    elif anti_last is not None and cur_turn > anti_last:
        anti_str = f"inactive (last fired {cur_turn - anti_last} turns ago)"
    else:
        anti_str = "inactive"

    # Framework confidence
    conf_pct = f"{fw_conf:.0%}"
    conf_label = "PROVISIONAL" if fw_prov else "CONFIDENT"
    reasons = ", ".join(d.replace("_", "-") for d in fw_disagree[:3]) if fw_disagree else "none"

    # Channel agreement summary
    chan_groups = {}
    for ch, val in fw_channels.items():
        chan_groups.setdefault(val, []).append(ch)
    # Find channels that agree (same assessment)
    agreeing = []
    for val, chs in chan_groups.items():
        if len(chs) >= 2 and val not in ("unavailable", "building"):
            agreeing.append("+".join(chs))
    channels_str = " agreeing".join([", ".join(agreeing)]) + " agreeing" if agreeing else "mixed"

    lines = []
    if changes:
        lines.append("APPARATUS CHANGED THIS TURN: " + "; ".join(changes))
    lines.extend([
        "APPARATUS STATE:",
        f"  sampling: temperature={temp:.2f} top_p={top_p:.2f}",
        f"  thresholds: low={t_low:.0f} high={t_high:.0f} ({thresh_age})",
        f"  anti_lora: {anti_str}",
        f"  framework_confidence: {conf_pct} {conf_label}",
        f"    reasons: {reasons}",
        f"  channels: {channels_str}",
    ])

    return "\n".join(lines), snapshot
