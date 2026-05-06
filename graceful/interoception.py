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

    Compressed format: single-line sections, conditional inclusion.
    When prev_state exists and nothing changed, returns minimal stub.

    state: dict with keys:
        temperature, top_p, threshold_low, threshold_high,
        threshold_source, threshold_set_turn, current_turn,
        anti_lora_active, anti_lora_last_turn,
        fw_confidence, fw_provisional, fw_disagreements, fw_channels

    prev_state: dict or None — previous turn's snapshot for change detection.

    Returns: (block_str, state_snapshot)
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

    # If nothing changed and we have a previous state, return minimal stub
    if prev_state and not changes:
        return "APPARATUS: unchanged", snapshot

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

    # Reasons: only include when confidence < 50% (saves space when confident)
    if fw_conf < 0.50 and fw_disagree:
        reasons_str = " reasons=" + ",".join(d.replace("_", "-") for d in fw_disagree[:3])
    else:
        reasons_str = ""

    # Channel agreement summary
    chan_groups = {}
    for ch, val in fw_channels.items():
        chan_groups.setdefault(val, []).append(ch)
    agreeing = []
    for val, chs in chan_groups.items():
        if len(chs) >= 2 and val not in ("unavailable", "building"):
            agreeing.append("+".join(chs))
    channels_str = " agreeing".join([", ".join(agreeing)]) + " agreeing" if agreeing else "mixed"

    # Compressed single-line format
    lines = []
    if changes:
        lines.append("APPARATUS CHANGED THIS TURN: " + "; ".join(changes))
    lines.append(
        f"APPARATUS: t={temp:.2f} p={top_p:.2f}"
        f" | thresh={t_low:.0f}/{t_high:.0f}({thresh_age})"
        f" | anti_lora={anti_str}"
        f" | conf={conf_pct} {conf_label}{reasons_str}"
        f" | ch={channels_str}"
    )

    return "\n".join(lines), snapshot
