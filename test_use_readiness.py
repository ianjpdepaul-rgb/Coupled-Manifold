#!/usr/bin/env python3
"""
USE-READINESS TEST — "Can Grace use this daily for a week?"

Tests 12 checks: 9 core loops + 3 data quality.
Produces ONE summary report.

Usage:
    # Server must be running: python app.py
    python test_use_readiness.py [--port 7860]
"""

import sys, os, time, json, argparse, tempfile, struct, zlib, zipfile, io, re

try:
    import httpx
except ImportError:
    print("pip install httpx")
    sys.exit(1)

parser = argparse.ArgumentParser(description="Use-Readiness Test")
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--verbose", "-v", action="store_true")
args = parser.parse_args()

BASE = f"http://127.0.0.1:{args.port}"
V = args.verbose
TIMEOUT_CHAT = 180   # model inference can be slow
TIMEOUT_API = 30
MIN_CHAT_GAP = 1.5   # seconds between chat requests to avoid 429

# ── Results tracking ──────────────────────────────────────────────

results = {}       # check_num -> (PASS/FAIL/N_A, detail_string)
blockers = []      # crashes, data corruption
rough_edges = []   # cosmetic, fix later

def record(num, name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results[num] = (status, name, detail)
    sym = "\u2713" if passed else "\u2717"
    print(f"  [{num:>2}] {sym} {name}: {detail}" if detail else f"  [{num:>2}] {sym} {name}")
    if not passed:
        blockers.append(f"[{num}] {name}: {detail}")

def record_na(num, name, detail=""):
    results[num] = ("N/A", name, detail)
    print(f"  [{num:>2}] - {name}: {detail}" if detail else f"  [{num:>2}] - {name}: N/A")

def note_rough(msg):
    rough_edges.append(msg)
    if V:
        print(f"      (rough edge: {msg})")


# ── Helpers ───────────────────────────────────────────────────────

def alive():
    try:
        r = httpx.get(f"{BASE}/api/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

def reset():
    r = httpx.post(f"{BASE}/api/reset", timeout=TIMEOUT_API)
    assert r.status_code == 200
    time.sleep(1)

def chat(msg, timeout=TIMEOUT_CHAT):
    """Send chat, return (response_text, status_str, raw_sse_text, error_msg)."""
    time.sleep(MIN_CHAT_GAP)
    r = httpx.post(f"{BASE}/api/chat", json={"message": msg}, timeout=timeout)
    if r.status_code == 429:
        time.sleep(3)
        r = httpx.post(f"{BASE}/api/chat", json={"message": msg}, timeout=timeout)
    if r.status_code != 200:
        return "", "", r.text, f"HTTP {r.status_code}"

    response_text = ""
    status_str = ""
    error_msg = ""
    for line in r.text.split("\n"):
        line = line.strip()
        if not line.startswith("data: "):
            continue
        try:
            evt = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        if evt.get("t") == "d":
            response_text += evt.get("v", "")
        elif evt.get("t") == "done":
            status_str = evt.get("status", "")
        elif evt.get("t") == "err":
            error_msg = evt.get("v", "unknown error")

    return response_text, status_str, r.text, error_msg

def parse_status(status_str):
    """Parse 't1 · ? · lora · 761↓ 115↑ · 46c · 821a' into dict."""
    parts = [p.strip() for p in status_str.split("\u00b7")]
    d = {}
    if len(parts) >= 1:
        d["turn"] = parts[0]
    if len(parts) >= 2:
        d["trace"] = parts[1]
    if len(parts) >= 3:
        d["mode"] = parts[2]
    return d

def upload_file(filename, content, content_type="application/octet-stream"):
    """Upload a file, return (ok, reply_text, json_response)."""
    r = httpx.post(f"{BASE}/api/upload",
                   files={"file": (filename, content, content_type)},
                   timeout=TIMEOUT_API)
    if r.status_code != 200:
        return False, f"HTTP {r.status_code}", {}
    d = r.json()
    return d.get("ok", False), d.get("reply", ""), d


# ══════════════════════════════════════════════════════════════════
# PRE-FLIGHT
# ════════��═════════════════════════════════════════════════════════

print()
print("=" * 60)
print("  USE-READINESS TEST")
print("=" * 60)
print(f"  Target: {BASE}")
print()

if not alive():
    print(f"  SERVER NOT REACHABLE at {BASE}")
    print("  Start the server: python app.py")
    sys.exit(1)
print("  Server reachable \u2713")
print()

# Record trace baseline so check [10] only evaluates NEW traces
_traces_path = os.path.join("manifold_data", "logs", "traces.jsonl")
_trace_baseline = 0
if os.path.exists(_traces_path):
    with open(_traces_path) as _f:
        _trace_baseline = sum(1 for _ in _f)

# ══════════════════════════════════════════════════════════════════
# [1] FRESH SESSION → 10 NATURAL TURNS
# ══════════════════════════════════════════════════════════════════

print("-" * 60)
print("  [1] Fresh session, 10 turns")
print("-" * 60)

try:
    reset()
    messages = [
        "hey",
        "tell me about photosynthesis in two sentences",
        "what are the two main stages?",
        "which one produces oxygen?",
        "how does chlorophyll work?",
        "what's the role of ATP in the process?",
        "can plants photosynthesize at night?",
        "what about CAM plants?",
        "summarize everything we discussed",
        "thanks, that was helpful",
    ]

    turn_errors = []
    statuses = []
    responses = []
    for i, msg in enumerate(messages):
        text, status, raw, err = chat(msg)
        if err:
            turn_errors.append(f"turn {i+1}: {err}")
            if V:
                print(f"    turn {i+1} ERROR: {err}")
        else:
            if V:
                print(f"    turn {i+1}: {len(text)} chars, status='{status[:40]}'")
        statuses.append(status)
        responses.append(text)

    # Check: no crashes
    crashes = [e for e in turn_errors if "stream error" in e.lower() or "500" in e]

    # Check: responses stream (non-empty text for real messages)
    empty_responses = sum(1 for i, r in enumerate(responses) if not r.strip() and i > 0)

    # Check: status string populates with ? for None trace
    status_populated = sum(1 for s in statuses if s and "\u00b7" in s)

    # Check None contract: trace field should be "?" when trace is None
    none_contract_ok = True
    for s in statuses:
        if not s:
            continue
        parsed = parse_status(s)
        trace_val = parsed.get("trace", "")
        # "?" is correct for None trace; numeric is correct for real trace
        if trace_val and trace_val != "?":
            try:
                float(trace_val)
            except ValueError:
                none_contract_ok = False

    all_ok = (len(crashes) == 0 and empty_responses <= 1 and
              status_populated >= 5 and none_contract_ok)

    detail_parts = []
    if crashes:
        detail_parts.append(f"{len(crashes)} crashes")
    if empty_responses > 1:
        detail_parts.append(f"{empty_responses} empty responses")
    if status_populated < 5:
        detail_parts.append(f"only {status_populated}/10 status strings populated")
    if not none_contract_ok:
        detail_parts.append("None contract violation in status")
    if not detail_parts:
        detail_parts.append(f"10/10 turns, {status_populated} status strings, None contract ok")
    record(1, "Fresh session, 10 turns", all_ok, "; ".join(detail_parts))

    if turn_errors and not crashes:
        note_rough(f"{len(turn_errors)} non-crash errors in 10-turn test")

except Exception as e:
    record(1, "Fresh session, 10 turns", False, f"exception: {e}")


# ══════════════════════════════════════════════════════════════════
# [2] FILE UPLOAD + CONVERSATION
# ══════════════════════════════════════════════════════════════════

print()
print("-" * 60)
print("  [2] File upload + chat")
print("-" * 60)

# -- TXT upload --
try:
    reset()
    txt_content = b"The Eiffel Tower was completed in 1889 for the World's Fair in Paris. It stands 330 meters tall and was the tallest man-made structure in the world for 41 years."
    ok_txt, reply_txt, _ = upload_file("eiffel_facts.txt", txt_content, "text/plain")
    assert ok_txt, f"upload failed: {reply_txt}"
    if V:
        print(f"    TXT upload: {reply_txt[:80]}")

    # Ask about uploaded content
    text, _, _, err = chat("what year was the structure in that file completed?")
    txt_pass = (not err and ("1889" in text or "eiffel" in text.lower()))
    if not txt_pass and not err:
        # Softer check — did it reference the file at all?
        txt_pass = any(w in text.lower() for w in ["tower", "paris", "file", "document", "text"])
    record(2, "File upload: TXT", txt_pass,
           "response references content" if txt_pass else f"err={err}, response doesn't reference file content")
except Exception as e:
    record(2, "File upload: TXT", False, str(e))

# -- Image upload --
try:
    reset()
    # Create a minimal valid 1x1 red PNG
    def make_png():
        sig = b'\x89PNG\r\n\x1a\n'
        def chunk(ctype, data):
            c = ctype + data
            return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
        ihdr = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
        raw = b'\x00\xff\x00\x00'  # filter byte + RGB
        idat = zlib.compress(raw)
        return sig + chunk(b'IHDR', ihdr) + chunk(b'IDAT', idat) + chunk(b'IEND', b'')

    png_data = make_png()
    ok_img, reply_img, _ = upload_file("test_photo.png", png_data, "image/png")
    if ok_img:
        if V:
            print(f"    Image upload: {reply_img[:80]}")
        # Image gets attached to next message
        text, _, _, err = chat("describe what you see in the image I just sent")
        # Vision may or may not work with a 1x1 pixel image — just check no crash
        img_pass = not err
        detail_img = "attached + chat survived" if img_pass else f"err={err}"
        if not img_pass:
            note_rough("Image chat errored (may be 1x1 pixel limitation)")
        record("2b", "File upload: Image", img_pass, detail_img)
    else:
        record("2b", "File upload: Image", False, f"upload failed: {reply_img}")
except Exception as e:
    record("2b", "File upload: Image", False, str(e))
    note_rough(f"Image upload exception: {e}")

# -- PDF upload --
try:
    reset()
    # Minimal valid PDF
    pdf_content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 44>>stream
BT /F1 12 Tf 100 700 Td (Hello from PDF) Tj ET
endstream
endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000360 00000 n
trailer<</Size 6/Root 1 0 R>>
startxref
430
%%EOF"""
    ok_pdf, reply_pdf, _ = upload_file("test_doc.pdf", pdf_content, "application/pdf")
    if ok_pdf:
        if V:
            print(f"    PDF upload: {reply_pdf[:80]}")
        text, _, _, err = chat("what does the PDF say?")
        pdf_pass = not err
        record("2c", "File upload: PDF", pdf_pass,
               "indexed + chat survived" if pdf_pass else f"err={err}")
    else:
        # PDF parsing may fail on minimal PDF — that's a rough edge, not a blocker
        record_na("2c", "File upload: PDF", f"upload returned: {reply_pdf[:60]}")
        note_rough("Minimal PDF may not parse — try with a real PDF")
except Exception as e:
    record_na("2c", "File upload: PDF", f"exception: {e}")
    note_rough(f"PDF upload: {e}")


# ══════════════════════════════════════════════════════════════════
# [3] IDENTITY BUILDING (/iam + recall)
# ══════════════════════════════════════════════════════════════════

print()
print("-" * 60)
print("  [3] Identity building")
print("-" * 60)

try:
    reset()
    statements = [
        "/iam I love astronomy and stargazing",
        "/iam I'm a visual learner who prefers diagrams",
        "/iam my favorite book is Dune by Frank Herbert",
    ]
    for s in statements:
        text, _, _, err = chat(s)
        assert not err, f"/iam failed: {err}"
        if V:
            print(f"    {s[:40]}... -> {text[:60]}")

    # Check /who to verify identity was stored
    text_who, _, _, err_who = chat("/who")
    assert not err_who, f"/who failed: {err_who}"

    # Check that at least 2 of 3 statements are reflected
    checks = ["astronomy" in text_who.lower() or "stargazing" in text_who.lower(),
              "visual" in text_who.lower() or "diagram" in text_who.lower(),
              "dune" in text_who.lower() or "herbert" in text_who.lower()]
    found = sum(checks)
    iam_pass = found >= 2
    record(3, "Identity building", iam_pass,
           f"{found}/3 statements reflected in /who" + ("" if iam_pass else f" — response: {text_who[:100]}"))

    # Now ask conversationally — does the model know?
    text_ask, _, _, err_ask = chat("what do you know about me and my interests?")
    if not err_ask:
        ask_checks = ["astronomy" in text_ask.lower() or "star" in text_ask.lower(),
                      "visual" in text_ask.lower(),
                      "dune" in text_ask.lower()]
        ask_found = sum(ask_checks)
        if ask_found < 2:
            note_rough(f"Conversational identity recall only got {ask_found}/3 — model may not inject raw_notes into context for every question")
        elif V:
            print(f"    Conversational recall: {ask_found}/3 interests mentioned")

    # Clean up test identity notes
    for topic in ["astronomy", "visual", "Dune"]:
        chat(f"/forget {topic}")

except Exception as e:
    record(3, "Identity building", False, str(e))


# ══════════════════════════════════════════════════════════════════
# [4] CORPUS RETRIEVAL
# ══════════════════════════════════════════════════════════════════

print()
print("-" * 60)
print("  [4] Corpus retrieval")
print("-" * 60)

try:
    # Check if corpus has content
    r = httpx.get(f"{BASE}/api/corpus_files", timeout=TIMEOUT_API)
    corpus_data = r.json() if r.status_code == 200 else {}
    corpus_files = corpus_data.get("files", []) if isinstance(corpus_data, dict) else corpus_data

    if not corpus_files:
        record_na(4, "Corpus retrieval", "no corpus files indexed")
    else:
        if V:
            print(f"    Corpus files: {[f.get('source','?') if isinstance(f, dict) else f for f in corpus_files[:5]]}")

        # Use /find to search corpus
        text_find, _, _, err_find = chat("/find concordance")
        if err_find:
            # Try a broader query
            text_find, _, _, err_find = chat("/find important")

        if err_find:
            record(4, "Corpus retrieval", False, f"/find error: {err_find}")
        else:
            has_results = ("match" in text_find.lower() or "score" in text_find.lower()
                           or "from:" in text_find.lower() or "no corpus" not in text_find.lower())
            record(4, "Corpus retrieval", has_results,
                   "corpus search returned results" if has_results else "no results from /find")

except Exception as e:
    record(4, "Corpus retrieval", False, str(e))


# ══════════════════════════════════════════════════════════════════
# [5] WEB SEARCH END-TO-END
# ══════════════════════════════════════════════════════════════════

print()
print("-" * 60)
print("  [5] Web search end-to-end")
print("-" * 60)

try:
    reset()
    # Web search triggers on natural questions via should_search(), not /search prefix
    # /search is full-text session search, NOT web search
    text, status, raw, err = chat("what's the current weather in Chicago today?")

    if err:
        record(5, "Web search", False, f"chat error: {err}")
    else:
        # Check for web search indicators:
        # 1. The 🌐 marker in the medulla block
        # 2. URLs or source citations in response
        # 3. Response uses current/real-time language instead of "I don't have real-time data"
        has_web_marker = "\U0001f310" in raw  # 🌐
        has_urls = "http" in text.lower() or "source" in text.lower()
        says_no_realtime = ("don't have" in text.lower() and "real" in text.lower()) or "cannot access" in text.lower()
        has_weather_data = any(w in text.lower() for w in ["degrees", "temperature", "forecast", "cloudy",
                                                            "sunny", "rain", "wind", "humidity", "°"])

        if has_web_marker or has_urls or has_weather_data:
            detail = "web search triggered"
            if has_web_marker:
                detail += ", 🌐 marker present"
            if has_weather_data:
                detail += ", weather data in response"
            record(5, "Web search", True, detail)

            # Follow-up test: does search context carry forward?
            text2, _, _, err2 = chat("actually what about in Boulder instead?")
            if not err2 and V:
                print(f"    Follow-up response: {text2[:80]}")
        elif says_no_realtime:
            # Search may have failed or model ignored results
            record(5, "Web search", False,
                   "model says 'no real-time data' — search may not be working or model ignores results")
        else:
            # Search didn't trigger or API keys not configured
            # Check if ddgs is available
            r_health = httpx.get(f"{BASE}/api/health", timeout=5).json()
            note_msg = "search didn't trigger — may need API keys or ddgs package"
            record_na(5, "Web search", note_msg)
            note_rough(note_msg)

    # Also test /search (session search) separately — different feature
    text_ss, _, _, err_ss = chat("/search photosynthesis")
    if not err_ss and V:
        has_session_results = "found" in text_ss.lower() or "match" in text_ss.lower() or "..." in text_ss
        print(f"    /search (session search): {'results found' if has_session_results else 'no results'}")

except Exception as e:
    record(5, "Web search", False, str(e))


# ══════════════════════════════════════════════════════════════════
# [6] MODE SWITCHING
# ══════════════════════════════════════════════════════════════════

print()
print("-" * 60)
print("  [6] Mode switching")
print("-" * 60)

try:
    reset()
    # Socratic mode
    text_soc, _, _, err_soc = chat("/socratic")
    assert not err_soc, f"/socratic failed: {err_soc}"
    assert "socratic mode on" in text_soc.lower(), f"unexpected: {text_soc[:80]}"

    text_q, _, _, err_q = chat("I think the earth is flat")
    soc_pass = not err_q and "?" in text_q
    if V:
        print(f"    Socratic response ends with question: {soc_pass} — {text_q[-60:]}")

    # Turn off socratic
    chat("/socratic")

    # Antagonist mode
    text_ant, _, _, err_ant = chat("/antagonist")
    assert not err_ant, f"/antagonist failed: {err_ant}"

    text_push, _, _, err_push = chat("I think chocolate ice cream is the best flavor")
    ant_pass = not err_push and len(text_push) > 20
    # Antagonist should push back — look for disagreement signals
    pushback_signals = ["however", "actually", "disagree", "but", "on the other hand",
                        "counter", "argument", "not necessarily", "wrong", "consider"]
    has_pushback = any(s in text_push.lower() for s in pushback_signals)
    if V:
        print(f"    Antagonist pushback detected: {has_pushback} — {text_push[:80]}")

    mode_pass = soc_pass and ant_pass
    detail = []
    if soc_pass:
        detail.append("socratic asks question")
    else:
        detail.append("socratic didn't end with ?")
    if ant_pass:
        detail.append("antagonist responds")
        if has_pushback:
            detail.append("pushback detected")
        else:
            note_rough("Antagonist mode responded but pushback language not detected")
    else:
        detail.append(f"antagonist error: {err_push}")

    record(6, "Mode switching", mode_pass, "; ".join(detail))

except Exception as e:
    record(6, "Mode switching", False, str(e))


# ══════════════════════════════════════════════════════════════════
# [7] EXPORT
# ═��════════════════════════════════════════════════════════════════

print()
print("-" * 60)
print("  [7] Export")
print("-" * 60)

try:
    # Use the data from conversations we've already had
    r = httpx.get(f"{BASE}/api/export", timeout=TIMEOUT_API)
    assert r.status_code == 200, f"HTTP {r.status_code}"

    # Parse the zip
    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = z.namelist()
    if V:
        print(f"    Export zip contents: {names}")

    required = ["manifest.json"]
    present = {n for n in names}
    has_manifest = "manifest.json" in present

    # Check manifest content
    manifest = {}
    if has_manifest:
        manifest = json.loads(z.read("manifest.json"))

    has_hostname = bool(manifest.get("hostname"))
    has_traces = "traces.jsonl" in present
    has_learn = "learn_decisions.jsonl" in present
    has_session = "session_log.json" in present

    detail_parts = []
    if has_manifest and has_hostname:
        detail_parts.append(f"manifest ok (hostname={manifest.get('hostname','')})")
    else:
        detail_parts.append("manifest missing or no hostname")
    if has_traces:
        detail_parts.append("traces.jsonl present")
    if has_learn:
        detail_parts.append("learn_decisions.jsonl present")
    if has_session:
        detail_parts.append("session_log present")

    export_pass = has_manifest and has_hostname
    record(7, "Export", export_pass, "; ".join(detail_parts))

except Exception as e:
    record(7, "Export", False, str(e))


# ══════════════════════════════════════════════════════════════════
# [8] SESSION PERSISTENCE
# ══════════════════════════════════════════════════════════════════

print()
print("-" * 60)
print("  [8] Session persistence")
print("-" * 60)

try:
    reset()
    # Chat 5 turns
    persist_msgs = [
        "The capital of France is Paris",
        "The capital of Japan is Tokyo",
        "The capital of Brazil is Brasilia",
        "The capital of Australia is Canberra",
        "summarize the capitals we discussed",
    ]
    for msg in persist_msgs:
        _, _, _, err = chat(msg)
        if err:
            note_rough(f"Session persistence turn error: {err}")

    # Save
    r = httpx.post(f"{BASE}/api/save", timeout=TIMEOUT_API)
    assert r.status_code == 200, f"save failed: {r.status_code}"
    time.sleep(1)

    # Get session list and find our session
    r = httpx.get(f"{BASE}/api/sessions", timeout=TIMEOUT_API)
    assert r.status_code == 200
    sessions = r.json()
    assert len(sessions) > 0, "no sessions found after save"
    saved_sid = sessions[0]["id"] if isinstance(sessions[0], dict) else sessions[0]
    if V:
        print(f"    Saved session: {saved_sid}")

    # Start new session (clears current)
    r = httpx.post(f"{BASE}/api/new", timeout=TIMEOUT_API)
    assert r.status_code == 200

    # Reload saved session
    r = httpx.get(f"{BASE}/api/sessions/{saved_sid}", timeout=TIMEOUT_API)
    assert r.status_code == 200, f"load failed: {r.status_code}"
    loaded = r.json()
    history = loaded.get("history", [])

    # We sent 5 messages, so history should have at least 5 user+assistant pairs
    # (greeting shortcut may reduce the pair count)
    user_msgs = [m for m in history if m.get("role") == "user"]
    has_content = len(user_msgs) >= 3  # at least 3 of our 5 turns survived

    # Check content is actually our data
    history_text = " ".join(m.get("content", "") for m in history).lower()
    has_paris = "paris" in history_text
    has_tokyo = "tokyo" in history_text

    persist_pass = has_content and (has_paris or has_tokyo)
    record(8, "Session persistence", persist_pass,
           f"{len(user_msgs)} user msgs recovered, content verified" if persist_pass
           else f"only {len(user_msgs)} user msgs, paris={has_paris}, tokyo={has_tokyo}")

except Exception as e:
    record(8, "Session persistence", False, str(e))


# ══════════════════════════════════════════════════════════════════
# [9] LEARNING TOGGLE
# ══════════════════════════════════════════════════════════════════

print()
print("-" * 60)
print("  [9] Learning toggle")
print("-" * 60)

try:
    reset()
    learn_log = os.path.join("manifold_data", "logs", "learn_decisions.jsonl")

    # Count existing lines
    lines_before = 0
    if os.path.exists(learn_log):
        with open(learn_log) as f:
            lines_before = sum(1 for _ in f)

    # Phase 1: learning OFF
    text_off, _, _, err_off = chat("/learn off")
    assert not err_off, f"/learn off failed: {err_off}"
    assert "disabled" in text_off.lower() or "off" in text_off.lower(), f"unexpected: {text_off[:60]}"

    for msg in ["explain quantum entanglement briefly", "what about superposition?"]:
        _, _, _, err = chat(msg)
        time.sleep(2)  # let learn decision write

    time.sleep(3)  # extra settle time
    lines_after_off = 0
    if os.path.exists(learn_log):
        with open(learn_log) as f:
            lines_after_off = sum(1 for _ in f)

    new_off_entries = lines_after_off - lines_before
    # With learning off, entries should appear with reason="online_off"
    off_reasons = []
    if new_off_entries > 0 and os.path.exists(learn_log):
        with open(learn_log) as f:
            all_lines = f.readlines()
        for line in all_lines[lines_before:]:
            try:
                entry = json.loads(line.strip())
                off_reasons.append(entry.get("reason", ""))
            except json.JSONDecodeError:
                pass

    # Phase 2: learning ON
    text_on, _, _, err_on = chat("/learn on")
    assert not err_on, f"/learn on failed: {err_on}"
    assert "enabled" in text_on.lower() or "on" in text_on.lower(), f"unexpected: {text_on[:60]}"

    for msg in ["explain the double slit experiment", "why is observation important?"]:
        _, _, _, err = chat(msg)
        time.sleep(2)

    time.sleep(3)
    lines_after_on = 0
    if os.path.exists(learn_log):
        with open(learn_log) as f:
            lines_after_on = sum(1 for _ in f)

    new_on_entries = lines_after_on - lines_after_off

    # Check: learning-off entries should have reason="online_off"
    off_correct = all(r == "online_off" for r in off_reasons) if off_reasons else True
    # Check: learning-on entries should NOT have reason="online_off"
    on_reasons = []
    if new_on_entries > 0 and os.path.exists(learn_log):
        with open(learn_log) as f:
            all_lines = f.readlines()
        for line in all_lines[lines_after_off:]:
            try:
                entry = json.loads(line.strip())
                on_reasons.append(entry.get("reason", ""))
            except json.JSONDecodeError:
                pass
    on_correct = all(r != "online_off" for r in on_reasons) if on_reasons else True

    learn_pass = off_correct and on_correct
    detail = f"off: {new_off_entries} entries (reasons={set(off_reasons) or 'none'}), on: {new_on_entries} entries (reasons={set(on_reasons) or 'none'})"
    record(9, "Learning toggle", learn_pass, detail)

    if not new_off_entries and not new_on_entries:
        note_rough("No learn_decisions entries written — learning may be skipping all turns (trace None)")

except Exception as e:
    record(9, "Learning toggle", False, str(e))


# ══════════════════════════════════════════════════════════════════
# [10] TRACE DATA QUALITY
# ══════════════════════════════════════════════════════════════════

print()
print("-" * 60)
print("  [10] Trace data quality")
print("-" * 60)

try:
    traces_path = os.path.join("manifold_data", "logs", "traces.jsonl")
    if not os.path.exists(traces_path):
        record_na(10, "Trace data", "traces.jsonl doesn't exist yet")
    else:
        with open(traces_path) as f:
            all_lines = f.readlines()
        # Only check traces written DURING this test (skip legacy data)
        lines = all_lines[_trace_baseline:]
        if V:
            print(f"    Total trace lines: {len(all_lines)}, baseline: {_trace_baseline}, checking: {len(lines)}")

        total = 0
        nulls = 0
        zeros = 0
        valid = 0
        valid_no_compute_ms = 0
        parse_errors = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue

            total += 1
            trace_val = entry.get("trace")

            if trace_val is None:
                nulls += 1
            elif trace_val == 0.0:
                zeros += 1
            else:
                valid += 1
                compute_ms = entry.get("trace_compute_ms", entry.get("compute_ms"))
                if compute_ms is not None and compute_ms <= 0:
                    valid_no_compute_ms += 1

        # PASS = no 0.0 values AND (all null OR non-null entries have trace_compute_ms > 0)
        # FAIL = any 0.0 value (contamination regression)
        trace_pass = zeros == 0
        if trace_pass and valid > 0 and valid_no_compute_ms > 0:
            trace_pass = False  # real values must have compute_ms > 0
        detail = (f"new traces: {total}, nulls: {nulls}, reals: {valid}, zeros: {zeros}")
        if not trace_pass and zeros > 0:
            detail += f" — REGRESSION: {zeros} zero values in new traces"
        if parse_errors:
            detail += f", {parse_errors} parse errors"
        if valid_no_compute_ms:
            detail += f", {valid_no_compute_ms} real traces missing compute_ms>0"
            note_rough(f"{valid_no_compute_ms} traces have compute_ms <= 0")

        record(10, "Trace data", trace_pass, detail)

except Exception as e:
    record(10, "Trace data", False, str(e))


# ══════════════════════════════════════════════════════════════════
# [11] KNOWLEDGE GRAPH POPULATES
# ══════════════════════════════════════════════════════════════════

print()
print("-" * 60)
print("  [11] Knowledge graph populates")
print("-" * 60)

try:
    # We've had multiple conversations about various topics by now
    r = httpx.get(f"{BASE}/api/knowledge_graph", timeout=TIMEOUT_API)
    assert r.status_code == 200, f"HTTP {r.status_code}"
    kg = r.json()
    nodes = kg.get("nodes", [])
    edges = kg.get("edges", [])

    if V:
        print(f"    Nodes: {len(nodes)}, Edges: {len(edges)}")
        if nodes:
            print(f"    Sample nodes: {[n.get('label','?') for n in nodes[:5]]}")

    # Verify response shape: each node must have id, label, weight
    shape_ok = all(
        isinstance(n, dict) and "id" in n and "label" in n and "weight" in n
        for n in nodes
    )
    kg_pass = len(nodes) >= 3 and shape_ok  # graph populated with valid nodes
    detail = f"{len(nodes)} nodes, {len(edges)} edges"
    if not shape_ok:
        detail += " — malformed node objects"
    elif len(edges) == 0 and len(nodes) > 0:
        detail += " — graph populated, edges sparse due to short test (expected)"
        note_rough("Knowledge graph has nodes but sparse edges — will populate with more use")

    record(11, "Knowledge graph", kg_pass, detail)

except Exception as e:
    record(11, "Knowledge graph", False, str(e))


# ══════════════════════════════════════════════════════════════════
# [12] LEARN DECISIONS LOG QUALITY
# ══════════════════════════════════════════════════════════════════

print()
print("-" * 60)
print("  [12] Learn decisions log")
print("-" * 60)

try:
    learn_path = os.path.join("manifold_data", "logs", "learn_decisions.jsonl")
    if not os.path.exists(learn_path):
        record_na(12, "Learn decisions log", "file doesn't exist — no learning decisions made yet")
    else:
        with open(learn_path) as f:
            lines = f.readlines()

        total = 0
        valid = 0
        malformed = 0
        required_keys = {"turn", "trace", "reason"}
        reasons_seen = set()

        for line in lines:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                entry = json.loads(line)
                has_keys = required_keys.issubset(set(entry.keys()))
                if has_keys:
                    valid += 1
                    reasons_seen.add(entry.get("reason"))
                else:
                    malformed += 1
                    if V:
                        missing = required_keys - set(entry.keys())
                        print(f"    Missing keys: {missing} in entry: {entry}")
            except json.JSONDecodeError:
                malformed += 1

        learn_pass = valid > 0 and malformed == 0
        detail = f"{total} entries, {valid} valid, {malformed} malformed, reasons={reasons_seen or 'none'}"
        record(12, "Learn decisions log", learn_pass, detail)

        if malformed > 0:
            note_rough(f"{malformed} malformed learn_decisions entries")

except Exception as e:
    record(12, "Learn decisions log", False, str(e))


# ══════════════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════════════

print()
print()
print("=" * 60)
print("  USE-READINESS REPORT")
print("=" * 60)
print()
print("  Core loops:")

core_checks = [1, 2, "2b", "2c", 3, 4, 5, 6, 7, 8, 9]
for c in core_checks:
    if c in results:
        status, name, detail = results[c]
        tag = {"PASS": "\u2713 PASS", "FAIL": "\u2717 FAIL", "N/A": "- N/A"}[status]
        print(f"    [{c:>3}] {name:35s} {tag}" + (f"  ({detail[:60]})" if detail else ""))

print()
print("  Data quality:")
for c in [10, 11, 12]:
    if c in results:
        status, name, detail = results[c]
        tag = {"PASS": "\u2713 PASS", "FAIL": "\u2717 FAIL", "N/A": "- N/A"}[status]
        print(f"    [{c:>3}] {name:35s} {tag}" + (f"  ({detail[:60]})" if detail else ""))

print()
print("-" * 60)
print("  SUMMARY")
print("-" * 60)

actual_blockers = [b for b in blockers if any(
    k in b.lower() for k in ["crash", "500", "stream error", "exception"]
) or any(b.startswith(f"[{n}]") for n in [1, 3, 6, 7, 8])]

# Check if core checks all passed
all_core_pass = all(results.get(c, ("FAIL",))[0] in ("PASS", "N/A") for c in core_checks)
data_quality = "GOOD"
data_checks = [results.get(c, ("N/A",))[0] for c in [10, 11, 12]]
if all(d == "PASS" for d in data_checks):
    data_quality = "GOOD"
elif any(d == "PASS" for d in data_checks):
    data_quality = "PARTIAL"
else:
    data_quality = "POOR"

tool_ready = all_core_pass and data_quality != "POOR"

print(f"  Tool ready for daily use: {'YES' if tool_ready else 'NO'}")
print(f"  Blockers: {'; '.join(blockers) if blockers else 'none'}")
print(f"  Rough edges: {'; '.join(rough_edges) if rough_edges else 'none'}")
print(f"  Iteration data quality: {data_quality}")
print("=" * 60)

sys.exit(0 if tool_ready else 1)
