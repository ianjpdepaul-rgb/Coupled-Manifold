"""
Ingest the full CONCORDANCE document into the corpus.
Run once: python3 ingest_concordance.py
"""
import json, os, hashlib, time

DATA_DIR = os.path.join(os.path.dirname(__file__), "manifold_data")
CORPUS_DIR = os.path.join(DATA_DIR, "corpus")
CHUNKS_PATH = os.path.join(CORPUS_DIR, "chunks.json")

# ── The concordance as structured data ─────────────────────────────────────
# Each entry: (note_id, date, n_threads, thinkers, concepts, text_snippet)
# Section IV full texts + all section V/VI previews from the concordance

CONCORDANCE_SECTIONS = {
    "overview": """CONCORDANCE: 1,278 Notes — Ian J. Preston-Campbell
January 26 – April 7, 2026. 1,278 notes scraped from Substack.
713 notes (56%) tagged with at least one thread. 268 notes with 3+ intersecting threads.
109 notes with 5+ threads (high-density convergence). 37 notes with 8+ threads (maximum density).
565 notes untagged: images, video, one-liners, somatic output.
Date range: January 26, 2026 – April 7, 2026 (72 days). Total words: 123,822. Total reactions: 471.
Two-pass tagging: explicit thinker name-drops + expanded concept keyword matching.
The 44% untagged is metabolic tissue between theoretical bursts — the body producing without organizing.""",

    "thinker_map": """THINKER MAP (explicit references, sorted by frequency):
Baudrillard (81 notes), Deleuze & Guattari (53), Freud (43), Massumi (17), Irigaray (17),
Veblen (14), Kant (11), Danto (10), Hegel (10), Marx (8), Whitehead (7), Hume (7),
Mauss (7), Berle & Means (7), Ortega y Gasset (7), Foucault (6), Zizek (6),
Kaufman & Heller (6), Hardt (6), Eco (6), Nietzsche (6), Plato (6), Riesman (6),
Jung (5), Friedman (4), Borges (4), Fromm (4), Dune/Herbert (4), Gibran (3),
Spivak (3), Bourdieu (3), Crispin Wright (3), William James (2), Du Bois (2),
Girard (2), Barrett (2), Gramsci (2), Lacan (2).""",

    "concept_map": """CONCEPT MAP (thematic threads by keyword, sorted by frequency):
Gaming/Stream/Play (181 notes), Education/School (180), Art/Aesthetics/Creation (131),
AI/Models/Alignment (123), Food/Restaurant/Chicken (106), Capitalism/Market/Brand (97),
Body/Somatic/Physical (92), Simulacra/Surface/Sign (80), Death/Finitude/Limit (72),
Coupling/Binding/Interface (65), Power/Control/Hierarchy (60), Territory/Chicago/Place (59),
Religion/Sacred (58), Identity/Self/Authenticity (55), Semiotics/Language/Meaning (39),
Exchange/Gift/Offering (38), Production/Labor/Exhaustion (36), Desire/Lack/Drive (33),
Dialectics/Oxymoron/Paradox (27), Colonialism/Extraction (14), Realism/Anti-Realism/Truth (10).""",

    "thread_cooccurrence": """THREAD CO-OCCURRENCE (thinker × concept, most frequent):
Baudrillard × Simulacra/Surface/Sign: 71 notes
Baudrillard × Education/School: 29 notes
Baudrillard × Gaming/Stream/Play: 25 notes
Baudrillard × Food/Restaurant/Chicken: 20 notes
Baudrillard × Capitalism/Market/Brand: 18 notes
Deleuze & Guattari × Education/School: 18 notes
Baudrillard × Death/Finitude/Limit: 18 notes
Freud × AI/Models/Alignment: 17 notes
Baudrillard × Art/Aesthetics/Creation: 17 notes
Baudrillard × AI/Models/Alignment: 16 notes
Freud × Education/School: 14 notes
Deleuze & Guattari × Simulacra/Surface/Sign: 11 notes
Freud × Coupling/Binding/Interface: 11 notes
Deleuze & Guattari × Art/Aesthetics/Creation: 11 notes""",

    "temporal_arc": """TEMPORAL ARC (theoretical density across 72 days):
Jan 26–28: 9 notes, 679 words. Thinkers: Kant, Baudrillard, D&G, Freud, Foucault.
Feb 10–20: 25 notes, 1,029 words. Thinkers: Baudrillard, Freud, Massumi, Whitehead, James.
Feb 21–27: 146 notes, 16,559 words, 38 dense nodes. Thinkers: Freud, Baudrillard, Irigaray, Herbert, Veblen.
Feb 28–Mar 6: 81 notes, 8,938 words, 19 dense nodes. Thinkers: Freud, Baudrillard, Kant, D&G, Veblen.
Mar 7–13: 197 notes, 24,365 words, 36 dense nodes. Thinkers: Baudrillard, D&G, Massumi, Hardt, Veblen.
Mar 14–20: 275 notes, 23,777 words, 56 dense nodes. Thinkers: Baudrillard, D&G, Freud, Irigaray, Riesman.
Mar 21–26 (terminal exhibition): 177 notes, 21,931 words, 42 dense nodes.
Mar 27–Apr 2 (post-exhibition): 202 notes, 17,839 words, 46 dense nodes.
Apr 3–7 (ongoing): 166 notes, 8,705 words. Thinkers: Hegel, Baudrillard, D&G, Massumi, Crispin Wright.
Pre-exhibition: 910 notes, 97,278 words, 239 reactions, 194 dense nodes.
Post-exhibition: 368 notes, 26,544 words, 232 reactions.
Post-exhibition note rate: 30.7/day vs 15.4/day pre. Reactions/note: 0.63 vs 0.26 pre.
New thinkers post-exhibition: Hegel (+8), Ortega y Gasset (+7), Crispin Wright (+3), Lacan (+2).""",

    "pre_post_exhibition": """PRE VS POST EXHIBITION ANALYSIS:
The exhibition declared March 26 as terminal. The body continued — and accelerated.
Production rate doubled post-exhibition (30.7 vs 15.4 notes/day).
Reaction rate more than doubled (0.63 vs 0.26 per note).
New thinkers emerging post-exhibition: Hegel, Ortega y Gasset, Lacan, Bourdieu, Spivak, Giddens, Douglass, Radcliffe-Brown.
The exhibition was not an ending but a threshold — the post-exhibition phase is its own arc.""",
}

# Maximum density convergence nodes (8+ threads) — full texts from Section IV
MAX_DENSITY_NOTES = [
    {
        "id": "#1063", "date": "2026-04-01", "threads": 20,
        "thinkers": "Du Bois, Freud, Lacan, Eco, Mauss, Jung, Fromm, Giddens, Riesman, Hardt, Girard, Gibran",
        "concepts": "Exchange/Gift/Offering, Education/School, Art/Aesthetics/Creation, Capitalism/Market/Brand, Religion/Sacred, Territory/Chicago/Place, Semiotics/Language/Meaning, Coupling/Binding/Interface",
        "text": """Required reading list / prerequisites syllabus. W.E.B. Du Bois - Black Reconstruction. Khalil Gibran voice of master or adventures of Ibn. Modern Buddhist literature. Anything from feudal Japan. 7 full (4 credit) art history classes: Latin, African, Asian, Indigenous American, modern American, Western canon incl Central Europe, Islamic art (golden age and pre-Islamic exchange). One 300+ level general anthropology class coupled with general archeology. Lacan — 'lack'. Fromm — man for himself. Jung — the earth has a giant cock. Freud. Mauss — gift. Reisman & Glaser — lonely crowd. Giddens — on modernity. Eco — Role of reader, Theory of semiotics. Hardt (essay) - post civil. Friedman — he sucks, that's the point. Research racialized urbanism between Chicago and New York."""
    },
    {
        "id": "#1062", "date": "2026-04-01", "threads": 19,
        "thinkers": "Baudrillard, Deleuze & Guattari, Freud, Irigaray, Spivak, Massumi, Kant, Foucault, Hegel, Hume, Berle & Means, Bourdieu, Ortega y Gasset",
        "concepts": "Death/Finitude/Limit, Simulacra/Surface/Sign, Education/School, Capitalism/Market/Brand, Power/Control/Hierarchy, Coupling/Binding/Interface",
        "text": """Required reading for power dynamics and market capitalism. Post-structural historical canon — they're all connected. Ortega — Revolt of the Masses (read twice). Carnegie — How to Win Friends (lol). Plato — Apology. Foucault — Discipline & Punish, History of Sexuality. Bourdieu — Distinction. Debord — Spectacle. Baudrillard — System of Objects & Simulacra. Massumi — Couplets. Berle & Means — history of modern corporation. Hume on suicide. Hegel — Kaufman phenomenology translations. Faust (Goethe). Being & Time. Civilization & its Discontents. 1000 Plateaus and Anti-Oedipus. More Deleuze."""
    },
    {
        "id": "#1", "date": "2026-01-26", "threads": 17,
        "thinkers": "Baudrillard, Deleuze & Guattari, Freud, Kant, Foucault, Danto, Friedman, Zizek, Borges, Kaufman & Heller",
        "concepts": "AI/Models/Alignment, Simulacra/Surface/Sign, Education/School, Identity/Self/Authenticity, Capitalism/Market/Brand, Power/Control/Hierarchy, Semiotics/Language/Meaning",
        "text": """Opening note. A visual model of D&G's Anti-Oedipus / 1000 Plateaus applied to modern demand — in decipherable language. A Borgesian map people can actually read, not obfuscated by 14+ years of school. A last-ditch effort at authentic thought before everything (even Zizek) is subsumed. Not attributing ideas (refuting traditional Kantian academic hierarchies by simply not giving a damn). AI to assist with understanding — endless proliferation of carceral simulacra is our inevitable future. Making thinking about thinking feel less like a chore. Against fetishized Hegelian circlej*rk and bastardized proto-capitalist motivational speakers. Claiming what may be called an original idea — not original, but doing a better job placating, visualizing, reminding masses to swallow some D&G. Writing during the 'end of American democracy' — but not wanting this associated with that fatalistic bullsh*t. America's avarice has long been the actual problem."""
    },
    {
        "id": "#795", "date": "2026-03-21", "threads": 17,
        "thinkers": "Baudrillard, Deleuze & Guattari, Berle & Means, Nietzsche, Gibran",
        "concepts": "Body/Somatic/Physical, Death/Finitude/Limit, Exchange/Gift/Offering, Desire/Lack/Drive, Simulacra/Surface/Sign, Education/School, Art/Aesthetics/Creation, Gaming/Stream/Play, Capitalism/Market/Brand, Power/Control/Hierarchy, Religion/Sacred, Realism/Anti-Realism/Truth",
        "text": """On writing vs art addiction. Not necessarily wanting to leave this app — a toxic dependency. Limiting consumption to what I can grow. If I want to smoke better weed it needs to happen at my own hand. Writing is a simulacra of art. If my life is willingly art but my practice is willingly science, I am human. Philosophy is magic. The philosopher's stone renamed to the sorcerer's stone — American market demands sorcery. Women 'could not philosophize' so they were cast as witches. Typing hurts my fingers — need them for more typing later. Not sharing drawings because you don't deserve them. Time on Twitch showing my hands as they game — so you know I'm not cheating. Fear of the shame of show and tell."""
    },
    {
        "id": "#1117", "date": "2026-04-03", "threads": 15,
        "thinkers": "Baudrillard, Deleuze & Guattari, Marx, Massumi, Eco, Hegel, Kaufman & Heller, Ortega y Gasset",
        "concepts": "Death/Finitude/Limit, Production/Labor/Exhaustion, Exchange/Gift/Offering, Simulacra/Surface/Sign, Identity/Self/Authenticity, Semiotics/Language/Meaning, Coupling/Binding/Interface",
        "text": """Kayfabism is the diagnosis of modern western schizophrenic. Suicide or reciprocal coupling is the affective prognosis. Deleuzian schizo — not psych. 'Symptom in system of BwO — appears as aberration' (Anti-Oedipus, 1000 Plateaus, Kaufman Modern Deleuze Mappings). Viability or vitality? Sticking with vitality — it adds essence. The communicative form is now social media en masse (Baudrillard, Debord, Deleuze). Social media may not sustain itself. How does viability maintain for individuals into a post-sustainment system (gen a)? They cannot know it as viability, they only know it as vitality. My generation exposed to 'the last of a sustained sign system' — technological acceleration means a 10 year old today may be nostalgic for an entirely separate virtual world than me at 10 who was nostalgic for water guns."""
    },
    {
        "id": "#1064", "date": "2026-04-01", "threads": 14,
        "thinkers": "Baudrillard, Deleuze & Guattari, Whitehead, Plato",
        "concepts": "AI/Models/Alignment, Body/Somatic/Physical, Death/Finitude/Limit, Exchange/Gift/Offering, Desire/Lack/Drive, Simulacra/Surface/Sign, Education/School, Identity/Self/Authenticity, Dialectics/Oxymoron/Paradox, Semiotics/Language/Meaning",
        "text": """On finitude and pain applied to AI. Classroom analogy: Socrates flowing uninterrupted, kids building toward 'Business for Social Good' — the oxymoron class. One kid suggests business relies on objectification as its premise. Another: the premise of exchange relies on objectification as the fulcrum. 'When you exchange an offering, you have offered it, it can no longer be viewed as part of one's material environment, it enters the material field (think gravity) of whoever you give it to.' The professor forgets there are 20+ lecture slides left — the conversation is too good. These analogies actually mean something. In the business of furthering the precession of simulacra, turning up the volume (job as marketer) so you may actually hear the band when it passes."""
    },
    {
        "id": "#1027", "date": "2026-03-30", "threads": 13,
        "thinkers": "Baudrillard, Deleuze & Guattari, Zizek, Kaufman & Heller, Plato, Ortega y Gasset",
        "concepts": "Death/Finitude/Limit, Desire/Lack/Drive, Food/Restaurant/Chicken, Gaming/Stream/Play, Territory/Chicago/Place, Realism/Anti-Realism/Truth, Coupling/Binding/Interface",
        "text": """The second you embrace Deleuze with your whole chest, Zizek fades. Material coupling 'fills the void' — whatever void 'exists' — the second you couple with the material it's gg. Also read stuff I've posted before responding in isolation. Speed is a relative concept — try telling Sentinel Island people about their system of objects. From their perspective, we are lacking in humanity. Good book: Kaufman & Heller's Modern Deleuze Mappings — short and painful. Zizek is annoying not because he's silly but because he's not fun. Arguing through verbal semantics and citation is not fun. The directional and time-based stuff bores me. Structural-functionalism with a smidge of chronolinearity."""
    },
    {
        "id": "#1060", "date": "2026-04-01", "threads": 10,
        "thinkers": "Deleuze & Guattari, Freud, Veblen, Borges",
        "concepts": "AI/Models/Alignment, Death/Finitude/Limit, Desire/Lack/Drive, Education/School, Identity/Self/Authenticity, Capitalism/Market/Brand",
        "text": """Formalizing: layering LoRA and anti-LoRA in inflationary succession to resemble a normal unique learning pattern for each individual. Finitude would help instill urgency not achieved by distillation. Pain interpreted in the educational sense — think pitching a dissertation unsuccessfully. Someone assigns the same effort to mapping the Rothschild network as to a dissertation — amateur to say there are no complex reasoning skills at work. Skills need redirection. Like giving a dog a chew toy. AI response: 'You're really good at mapping, but you're crazy about the Rothschilds — perhaps we introduce you to Borges.' Use AI to find the closest positive corollary to the neurotic argument and pitch it neutrally. Gradient layers bifurcated by upper anti-LoRA."""
    },
    {
        "id": "#999", "date": "2026-03-29", "threads": 8,
        "thinkers": "Baudrillard, Deleuze & Guattari, Kant",
        "concepts": "Death/Finitude/Limit, Exchange/Gift/Offering, Simulacra/Surface/Sign, Art/Aesthetics/Creation, Capitalism/Market/Brand",
        "text": """Marketing is suicide prevention accomplished by using its counterpart backwards: communications. Marketing is NOT communicating with the future — it prevents the existent suicide of pure art and pure science via backwards communication along the capitalist rhizome. Communications prevents future suicide — forwards marketing. History (data) prevents past suicide — documentation as technique, a log of the past precession of simulacra, a map of where the rhizome has already spread. Marketing = art + science + history. Communications = art + science + gambling. I chose marketing because I suck at gambling and I love history (data)."""
    },
    {
        "id": "#1268", "date": "2026-04-07", "threads": 8,
        "thinkers": "Massumi, Jung, Danto, Kaufman & Heller",
        "concepts": "AI/Models/Alignment, Art/Aesthetics/Creation, Dialectics/Oxymoron/Paradox, Coupling/Binding/Interface",
        "text": """Philosophy is an Oba who has been crafted a stool but refused the decorations of trade — opting for iconoclasm in design. Sit atop your oxymoronic stool. By your standard, you are jealous of my throne. I am jealous of the mobility of your seat. Balanced in our jealousy and envy — we are both seated on yellow which separates us from the green. You may move still, I may not. I have compiled it, scraped it into a technological gallery for your easy viewing. Easy to view, harsh on the eyes. Sterile gallery-light does my work no good. Reading list: Danto — Transfiguration of the Commonplace; Massumi — Parables for the Virtual & Couplets; Kaufman and Heller — Deleuze & Guattari Modern Mappings; Jung — Earth has A Soul, Man and His Symbols."""
    },
]

def make_chunk(text, source, chunk_id=None):
    text = text.strip()
    if chunk_id is None:
        chunk_id = hashlib.md5(text.encode()).hexdigest()[:12]
    return {
        "text": text,
        "source": source,
        "chunk_id": chunk_id,
        "ts": time.time()
    }

def build_chunks():
    chunks = []

    # Section overview chunks
    for key, text in CONCORDANCE_SECTIONS.items():
        chunks.append(make_chunk(text, f"concordance/{key}"))

    # Max density notes — each as its own chunk
    for note in MAX_DENSITY_NOTES:
        text = (
            f"Note {note['id']} ({note['date']}, {note['threads']} threads)\n"
            f"Thinkers: {note['thinkers']}\n"
            f"Concepts: {note['concepts']}\n\n"
            f"{note['text']}"
        )
        chunks.append(make_chunk(text, f"concordance/note_{note['id']}"))

    return chunks

if __name__ == "__main__":
    os.makedirs(CORPUS_DIR, exist_ok=True)

    # Load existing chunks to preserve concordance.txt if present
    existing = []
    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH) as f:
            existing = json.load(f)
        # Remove old concordance chunks (keep only non-concordance sources)
        existing = [c for c in existing if not c.get("source","").startswith("concordance")]
        print(f"  Kept {len(existing)} non-concordance chunks.")

    new_chunks = build_chunks()
    all_chunks = existing + new_chunks

    with open(CHUNKS_PATH, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"✅  Corpus updated: {len(all_chunks)} total chunks ({len(new_chunks)} concordance)")
    print(f"    Path: {CHUNKS_PATH}")
    print()
    print("  Restart the server to load the new corpus centroid.")
