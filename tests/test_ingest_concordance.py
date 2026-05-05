"""Characterization tests for ingest_concordance.py."""

from ingest_concordance import build_chunks, make_chunk, CONCORDANCE_SECTIONS, MAX_DENSITY_NOTES


class TestBuildChunks:
    def test_builds_chunks(self):
        chunks = build_chunks()
        assert len(chunks) > 0

    def test_section_count(self):
        chunks = build_chunks()
        section_chunks = [c for c in chunks if not c["source"].startswith("concordance/note_")]
        assert len(section_chunks) == len(CONCORDANCE_SECTIONS)

    def test_note_count(self):
        chunks = build_chunks()
        note_chunks = [c for c in chunks if c["source"].startswith("concordance/note_")]
        assert len(note_chunks) == len(MAX_DENSITY_NOTES)

    def test_chunk_structure(self):
        chunks = build_chunks()
        for c in chunks:
            assert "text" in c
            assert "source" in c
            assert "chunk_id" in c
            assert "ts" in c


class TestMakeChunk:
    def test_creates_chunk(self):
        c = make_chunk("test text", "test/source")
        assert c["text"] == "test text"
        assert c["source"] == "test/source"
        assert len(c["chunk_id"]) == 12

    def test_custom_chunk_id(self):
        c = make_chunk("text", "src", chunk_id="custom123")
        assert c["chunk_id"] == "custom123"
