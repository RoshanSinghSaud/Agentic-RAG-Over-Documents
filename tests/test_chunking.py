"""Property tests for chunk_documents(), plus the README skip rule in
load_documents(). Deliberately NOT asserting an exact chunk count: local runs
have produced 716 or 718 chunks and Render has produced 692, all off the same
`data/` folder, because arXiv serves different PDF revisions over time. A
count-based test would go red on a paper repost that changes nothing about
the code -- assert the properties the rest of the app actually depends on
instead.
"""
from pathlib import Path

from langchain_core.documents import Document

from src import config, ingestion

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_chunk_documents_produces_valid_chunks(monkeypatch):
    monkeypatch.setattr(config, "CHUNK_SIZE", 200)
    monkeypatch.setattr(config, "CHUNK_OVERLAP", 50)

    long_text = "".join(f"sentence number {i} of the synthetic test document. " for i in range(80))
    doc = Document(page_content=long_text, metadata={"source": "synthetic.txt"})

    chunks = ingestion.chunk_documents([doc])

    assert len(chunks) > 1

    for chunk in chunks:
        assert len(chunk.page_content) <= config.CHUNK_SIZE
        assert "start_index" in chunk.metadata  # citation keys depend on this

    for prev, nxt in zip(chunks, chunks[1:]):
        prev_end = prev.metadata["start_index"] + len(prev.page_content)
        assert nxt.metadata["start_index"] < prev_end, "consecutive chunks don't overlap"


def test_load_documents_skips_readme(tmp_path):
    # README.md is corpus-folder documentation, not a paper to answer
    # questions from -- load_documents() is supposed to skip it by name.
    # This exact rule is what keeps the Render build from trying to answer
    # questions about its own data/README.md.
    (tmp_path / "README.md").write_text("This describes the corpus, it is not a paper.")
    (tmp_path / "note.txt").write_text((FIXTURES_DIR / "rag_notes.txt").read_text())

    docs = ingestion.load_documents(data_dir=tmp_path)

    assert len(docs) == 1
    assert docs[0].metadata["source"] == str(tmp_path / "note.txt")
