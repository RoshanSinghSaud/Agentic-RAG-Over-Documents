"""Regression test for the 4x-ingestion-duplication bug (README: v0.4 vs
v0.4-fixed). `ingest()` used to append onto the existing Chroma collection
instead of replacing it — four `python main.py ingest` runs quietly grew the
index from 718 to 2872 chunks, and RRF's dedup starved the generator of
context. The fix added a `shutil.rmtree(config.CHROMA_DIR)` at the top of
`ingest()`, but nothing exercises "ingest more than once" in CI, so a fresh
regression here would go undetected until `/ready` reports an absurd chunk
count in prod.
"""
from pathlib import Path

from langchain_core.embeddings import DeterministicFakeEmbedding

from src import config, ingestion, retrieval

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_ingest_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", FIXTURES_DIR)
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path)

    # load_documents(data_dir: Path = config.DATA_DIR) binds its default at
    # import time, long before this test patches config.DATA_DIR, so the
    # patch above never reaches it. Force the real function onto the
    # fixtures dir instead of relying on the stale default.
    real_load_documents = ingestion.load_documents
    monkeypatch.setattr(ingestion, "load_documents", lambda: real_load_documents(FIXTURES_DIR))

    # Never hit the network in a test: fetch_corpus downloads 8 arXiv PDFs,
    # and the real OpenAIEmbeddings would call the embeddings API.
    monkeypatch.setattr(ingestion, "fetch_corpus", lambda *a, **kw: None)
    monkeypatch.setattr(
        ingestion,
        "OpenAIEmbeddings",
        lambda **kwargs: DeterministicFakeEmbedding(size=32),
    )

    ingestion.ingest()
    first_count = retrieval.index_size()

    ingestion.ingest()
    retrieval._vectorstore = None  # index_size() caches the store; force a re-read
    second_count = retrieval.index_size()

    assert second_count == first_count, (
        f"ingest() is not idempotent: {first_count} chunks became "
        f"{second_count} on a second run"
    )
