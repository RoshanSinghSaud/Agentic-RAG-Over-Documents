"""hybrid_retrieve() fuses dense (Chroma) + sparse (BM25) results and must
never hand the generator more than TOP_K chunks, or a doc that both
retrievers agree on twice.

_ensure_loaded() only rebuilds Chroma/BM25 when `_bm25 is None`, so setting
the module globals directly to fakes short-circuits it — no real vector
store, no real embeddings call, no real OpenAI key needed.
"""
from langchain_core.documents import Document

from src import config, retrieval


class _FakeVectorStore:
    def __init__(self, docs):
        self._docs = docs

    def similarity_search(self, question, k):
        return self._docs


class _FakeBM25:
    def __init__(self, docs):
        self._docs = docs

    def invoke(self, question):
        return self._docs


def _doc(source: str, start_index: int, content: str) -> Document:
    return Document(page_content=content, metadata={"source": source, "start_index": start_index})


def test_hybrid_retrieve_caps_at_top_k_and_dedupes_overlap(monkeypatch):
    # 4 chunks both retrievers agree on (ranked #1-4 in each list), plus 4
    # chunks only dense finds and 4 only sparse finds -> 12 unique candidates,
    # comfortably more than TOP_K, so truncation actually has to happen.
    #
    # The overlap is built as two SEPARATE Document objects per chunk (same
    # source/start_index/content, different instances) rather than one
    # shared object reused in both lists. That's what the real dense and
    # sparse retrievers do -- each constructs its own Document for the same
    # underlying chunk -- and it's the case that actually exercises
    # _doc_key's content-based dedup instead of accidentally passing via
    # object identity.
    overlap_dense = [_doc("shared", i, f"overlap {i}") for i in range(4)]
    overlap_sparse = [_doc("shared", i, f"overlap {i}") for i in range(4)]
    dense_only = [_doc("dense", i, f"dense only {i}") for i in range(4)]
    sparse_only = [_doc("sparse", i, f"sparse only {i}") for i in range(4)]

    monkeypatch.setattr(retrieval, "_bm25", _FakeBM25(overlap_sparse + sparse_only))
    monkeypatch.setattr(retrieval, "_vectorstore", _FakeVectorStore(overlap_dense + dense_only))

    # hybrid_retrieve(question, top_k=config.TOP_K) binds its default at
    # import time (same gotcha as load_documents in ingestion.py), so a
    # monkeypatch of config.TOP_K alone wouldn't reach it. Pass it through
    # explicitly so the assertion below is checking the real config value,
    # not a stale one.
    monkeypatch.setattr(config, "TOP_K", 5)
    result = retrieval.hybrid_retrieve("does not matter", top_k=config.TOP_K)

    assert len(result) == config.TOP_K

    # No chunk (identified by source + start_index, independent of which
    # retriever's Document object represents it) may appear twice in the
    # final list -- that's the dedup _doc_key exists to guarantee.
    identities = [(d.metadata["source"], d.metadata["start_index"]) for d in result]
    assert len(identities) == len(set(identities))

    # Every chunk both retrievers ranked in their top 4 outscores every
    # single-retriever chunk under RRF, so all 4 overlap chunks must survive
    # the cut to TOP_K=5.
    overlap_identities = {("shared", i) for i in range(4)}
    assert overlap_identities <= set(identities)


def test_reciprocal_rank_fusion_prefers_docs_ranked_first_in_both_lists():
    """Pure unit test, no mocks/I/O: documents the fusion rule itself."""
    # Two separate objects for "the same" chunk, as the two retrievers would
    # each produce -- see the comment in the hybrid_retrieve test above.
    consensus_from_dense = _doc("a", 0, "both retrievers agree")
    consensus_from_sparse = _doc("a", 0, "both retrievers agree")
    dense_only_top = _doc("b", 0, "only dense ranks this #1")
    sparse_only_top = _doc("c", 0, "only sparse ranks this #1")

    dense_ranked = [consensus_from_dense, dense_only_top]
    sparse_ranked = [consensus_from_sparse, sparse_only_top]

    fused = retrieval.reciprocal_rank_fusion([dense_ranked, sparse_ranked])

    # The chunk ranked #1 in both lists gets the sum of two top-rank
    # scores, so it must outrank a chunk that's #1 in only one list.
    assert (fused[0].metadata["source"], fused[0].metadata["start_index"]) == ("a", 0)
    assert len(fused) == 3  # the consensus chunk deduped to one entry, not two
