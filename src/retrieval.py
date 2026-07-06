"""Hybrid retrieval: dense (Chroma) + sparse (BM25), fused with Reciprocal Rank Fusion.

The BM25 index is rebuilt in-memory from the documents already stored in Chroma,
so there's a single source of truth (the persisted vector store) and no second
on-disk index to keep in sync.
"""
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from . import config

# Module-level singletons so we load Chroma / build BM25 only once per process.
_vectorstore = None
_bm25 = None
_all_docs = None


def _load_chroma() -> Chroma:
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )


def _all_documents_from_chroma(vs: Chroma):
    raw = vs.get()  # -> {'ids', 'documents', 'metadatas', ...}
    return [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]


def _ensure_loaded():
    global _vectorstore, _bm25, _all_docs
    if _vectorstore is not None:
        return
    _vectorstore = _load_chroma()
    _all_docs = _all_documents_from_chroma(_vectorstore)
    if not _all_docs:
        raise SystemExit("Chroma is empty. Run `python main.py ingest` first.")
    _bm25 = BM25Retriever.from_documents(_all_docs)
    _bm25.k = config.SPARSE_K


def _doc_key(doc: Document) -> str:
    """A stable identity for de-duplicating the same chunk across both retrievers."""
    src = doc.metadata.get("source", "")
    idx = doc.metadata.get("start_index", "")
    return f"{src}::{idx}::{hash(doc.page_content)}"


def reciprocal_rank_fusion(ranked_lists, k: int = config.RRF_K):
    """Fuse multiple ranked lists. RRF score = sum over lists of 1/(k + rank)."""
    scores: dict[str, float] = {}
    registry: dict[str, Document] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = _doc_key(doc)
            registry[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [registry[key] for key, _ in ordered]


def hybrid_retrieve(question: str, top_k: int = config.TOP_K):
    """Return the top_k chunks after dense + sparse retrieval and RRF fusion."""
    _ensure_loaded()
    dense = _vectorstore.similarity_search(question, k=config.DENSE_K)
    sparse = _bm25.invoke(question)
    fused = reciprocal_rank_fusion([dense, sparse])
    # NOTE (Layer 1 polish hook): a cross-encoder reranker would slot in here,
    # re-scoring `fused` before the top_k cut. RRF ordering is the v1 default.
    return fused[:top_k]
