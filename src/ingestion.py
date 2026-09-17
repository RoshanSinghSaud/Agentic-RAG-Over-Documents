"""Load documents from data/, chunk them, and persist the Chroma vector store.

Run once (or whenever the corpus changes):
    python main.py ingest
"""
import shutil
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .corpus import fetch_corpus


def load_documents(data_dir: Path = config.DATA_DIR):
    """Load every PDF / markdown / text file under data/, except README.md —
    that's corpus-folder documentation, not a paper to answer questions from."""
    docs = []
    for path in sorted(data_dir.rglob("*")):
        if path.name == "README.md":
            continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            docs.extend(PyPDFLoader(str(path)).load())
        elif suffix in {".md", ".txt"}:
            docs.extend(TextLoader(str(path), encoding="utf-8").load())
    return docs


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        add_start_index=True,  # records each chunk's offset -> stable citation keys
    )
    return splitter.split_documents(docs)


def build_vectorstore(chunks):
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.CHROMA_DIR),
    )


# Written only after a run completes successfully, so its presence — not
# index_size() — is the source of truth for "the index is usable". A run that
# dies mid-embedding (dropped connection, rate limit) leaves partial chunks in
# Chroma with no marker; ensure_index() treats that the same as empty and
# rebuilds, rather than serving off of a corpus that stopped partway through.
_MARKER_NAME = ".ingest_complete"


def _marker_path() -> Path:
    return config.CHROMA_DIR / _MARKER_NAME


def _timed(label: str, fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    print(f"  {label}: {time.perf_counter() - start:.1f}s")
    return result


def ingest(rebuild: bool = True):
    if rebuild:
        shutil.rmtree(config.CHROMA_DIR, ignore_errors=True)
    _timed("fetch_corpus", fetch_corpus, config.DATA_DIR)
    raw = _timed("load_documents", load_documents)
    if not raw:
        raise SystemExit(
            f"No documents found in {config.DATA_DIR}. "
            "Drop some PDFs/markdown there first (see data/README.md)."
        )
    chunks = _timed("chunk_documents", chunk_documents, raw)
    _timed("build_vectorstore", build_vectorstore, chunks)
    _marker_path().touch()
    print(
        f"Ingested {len(raw)} document pages -> {len(chunks)} chunks "
        f"into Chroma at {config.CHROMA_DIR}"
    )
    return chunks


def ensure_index() -> None:
    """Build the index if it's missing or incomplete; leave a good one alone.

    Called from the FastAPI lifespan so startup blocks until the store is
    populated, instead of serving requests against an empty or half-built
    index. No marker means either a clean-clone empty store or a prior ingest
    that never finished — both get a full rebuild, since partial chunks
    already sitting in Chroma aren't safe to just add more documents on top of.
    """
    if _marker_path().exists():
        return
    start = time.perf_counter()
    ingest(rebuild=True)
    print(f"Index build took {time.perf_counter() - start:.1f}s")
