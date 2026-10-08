"""Load documents from data/, chunk them, and persist the Chroma vector store.

Run once (or whenever the corpus changes):
    python main.py ingest
"""
import logging
import shutil
import tarfile
import tempfile
import time
from pathlib import Path

from chromadb.api.shared_system_client import SharedSystemClient
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .corpus import fetch_corpus

logger = logging.getLogger(__name__)


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
        # chromadb caches its client system per persist_directory for the life
        # of the process (SharedSystemClient). Deleting the directory doesn't
        # invalidate that cache, so a second ingest() in the same process
        # reused a connection pointing at now-gone files and every write
        # failed with "attempt to write a readonly database". Clearing the
        # cache forces the next Chroma(...) below to open a fresh connection.
        SharedSystemClient.clear_system_cache()
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


def _download(uri: str, dest: Path) -> None:
    """Fetch s3://bucket/key to dest. Its own function so tests can swap it out."""
    import boto3  # lazy: Render never sets INDEX_S3_URI, so it never pays this import

    bucket, _, key = uri.removeprefix("s3://").partition("/")
    boto3.client("s3").download_file(bucket, key, str(dest))


def _restore_from_s3(uri: str) -> bool:
    """Download a prebuilt index tarball and install it as CHROMA_DIR.

    The archive must contain a top-level ``.chroma/`` (built with
    ``tar czf ... -C <state> .chroma``) that carries the completion marker —
    an archive of a half-built index is rejected, same rule as a local one.
    Returns False on any failure so the caller can fall back to a full build;
    the failure is logged at ERROR, which the A3 CloudWatch alarm catches.
    """
    target = config.CHROMA_DIR
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
            tmp = Path(tmp)
            archive = tmp / "index.tgz"
            _download(uri, archive)
            with tarfile.open(archive) as tar:
                tar.extractall(tmp / "x", filter="data")  # blocks ../ and absolute paths
            restored = tmp / "x" / ".chroma"
            if not (restored / _MARKER_NAME).exists():
                raise RuntimeError(f"{uri} has no {_MARKER_NAME}; refusing a partial index")
            shutil.rmtree(target, ignore_errors=True)
            shutil.move(str(restored), str(target))
        return True
    except Exception:
        logger.exception("index restore from %s failed; falling back to a full ingest", uri)
        return False


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
    if config.INDEX_S3_URI and _restore_from_s3(config.INDEX_S3_URI):
        print(f"Index restored from {config.INDEX_S3_URI} in {time.perf_counter() - start:.1f}s")
        return
    ingest(rebuild=True)
    print(f"Index build took {time.perf_counter() - start:.1f}s")
