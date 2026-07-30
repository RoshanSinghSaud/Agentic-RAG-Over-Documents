"""Load documents from data/, chunk them, and persist the Chroma vector store.

Run once (or whenever the corpus changes):
    python main.py ingest
"""
import shutil
import urllib.request
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config

# The locked corpus (see data/README.md) — arXiv IDs are stable, so this is a
# reproducible substitute for committing the PDFs to git.
_CORPUS = [
    ("attention-is-all-you-need.pdf", "1706.03762"),
    ("Retrieval-Augmented_Generation_RAG.pdf", "2005.11401"),
    ("Dense Passage Retrieval for Open-Domain Question Answering.pdf", "2004.04906"),
    ("REAC T- SYNERGIZING REASONING AND ACTING IN LANGUAGE MODELS.pdf", "2210.03629"),
    ("Lost in the Middle- How Language Models Use Long Contexts.pdf", "2307.03172"),
    ("HyDE.pdf", "2212.10496"),
    ("Corrective-RAG.pdf", "2401.15884"),
    ("Self-RAG.pdf", "2310.11511"),
]


def fetch_corpus(data_dir: Path = config.DATA_DIR) -> None:
    """Seed data/ with the locked corpus from arXiv, but only on a clean clone —
    if data_dir already holds any document, leave it alone (someone may be
    testing a deliberately different subset)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    has_docs = any(p.suffix.lower() in {".pdf", ".md", ".txt"} for p in data_dir.iterdir())
    if has_docs:
        return
    for filename, arxiv_id in _CORPUS:
        dest = data_dir / filename
        print(f"Fetching {filename} from arXiv ({arxiv_id})...")
        urllib.request.urlretrieve(f"https://arxiv.org/pdf/{arxiv_id}", dest)


def load_documents(data_dir: Path = config.DATA_DIR):
    """Load every PDF / markdown / text file under data/."""
    docs = []
    for path in sorted(data_dir.rglob("*")):
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


def ingest():
    shutil.rmtree(config.CHROMA_DIR, ignore_errors=True)
    fetch_corpus()
    raw = load_documents()
    if not raw:
        raise SystemExit(
            f"No documents found in {config.DATA_DIR}. "
            "Drop some PDFs/markdown there first (see data/README.md)."
        )
    chunks = chunk_documents(raw)
    build_vectorstore(chunks)
    print(
        f"Ingested {len(raw)} document pages -> {len(chunks)} chunks "
        f"into Chroma at {config.CHROMA_DIR}"
    )
    return chunks
