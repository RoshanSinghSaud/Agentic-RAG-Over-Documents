"""Load documents from data/, chunk them, and persist the Chroma vector store.

Run once (or whenever the corpus changes):
    python main.py ingest
"""
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config


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
