"""Central configuration. Everything tunable lives here so nodes stay clean."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Models (swappable via env) ---
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# --- Paths ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHROMA_DIR = ROOT / ".chroma"
COLLECTION_NAME = "agentic_rag"

# --- Chunking ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# --- Retrieval ---
DENSE_K = 8      # candidates pulled from the vector store
SPARSE_K = 8     # candidates pulled from BM25
RRF_K = 60       # Reciprocal Rank Fusion constant (standard default)
TOP_K = 5        # final chunks handed to the generator
