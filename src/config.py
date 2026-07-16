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

# --- Layer 2: CRAG (corrective retrieval) ---
MIN_RELEVANT_DOCS = 2    # fewer surviving chunks than this => retrieval is "weak"
MAX_RETRIEVAL_LOOPS = 2  # max transform_query -> retrieve cycles before web fallback
WEB_SEARCH_K = 3         # Tavily results appended when we fall back to the web

# --- Layer 3: Self-RAG (answer self-correction) ---
MAX_GENERATION_LOOPS = 2  # max regenerations when the answer isn't grounded (hallucination)
                          # before we stop and return the best-effort answer
