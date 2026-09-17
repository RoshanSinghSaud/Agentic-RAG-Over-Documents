"""Central configuration.

Every value here can be overridden by an environment variable of the same name
in upper case (CHROMA_DIR, TOP_K, ...). The defaults are the values this project
has always used, so running locally with no environment set behaves exactly as
before. Deployment targets override only what differs — usually just the paths.

OPENAI_API_KEY has no default. A missing key stops the process at import,
before uvicorn binds a port, rather than surfacing as an auth error on the
first request.
"""
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

# Kept deliberately: langchain-openai reads OPENAI_API_KEY straight from
# os.environ, not from this Settings object. In a container the platform sets
# real environment variables, so this is a no-op; locally it is what loads .env
# into os.environ. Remove it and the app starts fine, then fails on the first
# request with an authentication error.
load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Secrets ---
    openai_api_key: str                       # required: no default, no start without it
    tavily_api_key: str | None = None         # optional: web search degrades, app still serves
    api_key: str                              # required: gates /ask and /resume (see app.py)

    # --- Models ---
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # --- Paths (the ones that actually differ per machine) ---
    data_dir: Path = ROOT / "data"
    chroma_dir: Path = ROOT / ".chroma"
    checkpoint_db: Path = ROOT / "checkpoint.db"
    collection_name: str = "agentic_rag"

    # --- Chunking ---
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # --- Retrieval ---
    dense_k: int = 8        # candidates pulled from the vector store
    sparse_k: int = 8       # candidates pulled from BM25
    rrf_k: int = 60         # Reciprocal Rank Fusion constant (standard default)
    top_k: int = 6          # final chunks handed to the generator

    # --- Layer 2: CRAG (corrective retrieval) ---
    min_relevant_docs: int = 2      # fewer surviving chunks than this => retrieval is "weak"
    max_retrieval_loops: int = 2    # max transform_query -> retrieve cycles before web fallback
    web_search_k: int = 3           # Tavily results appended when we fall back to the web

    # --- Layer 3: Self-RAG (answer self-correction) ---
    max_generation_loops: int = 2   # max regenerations when the answer isn't grounded,
                                    # before returning the best-effort answer


try:
    settings = Settings()
except ValidationError as exc:
    missing = ", ".join(str(e["loc"][0]).upper() for e in exc.errors())
    sys.exit(f"Configuration error: required environment variable(s) not set: {missing}")


# Module-level names, unchanged from before this file was refactored, so nothing
# else in the codebase has to know that settings now come from the environment.
LLM_MODEL = settings.llm_model
EMBEDDING_MODEL = settings.embedding_model
API_KEY = settings.api_key

DATA_DIR = settings.data_dir
CHROMA_DIR = settings.chroma_dir
CHECKPOINT_DB = settings.checkpoint_db
COLLECTION_NAME = settings.collection_name

CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap

DENSE_K = settings.dense_k
SPARSE_K = settings.sparse_k
RRF_K = settings.rrf_k
TOP_K = settings.top_k

MIN_RELEVANT_DOCS = settings.min_relevant_docs
MAX_RETRIEVAL_LOOPS = settings.max_retrieval_loops
WEB_SEARCH_K = settings.web_search_k

MAX_GENERATION_LOOPS = settings.max_generation_loops
