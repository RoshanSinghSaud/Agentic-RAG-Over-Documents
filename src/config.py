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
from pydantic import Field, ValidationError
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

    index_s3_uri: str | None = None  # s3://bucket/key of a prebuilt index tarball; unset = build locally
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
    # s3://bucket/key of a prebuilt index tarball (A4). Unset = build locally,
    # which is what Render does, so the live demo never touches S3.
    index_s3_uri: str | None = None

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

    # --- Layer 8 Day 21: cost guardrails ---
    # The two cost levers ship OFF. Measured on the 20-question golden set
    # (eval/results/summary_day21-*.md): batched grading -26% cost, routing
    # +batching -31%, but faithfulness 0.98 -> 0.92 and the web-search gate
    # fires less often. At ~$1 per 1,000 questions, grounding wins. One env
    # var turns either on.
    small_llm_model: str = "gpt-5-nano"  # answers simple questions when routing is on
    model_routing: bool = False          # True: simple questions answered by small_llm_model
    grade_batch: bool = False            # True: one grading call per round, not one per chunk
    # Global budget on node executions per invoke, above the two loop caps.
    # The longest legitimate path is 17 nodes, so 20 never cuts a normal run;
    # it stops a runaway one (e.g. a mis-set loop cap) with an answer, not a 500.
    max_graph_steps: int = Field(20, ge=10)
    session_token_limit: int = 100_000   # per thread_id; /ask refuses past it


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
INDEX_S3_URI = settings.index_s3_uri

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

SMALL_LLM_MODEL = settings.small_llm_model
MODEL_ROUTING = settings.model_routing
GRADE_BATCH = settings.grade_batch
MAX_GRAPH_STEPS = settings.max_graph_steps
SESSION_TOKEN_LIMIT = settings.session_token_limit
