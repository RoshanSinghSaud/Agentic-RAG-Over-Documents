"""Shared pytest fixtures.

src/config.py calls sys.exit() at import time if OPENAI_API_KEY or API_KEY
is missing from the environment, which would fail collection before any test
runs. Set dummy values here, before any `src`/`app` import pulls config.py in
transitively. os.environ.setdefault() only fills gaps, so a real .env loaded
by config.py's own load_dotenv() still wins locally.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("API_KEY", "test-api-key")

import pytest

from src import retrieval


@pytest.fixture(autouse=True)
def _reset_retrieval_singletons():
    """src/retrieval.py caches the vector store, BM25 index, and doc list in
    module-level globals so repeated calls in one process don't rebuild them.
    That's fine in production but leaks a fake/stubbed index from one test
    into the next here, so reset them after every test."""
    yield
    retrieval._vectorstore = None
    retrieval._bm25 = None
    retrieval._all_docs = None


@pytest.fixture(autouse=True)
def _reset_rate_limit():
    """app.py tracks request timestamps per IP in an in-memory dict to
    enforce 10 req/min. Left alone, enough API tests in one session start
    tripping 429s that have nothing to do with what's under test."""
    yield
    try:
        from app import _request_times
        _request_times.clear()
    except ImportError:
        pass
