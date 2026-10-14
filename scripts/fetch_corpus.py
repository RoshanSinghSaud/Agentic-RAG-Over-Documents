"""Download the corpus into data/ — run at Docker build time.

Deliberately imports only src.corpus, never src.ingestion or src.config:
config.py exits the process if OPENAI_API_KEY is unset, and the build stage
has no secrets available to it. src.corpus has no such dependency.

    python scripts/fetch_corpus.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.corpus import fetch_corpus  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

if __name__ == "__main__":
    fetch_corpus(DATA_DIR)
