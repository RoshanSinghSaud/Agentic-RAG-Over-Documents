"""The locked corpus (see data/README.md) and how to fetch it.

Deliberately stdlib-only, no `config` import: `config.py` fails at import time
without OPENAI_API_KEY, and this module also gets run standalone at Docker
build time (scripts/fetch_corpus.py), before any secrets exist.
"""
import urllib.request
from pathlib import Path

# arXiv IDs are stable, so this is a reproducible substitute for committing
# the PDFs to git.
CORPUS = [
    ("attention-is-all-you-need.pdf", "1706.03762"),
    ("Retrieval-Augmented_Generation_RAG.pdf", "2005.11401"),
    ("Dense Passage Retrieval for Open-Domain Question Answering.pdf", "2004.04906"),
    ("REAC T- SYNERGIZING REASONING AND ACTING IN LANGUAGE MODELS.pdf", "2210.03629"),
    ("Lost in the Middle- How Language Models Use Long Contexts.pdf", "2307.03172"),
    ("HyDE.pdf", "2212.10496"),
    ("Corrective-RAG.pdf", "2401.15884"),
    ("Self-RAG.pdf", "2310.11511"),
]


def fetch_corpus(data_dir: Path) -> None:
    """Download whichever CORPUS files are missing from data_dir.

    Checked by name, not "does the folder hold any documents at all" — a
    clean clone already ships data/README.md (tracked in git; the PDFs
    aren't), so an any-docs check would see that one file and skip fetching
    every paper.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename, arxiv_id in CORPUS:
        dest = data_dir / filename
        if dest.exists():
            continue
        print(f"Fetching {filename} from arXiv ({arxiv_id})...")
        urllib.request.urlretrieve(f"https://arxiv.org/pdf/{arxiv_id}", dest)
