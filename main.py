"""CLI entry point for the agentic-rag baseline (Layer 1).

Usage:
    python main.py ingest                  # build the index from data/

    op: Ingested 156 document pages -> 718 chunks into Chroma at /Users/Roshan_Singh_Saud/Claude/Projects/Agentic_RAG/.chroma
    
    
    python main.py ask "your question"     # query the baseline RAG
"""
import sys

from src import ingestion
from src.graph import graph


def run_query(question: str):
    result = graph.invoke({"question": question})

    print("\n=== ANSWER ===\n")
    print(result["generation"])

    print("\n=== SOURCES (retrieved chunks) ===\n")
    for i, d in enumerate(result["documents"], start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        loc = src + (f", p.{page}" if page is not None else "")
        print(f"[{i}] {loc}")


def main():
    args = sys.argv[1:]
    if args and args[0] == "ingest":
        ingestion.ingest()
        return
    if args and args[0] == "ask":
        question = " ".join(args[1:]).strip() or input("Question: ").strip()
        run_query(question)
        return
    print(__doc__)


if __name__ == "__main__":
    main()
