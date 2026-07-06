"""CLI entry point for agentic-rag (Layer 2: CRAG).

Usage:
    python main.py ingest                  # build the index from data/
    python main.py ask "your question"     # query the CRAG graph

Layer 2 requires TAVILY_API_KEY in .env (web-search fallback) in addition to
OPENAI_API_KEY. The graph now grades its own retrieval: watch the ---
GRADE/TRANSFORM/WEB SEARCH --- lines to see the corrective loop fire.
"""
import sys

from src import ingestion
from src.graph import graph


def run_query(question: str):
    # loop_count starts at 0; grade/transform/web_search manage it from there.
    result = graph.invoke({"question": question, "loop_count": 0})

    if result["question"] != question:
        print(f"\n(question was rewritten to: {result['question']!r})")

    print("\n=== ANSWER ===\n")
    print(result["generation"])

    print("\n=== SOURCES (context handed to the generator) ===\n")
    for i, d in enumerate(result["documents"], start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        loc = src + (f", p.{page}" if page is not None else "")
        tag = " [web]" if d.metadata.get("web") else ""
        print(f"[{i}] {loc}{tag}")


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
