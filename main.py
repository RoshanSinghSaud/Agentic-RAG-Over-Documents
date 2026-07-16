"""CLI entry point for agentic-rag (Layer 3: Self-RAG).

Usage:
    python main.py ingest                  # build the index from data/
    python main.py ask "your question"     # query the Self-RAG graph

Layer 3: every generated answer is now graded before it's returned — first for
grounding (did the model hallucinate beyond the retrieved chunks?), then for
usefulness (does it actually address the question?). Watch the
--- GRADE ANSWER / SELF-RAG --- lines: a hallucinated answer triggers a
regeneration, an off-topic answer triggers a query rewrite and re-retrieval.
Requires OPENAI_API_KEY (+ TAVILY_API_KEY for the web-search fallback) in .env.
"""
import sys

from src import ingestion
from src.graph import graph


def run_query(question: str):
    # Counters start at 0; the graders and routers manage them from there.
    result = graph.invoke(
        {"question": question, "loop_count": 0, "generate_count": 0}
    )

    if result["question"] != question:
        print(f"\n(question was rewritten to: {result['question']!r})")

    print("\n=== ANSWER ===\n")
    print(result["generation"])

    # Layer 3: surface the self-grade so the demo shows the answer was checked.
    grounded = result.get("answer_grounded")
    useful = result.get("answer_useful")
    if grounded is not None:
        verdict = (
            "passed (grounded + addresses the question)"
            if grounded and useful
            else "best effort (correction budget exhausted)"
        )
        # generate_count counts *failed grounding checks*; the last failure ends
        # the run rather than triggering another regeneration, so retries != N
        # regenerations — label it by what it actually measures.
        retries = result.get("generate_count", 0)
        print(f"\n=== SELF-CHECK: {verdict}"
              f"{f' after {retries} failed grounding check(s)' if retries else ''} ===")

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
