"""CLI entry point for agentic-rag.

Usage:
    python main.py ingest                  # build the index from data/
    python main.py ask [thread_id]         # interactive, memory-backed session

Conversations are checkpointed to SQLite per thread_id, so re-running
`python main.py ask demo` resumes the `demo` thread — including after a
process restart. Omitting thread_id uses the thread "default".

When retrieval stays weak after the rewrite budget is spent, the graph pauses
at a human approval gate before falling back to a web search — answer y/n at
the prompt. Requires OPENAI_API_KEY (+ TAVILY_API_KEY for the web fallback)
in .env.
"""
import sys
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph.types import Command

from src import config, ingestion
from src.graph import build_graph
from src.nodes import ABSTENTION

from langchain_core.messages import HumanMessage


conn = sqlite3.connect(str(config.CHECKPOINT_DB), check_same_thread=False)
saver = SqliteSaver(conn)
graph = build_graph(checkpointer=saver)


def run_query(question: str, run_config: dict):
    # State is thread-scoped under a checkpointer, but these fields are
    # *turn*-scoped, so every invoke resets them explicitly:
    #   - loop_count / generate_count: correction budgets. Without the reset,
    #     one budget-exhausting turn would permanently disable the corrective
    #     loops for the rest of the thread.
    #   - web_search / web_search_approved: routing flags. Each happens to be
    #     overwritten before it's read again, but resetting them here keeps
    #     that invariant explicit instead of depending on node order.
    # (`messages` is genuinely thread-scoped and appends via add_messages;
    # the answer side is appended by the graph's `finalize` node.)
    result = graph.invoke(
        {
            "question": question,
            "messages": [HumanMessage(content=question)],
            "loop_count": 0,
            "generate_count": 0,
            "web_search": False,
            "web_search_approved": False,
        },
        config=run_config,
    )

    # Layer 4 HITL: the graph paused at human_approval_gate. Surface the
    # payload, collect a decision, and resume. Today the gate can fire at most
    # once per turn (every path back to it needs rewrite budget, which is
    # already exhausted when it fires) — the `while` is insurance against
    # future rewiring, not a live loop.
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print(f"\n[APPROVAL NEEDED] {payload['reason']}")
        print(f"Query: {payload['question']}")
        answer = input(f"{payload['instruction']} ").strip()
        result = graph.invoke(Command(resume=answer), config=run_config)

    if result["question"] != question:
        print(f"\n(question was rewritten to: {result['question']!r})")

    print("\n=== ANSWER ===\n")
    print(result["generation"])

    # Layer 3: surface the self-grade so the demo shows the answer was checked.
    grounded = result.get("answer_grounded")
    useful = result.get("answer_useful")
    if grounded is not None:
        if ABSTENTION.lower() in result["generation"].lower():
            # Accepted, but don't label an "I don't know" as a passing answer.
            verdict = "honest abstention (corpus can't answer this)"
        elif grounded and useful:
            verdict = "passed (grounded + addresses the question)"
        else:
            verdict = "best effort (correction budget exhausted)"
        # generate_count counts *failed grounding checks*; the last failure ends
        # the run rather than triggering another regeneration, so retries != N
        # regenerations — label it by what it actually measures.
        retries = result.get("generate_count", 0)
        print(f"\n=== SELF-CHECK: {verdict}"
              f"{f' after {retries} failed grounding check(s)' if retries else ''} ===")

    print("\n=== SOURCES (context handed to the generator) ===\n")
    if not result["documents"]:
        print("(none — no relevant corpus chunks, and no web results)")
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
        # A stable thread_id is what makes memory resumable: the same id on a
        # later run (even after a restart) picks the conversation back up from
        # its checkpoints. Pass one explicitly to keep threads separate.
        thread_id = args[1] if len(args) > 1 else "default"
        run_config = {"configurable": {"thread_id": thread_id}}
        print(f"[thread: {thread_id}]")
        while True:
            question = input("\nQuestion (or 'quit'): ").strip()
            if not question:
                continue
            if question.lower() == "quit":
                break
            run_query(question, run_config)
        return
    print(__doc__)


if __name__ == "__main__":
    main()
