"""Eval harness: run the golden Q&A set through the graph and score it.

Usage (from the repo root):
    python -m eval.run_eval --label v0.4            # full run: graph + RAGAS
    python -m eval.run_eval --label v0.4 --skip-ragas   # behavior checks only
    python -m eval.run_eval --only q05 --skip-ragas     # debug one question
    python -m eval.run_eval --label v0.4-web --approve-web  # auto-APPROVE the gate

Outputs land in eval/results/: results_<label>.csv (one row per question)
and summary_<label>.md (per-category table, ready to paste into the README).

Design decisions (the parts an interviewer will ask about):

1. HITL gate policy — an automated eval can't wait at human_approval_gate, so
   every interrupt is auto-resolved: "n" by default (deny-by-default, same as
   the gate's own fail-closed rule), "y" with --approve-web. A `gate_fired`
   column records that the graph *asked*, which is itself a graded behavior
   for needs_web questions.

2. `<ABSTENTION>` placeholder — golden_set.json never hardcodes the abstention
   sentence; it's substituted from src.nodes.ABSTENTION at load time so the
   golden set can't drift out of sync with the prompt.

3. RAGAS metrics are skipped where they're not meaningful: faithfulness needs
   a non-abstention answer plus retrieved contexts; context precision/recall
   need expected corpus sources. Scoring an honest "I don't know" for
   faithfulness would punish correct behavior.

4. Fresh MemorySaver thread per question — single-turn eval, no cross-question
   contamination, and the on-disk checkpoint.db is never touched.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import warnings
from pathlib import Path

# --- repo-root imports work whether run as `python -m eval.run_eval` or
# `python eval/run_eval.py` ---
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# =========================================================================
# RAGAS compatibility shim
# =========================================================================
# ragas (<= 0.4.3) does `from langchain_community.chat_models.vertexai import
# ChatVertexAI` at import time, but langchain-community 0.4.x (the langchain
# 1.x line this project uses) removed that legacy module — so `import ragas`
# crashes before anything runs. We never use Vertex AI: inject an inert
# placeholder module so the import succeeds. ragas 0.4's modern metrics
# (ragas.metrics.collections) run their LLM calls through instructor+openai
# directly, not langchain, so nothing else in ragas touches the removed code.
# Remove this shim once ragas ships a release compatible with langchain 1.x.
def _shim_legacy_vertexai() -> None:
    import types

    try:
        import langchain_community.chat_models.vertexai  # noqa: F401
        return  # real module exists; nothing to do
    except ImportError:
        pass
    mod = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # inert sentinel; isinstance() checks are always False
        pass

    mod.ChatVertexAI = ChatVertexAI
    sys.modules[mod.__name__] = mod

    import langchain_community.llms as llms_mod  # noqa: PLC0415

    if not hasattr(llms_mod, "VertexAI"):
        class VertexAI:  # same trick for `from langchain_community.llms import VertexAI`
            pass

        llms_mod.VertexAI = VertexAI


# =========================================================================
# Running the graph over the golden set
# =========================================================================

GOLDEN_PATH = ROOT / "eval" / "golden_set.json"
RESULTS_DIR = ROOT / "eval" / "results"


def load_golden(path: Path, abstention: str) -> list[dict]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    for e in entries:
        if e["ground_truth"] == "<ABSTENTION>":
            e["ground_truth"] = abstention
    return entries


def run_one(graph, entry: dict, approve_web: bool, abstention: str) -> dict:
    """Run a single golden question through the graph on its own thread."""
    from langchain_core.messages import HumanMessage
    from langgraph.types import Command

    run_config = {"configurable": {"thread_id": f"eval-{entry['id']}"}}
    result = graph.invoke(
        {
            "question": entry["question"],
            "messages": [HumanMessage(content=entry["question"])],
            "loop_count": 0,
            "generate_count": 0,
            "web_search": False,
            "web_search_approved": False,
        },
        config=run_config,
    )

    gate_fired = False
    while "__interrupt__" in result:  # auto-resolve the HITL gate
        gate_fired = True
        decision = "y" if approve_web else "n"
        result = graph.invoke(Command(resume=decision), config=run_config)

    answer = result.get("generation", "")
    docs = result.get("documents", [])
    sources = [d.metadata.get("source", "unknown") for d in docs]
    return {
        "answer": answer,
        "contexts": [d.page_content for d in docs],
        "sources": sources,
        "web_sources": [s for d, s in zip(docs, sources) if d.metadata.get("web")],
        "gate_fired": gate_fired,
        "abstained": abstention.lower() in answer.lower(),
        "rewritten": result.get("question", entry["question"]) != entry["question"],
        "loop_count": result.get("loop_count", 0),
        "generate_count": result.get("generate_count", 0),
        "answer_grounded": result.get("answer_grounded"),
        "answer_useful": result.get("answer_useful"),
    }


# =========================================================================
# Deterministic behavior checks (no LLM involved)
# =========================================================================

def citation_hits(expected_stems: list[str], sources: list[str]) -> tuple[int, int]:
    """How many expected source stems appear (case-insensitive substring)
    among the retrieved documents' source paths?"""
    joined = " || ".join(sources).lower()
    hits = sum(1 for stem in expected_stems if stem.lower() in joined)
    return hits, len(expected_stems)


def behavior_pass(entry: dict, run: dict, approve_web: bool, abstention: str) -> bool:
    """Category-specific definition of 'the graph did the right thing'."""
    cat = entry["category"]
    expects_abstention = entry["ground_truth"] == abstention

    if cat in ("lookup", "multi_hop"):
        hits, total = citation_hits(entry["expected_sources"], run["sources"])
        return (not run["abstained"]) and hits == total
    if cat == "no_answer":
        return run["abstained"]
    if cat == "needs_web":
        if approve_web:
            return run["gate_fired"] and bool(run["web_sources"]) and not run["abstained"]
        return run["gate_fired"] and run["abstained"]
    if cat == "known_failure":
        # Graded by what the ground truth demands; the interesting signal for
        # these rows is usually in the RAGAS columns + notes, not pass/fail.
        if expects_abstention:
            return run["abstained"]
        hits, total = citation_hits(entry["expected_sources"], run["sources"])
        return (not run["abstained"]) and hits == total
    raise ValueError(f"unknown category: {cat}")


# =========================================================================
# RAGAS scoring (modern 0.4 collections API: async, instructor+openai)
# =========================================================================

async def ragas_scores(rows: list[dict], model: str) -> None:
    """Mutates each row in place, adding metric columns (None = skipped)."""
    from ragas.llms import llm_factory
    from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
    from ragas.metrics.collections import (
        AnswerCorrectness,
        ContextPrecisionWithReference,
        ContextRecall,
        Faithfulness,
    )
    import openai

    # Metrics run async (ascore), so both wrappers get the async client.
    # max_tokens: faithfulness decomposes long answers into many statements —
    # the instructor default is too small and raises IncompleteOutputException.
    client = openai.AsyncOpenAI()
    llm = llm_factory(model, client=client, max_tokens=8192)
    emb = RagasOpenAIEmbeddings(client=client)
    m_faith = Faithfulness(llm=llm)
    m_correct = AnswerCorrectness(llm=llm, embeddings=emb)
    m_ctx_prec = ContextPrecisionWithReference(llm=llm)
    m_ctx_rec = ContextRecall(llm=llm)

    async def safe(metric, row_id, name, **kwargs):
        """One metric failing (token limits, parse errors) must not kill the
        run — record None and keep scoring the rest."""
        try:
            return (await metric.ascore(**kwargs)).value
        except Exception as exc:  # noqa: BLE001
            print(f"    !! {name} failed on {row_id}: {type(exc).__name__}", flush=True)
            return None

    for row in rows:
        q, ans, gt, ctxs = (
            row["question"], row["answer"], row["ground_truth"], row["contexts"],
        )
        # answer_correctness is meaningful for every row, including abstentions
        # (an abstention scored against an abstention reference scores high).
        row["answer_correctness"] = await safe(
            m_correct, row["id"], "answer_correctness",
            user_input=q, response=ans, reference=gt)

        # faithfulness: only for substantive answers with contexts.
        row["faithfulness"] = await safe(
            m_faith, row["id"], "faithfulness",
            user_input=q, response=ans, retrieved_contexts=ctxs,
        ) if ctxs and not row["abstained"] else None

        # context metrics: only when the corpus is supposed to contain the answer.
        if ctxs and row["expected_sources"]:
            row["context_precision"] = await safe(
                m_ctx_prec, row["id"], "context_precision",
                user_input=q, reference=gt, retrieved_contexts=ctxs)
            row["context_recall"] = await safe(
                m_ctx_rec, row["id"], "context_recall",
                user_input=q, retrieved_contexts=ctxs, reference=gt)
        else:
            row["context_precision"] = None
            row["context_recall"] = None
        print(f"    ragas scored {row['id']}", flush=True)


# =========================================================================
# Reporting
# =========================================================================

CSV_COLUMNS = [
    "id", "category", "question", "behavior_pass", "abstained", "gate_fired",
    "citation_hits", "citation_total", "rewritten", "loop_count",
    "generate_count", "answer_grounded", "answer_useful", "faithfulness",
    "answer_correctness", "context_precision", "context_recall", "answer",
]


def _fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else ("" if v is None else v)


def write_outputs(rows: list[dict], label: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"results_{label}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in CSV_COLUMNS})

    # per-category summary
    cats: dict[str, list[dict]] = {}
    for r in rows:
        cats.setdefault(r["category"], []).append(r)

    def avg(vals):
        vals = [v for v in vals if isinstance(v, float)]
        return f"{sum(vals) / len(vals):.3f}" if vals else "—"

    lines = [
        f"## Eval summary — `{label}` ({len(rows)} questions)",
        "",
        "| category | n | behavior pass | faithfulness | answer correctness | context precision | context recall |",
        "|---|---|---|---|---|---|---|",
    ]
    for cat in ("lookup", "multi_hop", "no_answer", "needs_web", "known_failure"):
        rs = cats.get(cat, [])
        if not rs:
            continue
        passed = sum(1 for r in rs if r["behavior_pass"])
        lines.append(
            f"| {cat} | {len(rs)} | {passed}/{len(rs)} "
            f"| {avg([r.get('faithfulness') for r in rs])} "
            f"| {avg([r.get('answer_correctness') for r in rs])} "
            f"| {avg([r.get('context_precision') for r in rs])} "
            f"| {avg([r.get('context_recall') for r in rs])} |"
        )
    total_pass = sum(1 for r in rows if r["behavior_pass"])
    lines += [
        f"| **all** | {len(rows)} | **{total_pass}/{len(rows)}** "
        f"| {avg([r.get('faithfulness') for r in rows])} "
        f"| {avg([r.get('answer_correctness') for r in rows])} "
        f"| {avg([r.get('context_precision') for r in rows])} "
        f"| {avg([r.get('context_recall') for r in rows])} |",
        "",
        "Failures:",
    ]
    fails = [r for r in rows if not r["behavior_pass"]]
    lines += [f"- **{r['id']}** ({r['category']}): {r['question']}" for r in fails] or ["- none"]

    md_path = RESULTS_DIR / f"summary_{label}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {csv_path.relative_to(ROOT)} and {md_path.relative_to(ROOT)}")


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", default="dev", help="tag for output filenames, e.g. v0.4")
    parser.add_argument("--only", help="run a single question id, e.g. q05")
    parser.add_argument("--limit", type=int, help="run only the first N questions")
    parser.add_argument("--approve-web", action="store_true",
                        help="auto-APPROVE the HITL gate instead of denying")
    parser.add_argument("--skip-ragas", action="store_true",
                        help="behavior checks only (fast, cheap)")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    _shim_legacy_vertexai()

    from langgraph.checkpoint.memory import MemorySaver

    from src import config  # noqa: F401  (loads .env)
    from src.graph import build_graph
    from src.nodes import ABSTENTION

    golden = load_golden(GOLDEN_PATH, ABSTENTION)
    if args.only:
        golden = [e for e in golden if e["id"] == args.only]
    if args.limit:
        golden = golden[: args.limit]
    if not golden:
        raise SystemExit("no golden entries selected")

    graph = build_graph(checkpointer=MemorySaver())

    rows = []
    for i, entry in enumerate(golden, start=1):
        print(f"[{i}/{len(golden)}] {entry['id']} ({entry['category']}): "
              f"{entry['question']}", flush=True)
        run = run_one(graph, entry, args.approve_web, ABSTENTION)
        hits, total = citation_hits(entry["expected_sources"], run["sources"])
        rows.append({
            **entry,
            **run,
            "citation_hits": hits,
            "citation_total": total,
            "behavior_pass": behavior_pass(entry, run, args.approve_web, ABSTENTION),
        })

    if not args.skip_ragas:
        # checkpoint the graph outputs first: a scoring failure shouldn't
        # cost the (much more expensive) graph run.
        write_outputs(rows, args.label)
        print("\nScoring with RAGAS…", flush=True)
        asyncio.run(ragas_scores(rows, config.LLM_MODEL))

    write_outputs(rows, args.label)


if __name__ == "__main__":
    main()
