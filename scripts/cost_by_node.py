"""Cost, tokens and latency per query, broken down by graph node (Day 20).

Pulls one eval run's traces back out of LangSmith and aggregates them by the
LangGraph node each run happened in. No node code is involved: LangGraph
stamps every run inside a node with metadata["langgraph_node"], so an LLM call
made inside grade_documents is attributed to grade_documents automatically.

Two kinds of run are read:
  - LLM runs (run_type="llm")      -> tokens in/out and cost, grouped by node
  - node runs (name == the node)   -> wall-clock latency per node

Usage, from the repo root, with LANGSMITH_* set in .env:
    python -m eval.run_eval --label day20 --skip-ragas   # produce the traces
    python scripts/cost_by_node.py --label day20         # read them back

Writes eval/results/cost_by_node_<label>.md and prints the same table.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Graph order, so the table reads top to bottom like the flow.
NODES = [
    "retrieve", "grade_documents", "transform_query", "human_approval_gate",
    "web_search", "generate", "grade_answer", "finalize",
]

# Used only if LangSmith returned no cost for a run (model missing from its
# price table). USD per 1M tokens, (input, output). Check against
# https://platform.openai.com/docs/pricing if you change LLM_MODEL.
FALLBACK_PRICE = {"gpt-4o-mini": (0.15, 0.60)}


def _meta(run) -> dict:
    return (run.extra or {}).get("metadata") or {}


def _secs(run) -> float:
    if run.start_time and run.end_time:
        return (run.end_time - run.start_time).total_seconds()
    return 0.0


def _cost(run) -> tuple[float, bool]:
    """(cost in USD, True if computed locally from the fallback price)."""
    if run.total_cost is not None:
        return float(run.total_cost), False
    model = _meta(run).get("ls_model_name", "")
    for name, (p_in, p_out) in FALLBACK_PRICE.items():
        if model.startswith(name):
            return ((run.prompt_tokens or 0) * p_in
                    + (run.completion_tokens or 0) * p_out) / 1e6, True
    return 0.0, True


def fetch(client: Client, project: str, label: str, hours: float) -> list:
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    # The server-side metadata filter keeps the download small; the
    # client-side check below is the source of truth either way.
    # list_runs is deprecated in favour of client.runs.query (removal after
    # 2027-01-31) — still the simplest call on langsmith 0.12.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        runs = list(client.list_runs(
            project_name=project,
            start_time=since,
            filter=f'and(eq(metadata_key, "eval_label"), eq(metadata_value, "{label}"))',
        ))
    # end_time is None for a run that never finished (process killed
    # mid-trace); its tokens and latency would read as zeros.
    return [r for r in runs
            if _meta(r).get("eval_label") == label and r.end_time is not None]


def summarise(runs: list, label: str, project: str) -> str:
    threads = {_meta(r).get("thread_id") for r in runs} - {None}
    n = len(threads)
    if n == 0:
        raise SystemExit(f"no runs with eval_label={label!r} in project {project!r}")

    calls = defaultdict(int)
    tok_in = defaultdict(int)
    tok_out = defaultdict(int)
    cost = defaultdict(float)
    node_secs = defaultdict(list)
    per_query_cost = defaultdict(float)
    per_query_secs = defaultdict(float)
    nodes_seen = defaultdict(set)
    fallback = 0

    for r in runs:
        m = _meta(r)
        node = m.get("langgraph_node")
        if r.run_type == "llm":
            c, used_fallback = _cost(r)
            fallback += used_fallback
            key = node or "(outside a node)"
            calls[key] += 1
            tok_in[key] += r.prompt_tokens or 0
            tok_out[key] += r.completion_tokens or 0
            cost[key] += c
            per_query_cost[m.get("thread_id")] += c
        elif node and r.name == node:          # the node's own run
            node_secs[node].append(_secs(r))
            nodes_seen[m.get("thread_id")].add(node)
        elif r.parent_run_id is None:          # a trace root (2 if the gate fired)
            per_query_secs[m.get("thread_id")] += _secs(r)

    total_cost = sum(cost.values())
    total_time = sum(sum(v) for v in node_secs.values())
    order = [x for x in NODES if x in cost or x in node_secs]
    order += sorted(k for k in cost if k not in NODES)

    lines = [
        f"# Cost per query by node — `{label}`",
        "",
        f"{n} queries · project `{project}` · pulled "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "| node | LLM calls / q | tokens in / q | tokens out / q | $ / query "
        "| % of cost | p50 latency / run | time / q | % of time |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in order:
        secs = node_secs.get(k, [])
        lines.append(
            f"| `{k}` | {calls[k] / n:.1f} | {tok_in[k] / n:,.0f} | "
            f"{tok_out[k] / n:,.0f} | ${cost[k] / n:.6f} | "
            f"{100 * cost[k] / total_cost if total_cost else 0:.0f}% | "
            f"{statistics.median(secs) if secs else 0:.2f}s | "
            f"{sum(secs) / n:.2f}s | "
            f"{100 * sum(secs) / total_time if total_time else 0:.0f}% |"
        )

    q_costs = sorted(per_query_cost.values())
    worst = max(per_query_cost, key=per_query_cost.get)
    top = max(cost, key=cost.get)
    slow = max(node_secs, key=lambda k: sum(node_secs[k]))
    lines += [
        "",
        f"- **Cost per query:** mean ${total_cost / n:.6f} · median "
        f"${statistics.median(q_costs):.6f} · max ${q_costs[-1]:.6f} (`{worst}`) "
        f"· ≈ ${1000 * total_cost / n:.2f} per 1,000 queries",
        f"- **Tokens per query:** {sum(tok_in.values()) / n:,.0f} in / "
        f"{sum(tok_out.values()) / n:,.0f} out",
        f"- **End-to-end latency:** median "
        f"{statistics.median(per_query_secs.values()):.1f}s per query",
        f"- **Most expensive node:** `{top}` — "
        f"{100 * cost[top] / total_cost:.0f}% of cost, "
        f"{calls[top] / n:.1f} LLM calls per query",
        f"- **Slowest node:** `{slow}` — "
        f"{100 * sum(node_secs[slow]) / total_time:.0f}% of node time",
    ]
    # A query that rewrote at least once went down the CRAG corrective path:
    # every rewrite re-runs retrieve + grade_documents on a fresh chunk set.
    rewrote = [t for t in per_query_cost if "transform_query" in nodes_seen[t]]
    direct = [t for t in per_query_cost if t not in rewrote]
    if rewrote and direct:
        mean = lambda ts: sum(per_query_cost[t] for t in ts) / len(ts)
        lines.append(
            f"- **By path:** direct {len(direct)} q at ${mean(direct):.6f} · "
            f"corrective (rewrote) {len(rewrote)} q at ${mean(rewrote):.6f} "
            f"({mean(rewrote) / mean(direct):.1f}×)"
        )
    lines += [
        "",
        "Not in the cost column: the query embedding inside `retrieve` "
        "(OpenAIEmbeddings isn't traced as an LLM run; ~20 tokens at "
        "$0.02/1M, effectively $0) and Tavily credits in `web_search`.",
    ]
    if fallback:
        lines.append(f"\n{fallback} LLM run(s) had no LangSmith cost; priced "
                     "locally from FALLBACK_PRICE.")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", required=True, help="the --label given to eval.run_eval")
    ap.add_argument("--project", default=os.getenv("LANGSMITH_PROJECT", "agentic-rag"))
    ap.add_argument("--hours", type=float, default=24, help="look-back window")
    ap.add_argument("--out", type=Path, help="default: eval/results/cost_by_node_<label>.md")
    args = ap.parse_args()

    runs = fetch(Client(), args.project, args.label, args.hours)
    report = summarise(runs, args.label, args.project)
    out = args.out or ROOT / "eval" / "results" / f"cost_by_node_{args.label}.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"written: {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
