"""FastAPI service for agentic-rag.

Every input() in main.py becomes a request field here; every print() becomes
a response field. thread_id and question/approved travel in the body (no
path params) — see AskRequest / ResumeRequest below.

Run with:
    uvicorn app:app --reload
"""
import logging
import secrets
import sqlite3
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Annotated, Literal, Union

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from src import config, ingestion
from src.budget import SessionLedger
from src.cost import CostTracker, log_request_cost
from src.graph import build_graph
from src.nodes import ABSTENTION
from src.retrieval import index_size


# Root handler so our own loggers (e.g. src.nodes) print a level name. Without
# it, Python's last-resort handler emits the bare message and the CloudWatch
# metric filter on "ERROR" never matches. Uvicorn's loggers don't propagate to
# root, so this doesn't duplicate its lines.
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:    %(name)s - %(message)s",
)

conn = sqlite3.connect(str(config.CHECKPOINT_DB), check_same_thread=False)
saver = SqliteSaver(conn)
graph = build_graph(checkpointer=saver)
ledger = SessionLedger(config.CHECKPOINT_DB)  # Day 21: tokens spent per thread_id


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs to completion before uvicorn starts accepting connections, so no
    # request can ever see an empty or half-built index.
    ingestion.ensure_index()
    yield


app = FastAPI(title="Agentic RAG", lifespan=lifespan)

_api_key_header = APIKeyHeader(name="X-API-Key")


def require_api_key(key: str = Security(_api_key_header)) -> None:
    # compare_digest, not ==: a plain string comparison short-circuits on the
    # first mismatched byte, which leaks how many leading characters of the
    # guess were correct via response timing.
    if not secrets.compare_digest(key, config.API_KEY):
        raise HTTPException(status_code=401, detail="invalid API key")


# Per-IP request timestamps, in memory. Only correct with a single worker
# process (see Dockerfile) — each worker would keep its own dict, so a client
# could get RATE_LIMIT requests through per worker instead of in total.
RATE_LIMIT = 10          # requests
RATE_LIMIT_WINDOW = 60.0  # seconds
_request_times: dict[str, list[float]] = defaultdict(list)


def rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    times = _request_times[ip]
    cutoff = now - RATE_LIMIT_WINDOW
    while times and times[0] < cutoff:
        times.pop(0)
    if len(times) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="rate limit exceeded: 10 requests/minute")
    times.append(now)


class AskRequest(BaseModel):
    thread_id: str
    question: str


class Citation(BaseModel):
    index: int
    source: str
    page: int | None = None
    web: bool = False


class AskResponse(BaseModel):
    status: Literal["completed"] = "completed"
    thread_id: str
    answer: str
    rewritten_question: str | None = None   # only set if CRAG rewrote it
    verdict: str | None = None              # the SELF-CHECK line, e.g. "passed (grounded + addresses the question)"
    retries: int = 0                        # failed grounding checks before verdict
    sources: list[Citation] = []


class AskInterruptedResponse(BaseModel):
    status: Literal["needs_approval"] = "needs_approval"
    thread_id: str
    reason: str          # payload["reason"]
    question: str        # payload["question"] — the query pending web search
    instruction: str     # payload["instruction"] — what the caller is being asked to decide


class ResumeRequest(BaseModel):
    thread_id: str
    approved: bool         # web search: approve or deny


AskResult = Annotated[Union[AskResponse, AskInterruptedResponse], Field(discriminator="status")]


def _run_config(thread_id: str) -> dict:
    # run_name/tags only matter when LangSmith tracing is on
    # (LANGSMITH_TRACING=true); otherwise they are ignored. The "api" tag
    # separates live traffic from eval runs ("eval") in the same project.
    # thread_id needs no copy here: LangGraph adds configurable values to the
    # run metadata itself, which is what LangSmith's Threads view groups on.
    return {
        "configurable": {"thread_id": thread_id},
        "run_name": "agentic-rag",
        "tags": ["api"],
    }


def _invoke_tracked(endpoint: str, thread_id: str, graph_input, run_config: dict) -> dict:
    """graph.invoke with a CostTracker attached. Whatever happens, the tokens
    spent are added to the session's total and one JSON cost line is logged:
    a request that fails halfway still spent money."""
    tracker = CostTracker()
    started = time.perf_counter()
    status, result = "failed", {}
    try:
        result = graph.invoke(graph_input, config={**run_config, "callbacks": [tracker]})
        status = "needs_approval" if "__interrupt__" in result else "completed"
        return result
    finally:
        usage = tracker.summary()
        session_tokens = ledger.add(
            thread_id, usage["input_tokens"] + usage["output_tokens"], usage["cost_usd"]
        )
        log_request_cost(
            endpoint=endpoint,
            thread_id=thread_id,
            status=status,
            route=result.get("route"),
            rewrites=result.get("loop_count"),
            regenerations=result.get("generate_count"),
            latency_ms=round(1000 * (time.perf_counter() - started)),
            session_tokens=session_tokens,
            session_token_limit=config.SESSION_TOKEN_LIMIT,
            **usage,
        )


def _citations(documents) -> list[Citation]:
    citations = []
    for i, d in enumerate(documents, start=1):
        citations.append(
            Citation(
                index=i,
                source=d.metadata.get("source", "unknown"),
                page=d.metadata.get("page"),
                web=bool(d.metadata.get("web")),
            )
        )
    return citations


def _to_result(result: dict, thread_id: str, question_before: str) -> AskResult:
    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        return AskInterruptedResponse(
            thread_id=thread_id,
            reason=payload["reason"],
            question=payload["question"],
            instruction=payload["instruction"],
        )

    grounded = result.get("answer_grounded")
    useful = result.get("answer_useful")
    verdict = None
    if grounded is not None:
        if ABSTENTION.lower() in result["generation"].lower():
            verdict = "honest abstention (corpus can't answer this)"
        elif grounded and useful:
            verdict = "passed (grounded + addresses the question)"
        else:
            verdict = "best effort (correction budget exhausted)"

    return AskResponse(
        thread_id=thread_id,
        answer=result["generation"],
        rewritten_question=result["question"] if result["question"] != question_before else None,
        verdict=verdict,
        retries=result.get("generate_count", 0),
        sources=_citations(result["documents"]),
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResult, dependencies=[Depends(rate_limit), Depends(require_api_key)])
def ask(request: AskRequest) -> AskResult:
    run_config = _run_config(request.thread_id)

    if graph.get_state(run_config).interrupts:
        raise HTTPException(
            status_code=409,
            detail=f"thread '{request.thread_id}' is paused awaiting approval; call /resume instead.",
        )

    # Day 21: per-session token ceiling. Checked here, not in /resume: a turn
    # that is already paused at the approval gate is allowed to finish.
    used = ledger.tokens_used(request.thread_id)
    if used >= config.SESSION_TOKEN_LIMIT:
        log_request_cost(
            endpoint="/ask", thread_id=request.thread_id, status="refused_session_budget",
            session_tokens=used, session_token_limit=config.SESSION_TOKEN_LIMIT,
            llm_calls=0, input_tokens=0, output_tokens=0, cost_usd=0.0,
        )
        raise HTTPException(
            status_code=429,
            detail=(
                f"session token budget used up for thread '{request.thread_id}': "
                f"{used:,} of {config.SESSION_TOKEN_LIMIT:,} tokens. Start a new thread_id."
            ),
        )

    result = _invoke_tracked(
        "/ask",
        request.thread_id,
        {
            "question": request.question,
            "messages": [HumanMessage(content=request.question)],
            "loop_count": 0,
            "generate_count": 0,
            "web_search": False,
            "web_search_approved": False,
            "route": None,  # turn-scoped too: an interrupted turn has no route yet
        },
        run_config,
    )
    return _to_result(result, request.thread_id, request.question)


@app.post("/resume", response_model=AskResult, dependencies=[Depends(rate_limit), Depends(require_api_key)])
def resume(request: ResumeRequest) -> AskResult:
    run_config = _run_config(request.thread_id)

    snapshot = graph.get_state(run_config)
    if not snapshot.interrupts:
        raise HTTPException(
            status_code=409,
            detail=f"thread '{request.thread_id}' has no pending approval to resume.",
        )
    question_before = snapshot.values.get("question", "")

    result = _invoke_tracked(
        "/resume", request.thread_id, Command(resume=request.approved), run_config
    )
    return _to_result(result, request.thread_id, question_before)

@app.get("/ready")
def ready() -> dict:
    """Liveness vs readiness: /health says the process is up; this says it can
    actually answer a question. 503 (not 200-with-a-flag) so load balancers
    and orchestrators route traffic away until ingestion has run.

    Every failure path here answers 503. A readiness probe that raises is a
    probe reporting 500 for the one condition it exists to describe.
    """
    try:
        count = index_size()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"vector store unreachable at {config.CHROMA_DIR}: {exc}",
        ) from exc

    if count == 0:
        raise HTTPException(
            status_code=503,
            detail="document index is empty; run `python main.py ingest` first.",
        )
    return {"status": "ready", "documents": count}
