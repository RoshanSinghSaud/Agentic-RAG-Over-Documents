"""Day 21 cost guardrails: no network, no API key, no index needed."""
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langgraph.checkpoint.memory import MemorySaver

from src import config, nodes
from src.budget import SessionLedger
from src.cost import CostTracker, price_for
from src.routing import classify


# --- routing ---------------------------------------------------------------

@pytest.mark.parametrize("question", [
    "How does CRAG differ from HyDE?",
    "Self-RAG and ReAct both let a model control its next step - compare them.",
    "What retriever does RAG build on, and how does it represent passages?",
])
def test_comparisons_and_two_part_questions_are_complex(question):
    assert classify(question) == "complex"


def test_short_lookup_is_simple():
    assert classify("What are reflection tokens in Self-RAG?") == "simple"


# --- cost accounting ---------------------------------------------------------

def _llm_result(model: str, tokens_in: int, tokens_out: int) -> LLMResult:
    msg = AIMessage(
        content="ok",
        response_metadata={"model_name": model},
        usage_metadata={"input_tokens": tokens_in, "output_tokens": tokens_out,
                        "total_tokens": tokens_in + tokens_out},
    )
    return LLMResult(generations=[[ChatGeneration(message=msg)]])


def test_price_lookup_matches_dated_model_names():
    assert price_for("gpt-4o-mini-2024-07-18") == (0.15, 0.60)
    assert price_for("some-unknown-model") is None


def test_cost_tracker_sums_calls_tokens_and_cost_across_models():
    t = CostTracker()
    t.on_llm_end(_llm_result("gpt-4o-mini-2024-07-18", 1_000_000, 0))
    t.on_llm_end(_llm_result("gpt-5-nano-2025-08-07", 0, 1_000_000))
    s = t.summary()
    assert s["llm_calls"] == 2
    assert s["input_tokens"] == 1_000_000 and s["output_tokens"] == 1_000_000
    assert s["cost_usd"] == pytest.approx(0.15 + 0.40)


# --- session ledger ------------------------------------------------------------

def test_ledger_accumulates_per_thread(tmp_path):
    ledger = SessionLedger(tmp_path / "usage.db")
    assert ledger.tokens_used("a") == 0
    assert ledger.add("a", 1_000, 0.001) == 1_000
    assert ledger.add("a", 500, 0.0005) == 1_500
    assert ledger.tokens_used("b") == 0


def test_ask_refuses_a_thread_past_its_token_budget(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import app as app_module

    monkeypatch.setattr(app_module, "ledger", SessionLedger(tmp_path / "usage.db"))
    app_module.ledger.add("spent", config.SESSION_TOKEN_LIMIT, 0.02)
    monkeypatch.setattr(app_module.graph, "get_state", lambda cfg: type("S", (), {"interrupts": ()})())
    client = TestClient(app_module.app)
    r = client.post("/ask", json={"thread_id": "spent", "question": "hi"},
                    headers={"X-API-Key": config.API_KEY})
    assert r.status_code == 429
    assert "token budget" in r.json()["detail"]


# --- global step budget ----------------------------------------------------------

def test_grading_router_finishes_when_no_room_for_a_rewrite():
    state = {"web_search": True, "loop_count": 0, "remaining_steps": 6}
    # a rewrite round needs 6 more nodes; only 5 fit -> skip to the web path
    assert nodes.decide_after_grading(state) == "web_search"
    state["remaining_steps"] = 4
    assert nodes.decide_after_grading(state) == "generate"


def test_answer_router_accepts_best_effort_when_budget_is_spent():
    state = {"answer_grounded": False, "answer_useful": False,
             "generate_count": 0, "remaining_steps": 3}
    assert nodes.decide_after_answer(state) == "end"


def test_runaway_rewrite_loop_is_stopped_by_the_step_budget(monkeypatch):
    """Mis-set the rewrite cap to 1,000 and make retrieval always look weak.
    Without the step budget the graph would rewrite until LangGraph raised
    GraphRecursionError (a 500). With it, the routers rewrite only while a
    full round still fits, then answer with what they have."""
    monkeypatch.setattr(config, "MAX_RETRIEVAL_LOOPS", 1_000)
    monkeypatch.setattr(config, "MAX_GRAPH_STEPS", 12)
    monkeypatch.setattr(nodes, "retrieve", lambda s: {"documents": [Document(page_content="x")]})
    monkeypatch.setattr(nodes, "grade_documents", lambda s: {"documents": [], "web_search": True})
    monkeypatch.setattr(nodes, "transform_query",
                        lambda s: {"question": s["question"], "loop_count": s.get("loop_count", 0) + 1})
    monkeypatch.setattr(nodes, "generate", lambda s: {"generation": nodes.ABSTENTION})
    from src.graph import build_graph

    graph = build_graph(checkpointer=MemorySaver())
    result = graph.invoke(
        {"question": "q", "messages": [HumanMessage(content="q")], "loop_count": 0},
        config={"configurable": {"thread_id": "runaway"}},
    )
    assert result["generation"] == nodes.ABSTENTION   # finished with an answer
    assert result["loop_count"] == 2                   # 12 steps fit exactly two rewrite rounds
