"""/health and /ready. Constructing TestClient(app) directly, with no `with`
block, skips the FastAPI lifespan -- as a context manager it would run
`ensure_index()` on entry and try to build the real corpus in CI. Plain
instantiation never touches the lifespan at all (verified: /health answers
and ensure_index is never called).

/health is unconditional liveness (Day 3): it must return 200 even with an
empty or unreachable index, so the platform doesn't kill a container that's
merely still building. /ready is the readiness split introduced the same day:
it must report the real document count, and 503 -- not 200 with a flag --
when the index is empty. That 503 is the entire reason /ready exists instead
of just deepening /health.
"""
from fastapi.testclient import TestClient

import app as app_module
from src import config


def test_health_is_always_200():
    client = TestClient(app_module.app)
    resp = client.get("/health")
    assert resp.status_code == 200


def test_ready_reports_document_count_when_index_is_populated(monkeypatch):
    monkeypatch.setattr(app_module, "index_size", lambda: 692)
    client = TestClient(app_module.app)

    resp = client.get("/ready")

    assert resp.status_code == 200
    assert resp.json()["documents"] == 692


def test_ready_returns_503_when_index_is_empty(monkeypatch):
    monkeypatch.setattr(app_module, "index_size", lambda: 0)
    client = TestClient(app_module.app)

    resp = client.get("/ready")

    assert resp.status_code == 503


def test_ask_with_missing_question_returns_422():
    # /ask's dependencies (rate limit, API key) resolve before the body is
    # parsed against AskRequest, so omitting the X-API-Key header here would
    # fail on auth (401/403) and never reach validation at all -- that would
    # be testing require_api_key, not the request schema. Send a valid key
    # so the only thing wrong with the request is the missing "question".
    client = TestClient(app_module.app)

    resp = client.post(
        "/ask",
        json={"thread_id": "t1"},
        headers={"X-API-Key": config.API_KEY},
    )

    assert resp.status_code == 422
