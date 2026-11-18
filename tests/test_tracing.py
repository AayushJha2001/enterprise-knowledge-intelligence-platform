"""Request traces and the /admin/metrics and /admin/traces endpoints."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.core.config import get_settings
from tests.helpers import bearer, create_user, ingest, run_db

DOC = b"# HR Handbook\n\n## Leave\n\nParents receive sixteen weeks of paid parental leave."


def traces() -> list[dict]:
    async def query(conn):
        rows = await conn.execute(text("SELECT * FROM request_traces ORDER BY created_at"))
        return [dict(r._mapping) for r in rows]

    return run_db(query)


@pytest.fixture
def doc(client: TestClient) -> str:
    return ingest(client, "hr.md", DOC)


def test_search_is_traced(client: TestClient, doc, admin_id) -> None:
    client.post("/search", json={"query": "paid parental leave"}, headers={"X-Request-ID": "r-1"})
    [trace] = traces()
    assert trace["endpoint"] == "search"
    assert trace["status"] == "ok"
    assert trace["request_id"] == "r-1"
    assert trace["user_id"] == admin_id
    assert trace["query"] == "paid parental leave"
    assert trace["result_count"] == 1 and trace["top_score"] is not None
    assert trace["latency_ms"] > 0
    assert {"embed", "dense", "keyword"} <= trace["timings_ms"].keys()
    assert len(trace["source_chunk_ids"]) == 1


@pytest.mark.parametrize("stream", [False, True])
def test_ask_is_traced(client: TestClient, doc, stream: bool) -> None:
    client.post("/ask", json={"question": "paid parental leave weeks", "stream": stream})
    [trace] = traces()
    assert trace["endpoint"] == "ask"
    assert trace["status"] == "answered"
    assert trace["model"] == "extractive-v1"
    assert trace["cost_usd"] == 0.0
    assert "llm_total" in trace["timings_ms"]
    assert trace["result_count"] >= 1


def test_ask_failure_is_traced(client: TestClient, doc, monkeypatch) -> None:
    gateway = client.app.state.answer_service.gateway

    async def broken(**kwargs):
        raise RuntimeError("model exploded")
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(gateway, "stream", broken)
    response = client.post("/ask", json={"question": "paid parental leave", "stream": False})
    assert response.status_code == 500
    [trace] = traces()
    assert trace["status"] == "error"
    assert "model exploded" in trace["error"]


def test_extract_is_traced(client: TestClient, doc) -> None:
    client.post(
        "/extract",
        json={"query": "parental leave", "fields": [{"name": "weeks", "description": "weeks"}]},
    )
    [trace] = traces()
    assert trace["endpoint"] == "extract"
    assert trace["status"] == "extracted"


def test_queries_can_be_left_out(client: TestClient, doc, monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "trace_store_queries", False)
    client.post("/search", json={"query": "secret question"})
    assert traces()[0]["query"] is None


def test_trace_failure_never_breaks_the_request(client: TestClient, doc, monkeypatch) -> None:
    def explode():
        raise RuntimeError("database is down")

    monkeypatch.setattr(client.app.state.tracer, "session_factory", explode)
    assert client.post("/search", json={"query": "parental leave"}).status_code == 200


@pytest.fixture
def cached(client: TestClient) -> Iterator[TestClient]:
    client.app.state.cache.enabled = True
    yield client
    client.app.state.cache.enabled = False


def test_no_cache_header_bypasses_cache(cached: TestClient, doc) -> None:
    assert cached.post("/search", json={"query": "parental leave"}).json()["cached"] is False
    assert cached.post("/search", json={"query": "parental leave"}).json()["cached"] is True
    fresh = cached.post(
        "/search", json={"query": "parental leave"}, headers={"Cache-Control": "no-cache"}
    )
    assert fresh.json()["cached"] is False
    ask = {"question": "paid parental leave", "stream": False}
    cached.post("/ask", json=ask)
    assert cached.post("/ask", json=ask).json()["cached"] is True
    assert (
        cached.post("/ask", json=ask, headers={"Cache-Control": "no-cache"}).json()["cached"]
        is False
    )


def test_metrics(cached: TestClient, doc) -> None:
    for _ in range(3):
        cached.post("/search", json={"query": "parental leave"})  # 1 miss, 2 cache hits
    cached.post("/search", json={"query": "zzz qqq"}, headers={"Cache-Control": "no-cache"})
    cached.post("/ask", json={"question": "paid parental leave", "stream": False})

    body = cached.get("/admin/metrics?hours=1").json()
    assert body["window_hours"] == 1
    metrics = {m["endpoint"]: m for m in body["endpoints"]}
    assert metrics["search"]["requests"] == 4
    assert metrics["search"]["cache_hit_rate"] == 0.5
    assert metrics["search"]["errors"] == 0
    assert metrics["search"]["latency_p95_ms"] >= metrics["search"]["latency_p50_ms"]
    assert metrics["ask"]["requests"] == 1
    assert metrics["ask"]["cost_usd"] == 0.0


def test_traces_listing(client: TestClient, doc) -> None:
    client.post("/search", json={"query": "parental leave"})
    client.post("/ask", json={"question": "paid parental leave", "stream": False})
    assert [t["endpoint"] for t in client.get("/admin/traces").json()] == ["ask", "search"]
    assert [t["endpoint"] for t in client.get("/admin/traces?endpoint=search").json()] == ["search"]


def test_observability_endpoints_are_admin_only(client: TestClient) -> None:
    member = create_user("member@example.com")
    assert client.get("/admin/metrics", headers=bearer(member)).status_code == 403
    assert client.get("/admin/traces", headers=bearer(member)).status_code == 403
