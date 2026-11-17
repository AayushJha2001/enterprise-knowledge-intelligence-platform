from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory
from fastapi.testclient import TestClient


def _head_revision() -> str:
    root = Path(__file__).resolve().parents[1]
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "migrations"))
    return ScriptDirectory.from_config(config).get_current_head()


def test_live_returns_ok(client: TestClient) -> None:
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_reports_all_dependencies_ok(client: TestClient) -> None:
    response = client.get("/health/ready")
    body = response.json()
    assert response.status_code == 200, body
    assert body["status"] == "ok"
    assert body["checks"] == {"postgres": "ok", "redis": "ok"}
    assert body["schema_version"] == _head_revision()


def test_ready_returns_503_when_redis_is_down(client: TestClient) -> None:
    redis = client.app.state.redis

    async def failing_ping() -> bool:
        raise ConnectionError("redis unavailable")

    original_ping = redis.ping
    redis.ping = failing_ping
    try:
        response = client.get("/health/ready")
    finally:
        redis.ping = original_ping

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["redis"] == "error: ConnectionError"
    assert body["checks"]["postgres"] == "ok"


def test_request_id_is_generated(client: TestClient) -> None:
    response = client.get("/health/live")
    assert len(response.headers["x-request-id"]) == 32


def test_request_id_is_echoed_from_caller(client: TestClient) -> None:
    response = client.get("/health/live", headers={"x-request-id": "trace-abc-123"})
    assert response.headers["x-request-id"] == "trace-abc-123"
