"""The built-in web UI, security headers and CORS."""

import re

from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.core.middleware import CONTENT_SECURITY_POLICY
from app.main import create_app


def test_ui_is_served_with_strict_csp(client: TestClient) -> None:
    response = client.get("/", headers={"Authorization": ""})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Knowledge Assistant" in response.text
    csp = response.headers["content-security-policy"]
    assert csp == CONTENT_SECURITY_POLICY
    assert "script-src 'self'" in csp and "unsafe-inline" not in csp
    assert "frame-ancestors 'none'" in csp


def test_ui_assets_are_served(client: TestClient) -> None:
    js = client.get("/app.js")
    assert js.status_code == 200
    assert "javascript" in js.headers["content-type"]
    # All server text goes through textContent: no HTML-parsing sinks in the UI code.
    assert not re.search(r"\.(innerHTML|outerHTML)\s*=|insertAdjacentHTML|document\.write", js.text)
    assert client.get("/styles.css").headers["content-type"].startswith("text/css")


def test_security_headers_on_api_responses(client: TestClient) -> None:
    response = client.get("/health/live")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert "strict-transport-security" not in response.headers  # plain http in tests


def test_api_docs_are_exempt_from_csp(client: TestClient) -> None:
    # Swagger UI loads its scripts from a CDN, which the strict policy would block.
    response = client.get("/docs")
    assert response.status_code == 200
    assert "content-security-policy" not in response.headers


def test_api_routes_take_precedence_over_the_ui(client: TestClient) -> None:
    assert client.get("/health/live").json() == {"status": "ok"}
    assert client.get("/no-such-page").status_code == 404


def test_cors_is_off_by_default(client: TestClient) -> None:
    response = client.options(
        "/search",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in response.headers


def test_cors_allows_only_configured_origins(monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "cors_allow_origins", ["https://kb.example.com"])
    with TestClient(create_app()) as client:
        preflight = {"Access-Control-Request-Method": "POST"}
        allowed = client.options(
            "/search", headers={"Origin": "https://kb.example.com", **preflight}
        )
        assert allowed.headers["access-control-allow-origin"] == "https://kb.example.com"
        assert "authorization" in allowed.headers["access-control-allow-headers"].lower()
        denied = client.options("/search", headers={"Origin": "https://evil.example", **preflight})
        assert "access-control-allow-origin" not in denied.headers
