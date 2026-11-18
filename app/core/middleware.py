import logging
import time
import uuid

from fastapi import FastAPI, Request

logger = logging.getLogger("app.request")

REQUEST_ID_HEADER = "x-request-id"

# The UI loads only its own files and talks only to this origin, so the policy can be
# strict: no inline scripts, no third-party code, no framing (clickjacking).
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; "
    "frame-ancestors 'none'"
)
# Swagger UI and ReDoc load their scripts from a CDN, so they get no CSP.
_CSP_EXEMPT_PREFIXES = ("/docs", "/redoc", "/openapi.json")


def add_security_headers(app: FastAPI) -> None:
    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        headers = response.headers
        headers.setdefault("X-Content-Type-Options", "nosniff")
        headers.setdefault("X-Frame-Options", "DENY")
        headers.setdefault("Referrer-Policy", "no-referrer")
        headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        if not request.url.path.startswith(_CSP_EXEMPT_PREFIXES):
            headers.setdefault("Content-Security-Policy", CONTENT_SECURITY_POLICY)
        if request.url.scheme == "https":
            headers.setdefault("Strict-Transport-Security", "max-age=31536000")
        return response


def add_request_context(app: FastAPI) -> None:
    """Give every request an id (reused from the caller if sent) and log its latency."""

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        request.state.request_id = request_id
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        finally:
            logger.info(
                "request",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "duration_ms": round((time.perf_counter() - start) * 1000, 2),
                    "client_ip": request.client.host if request.client else None,
                },
            )
