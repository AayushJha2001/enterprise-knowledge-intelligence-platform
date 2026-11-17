import logging
import time
import uuid

from fastapi import FastAPI, Request

logger = logging.getLogger("app.request")

REQUEST_ID_HEADER = "x-request-id"


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
