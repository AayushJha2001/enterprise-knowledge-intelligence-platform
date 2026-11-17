"""Structured JSON logging, so logs can be searched and aggregated (e.g. in Azure Monitor)."""

import json
import logging
import sys
from datetime import UTC, datetime

# Fields passed via `logger.info("...", extra={...})` that should appear in the JSON line.
_EXTRA_FIELDS = (
    "request_id",
    "method",
    "path",
    "status_code",
    "duration_ms",
    "client_ip",
    "document_id",
    "document_status",
    "chunk_count",
    "attempt",
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in _EXTRA_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())

    # Uvicorn's own access log duplicates our request log; keep its error log.
    logging.getLogger("uvicorn.access").handlers = []
    logging.getLogger("uvicorn.access").propagate = False
    # Model downloads log every HTTP request at INFO.
    for noisy in ("httpx", "huggingface_hub", "fastembed"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
