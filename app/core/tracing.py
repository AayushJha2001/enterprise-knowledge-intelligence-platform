"""Per-request traces: one row per search, answer or extraction.

Traces are written to Postgres (so they can be queried with SQL and aggregated by
/admin/metrics) on a separate short session. Writing a trace never fails the request it
describes: errors are logged and dropped.
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.db.models import RequestTrace

logger = logging.getLogger(__name__)


@dataclass
class Trace:
    endpoint: str
    status: str
    latency_ms: float
    user_id: uuid.UUID | None = None
    request_id: str | None = None
    query: str | None = None
    cached: bool = False
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    result_count: int = 0
    top_score: float | None = None
    source_chunk_ids: list[str] = field(default_factory=list)
    error: str | None = None

    def apply_usage(self, usage: dict[str, Any] | None) -> None:
        usage = usage or {}
        self.input_tokens = int(usage.get("input_tokens") or 0)
        self.output_tokens = int(usage.get("output_tokens") or 0)
        self.cache_read_tokens = int(usage.get("cache_read_input_tokens") or 0)


class TraceRecorder:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession], settings: Settings):
        self.session_factory = session_factory
        self.settings = settings

    async def record(self, trace: Trace) -> None:
        if not self.settings.trace_enabled:
            return
        if not self.settings.trace_store_queries:
            trace.query = None
        try:
            async with self.session_factory() as session:
                session.add(RequestTrace(**trace.__dict__))
                await session.commit()
        except Exception:
            logger.warning("could not record trace", exc_info=True)
