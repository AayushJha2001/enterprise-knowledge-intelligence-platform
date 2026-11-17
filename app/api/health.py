"""Liveness and readiness probes (used by Docker, Azure Container Apps and load balancers).

- /health/live  : the process is up. Never touches dependencies, so a slow DB won't
                  get a healthy container restarted.
- /health/ready : Postgres, Redis and the schema are usable. Returns 503 otherwise, so
                  traffic is held back until the instance can actually serve requests.
"""

import asyncio
from typing import Literal

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel
from sqlalchemy import text

from app.core.config import get_settings

router = APIRouter(prefix="/health", tags=["health"])


class LiveResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadyResponse(BaseModel):
    status: Literal["ok", "degraded"]
    checks: dict[str, str]
    schema_version: str | None = None


@router.get("/live", response_model=LiveResponse)
async def live() -> LiveResponse:
    return LiveResponse()


@router.get("/ready", response_model=ReadyResponse)
async def ready(request: Request, response: Response) -> ReadyResponse:
    timeout = get_settings().health_check_timeout_s
    checks: dict[str, str] = {}
    schema_version: str | None = None

    async def check_postgres() -> str | None:
        async with request.app.state.db_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
            result = await conn.execute(text("SELECT version_num FROM alembic_version"))
            return result.scalar_one_or_none()

    try:
        schema_version = await asyncio.wait_for(check_postgres(), timeout)
        checks["postgres"] = "ok" if schema_version else "error: no migrations applied"
    except Exception as exc:  # report, don't crash, the probe
        checks["postgres"] = f"error: {type(exc).__name__}"

    try:
        await asyncio.wait_for(request.app.state.redis.ping(), timeout)
        checks["redis"] = "ok"
    except Exception as exc:
        checks["redis"] = f"error: {type(exc).__name__}"

    healthy = all(value == "ok" for value in checks.values())
    response.status_code = 200 if healthy else 503
    return ReadyResponse(
        status="ok" if healthy else "degraded",
        checks=checks,
        schema_version=schema_version,
    )
