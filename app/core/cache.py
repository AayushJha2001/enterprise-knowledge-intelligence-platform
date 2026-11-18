"""Redis caching for search results, answers and query embeddings.

Correctness rules:
- Keys include the caller's *access scope*: "admin", or a hash of their sorted group ids.
  Two users with the same groups can see exactly the same documents, so they may share
  cached results; users with different access never do.
- Keys include an *index version*, a counter bumped whenever searchable content or access
  changes (document ingested, deleted, regrouped). Bumping it orphans every old entry at
  once, with no key scanning; orphans expire through their TTL.
- Redis is an optimisation, not a dependency: if it fails, requests run uncached.
"""

import hashlib
import json
import logging
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.config import Settings
from app.core.security import Principal

logger = logging.getLogger(__name__)

INDEX_VERSION_KEY = "ekip:index:version"


def access_scope(principal: Principal) -> str:
    if principal.is_admin:
        return "admin"
    groups = ",".join(sorted(str(g) for g in principal.group_ids))
    return "groups:" + hashlib.sha256(groups.encode()).hexdigest()[:16]


def normalize_query(text: str) -> str:
    return " ".join(text.lower().split())


class Cache:
    def __init__(self, redis: Redis, settings: Settings) -> None:
        self.redis = redis
        self.enabled = settings.cache_enabled

    @staticmethod
    def key(kind: str, *parts: Any) -> str:
        raw = json.dumps(parts, default=str, sort_keys=True)
        return f"ekip:{kind}:{hashlib.sha256(raw.encode()).hexdigest()}"

    async def get_json(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        try:
            raw = await self.redis.get(key)
        except RedisError as exc:
            logger.warning("cache read failed: %s", exc)
            return None
        return json.loads(raw) if raw else None

    async def set_json(self, key: str, value: Any, ttl_s: int) -> None:
        if not self.enabled:
            return
        try:
            await self.redis.set(key, json.dumps(value, default=str), ex=ttl_s)
        except RedisError as exc:
            logger.warning("cache write failed: %s", exc)

    async def index_version(self) -> int:
        try:
            return int(await self.redis.get(INDEX_VERSION_KEY) or 0)
        except RedisError as exc:
            logger.warning("cache read failed: %s", exc)
            return -1  # unknown: callers skip the cache

    async def bump_index_version(self) -> None:
        try:
            await self.redis.incr(INDEX_VERSION_KEY)
        except RedisError as exc:
            # Stale results stay possible until TTLs expire; log loudly.
            logger.error("could not invalidate caches: %s", exc)
