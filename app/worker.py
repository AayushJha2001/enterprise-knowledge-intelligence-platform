"""Ingestion worker process: `python -m app.worker`.

Runs alongside the API (a separate container), so slow parsing and embedding never block
API requests. Scale out by running more workers; the queue hands each document to one.
"""

import asyncio
import contextlib
import logging
import signal

from redis.asyncio import Redis

from app.core.cache import Cache
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.db.session import build_engine, build_session_factory
from app.ingestion.embeddings import Embedder, build_embedder
from app.ingestion.pipeline import process_next, requeue_stale
from app.ingestion.storage import LocalFileStorage

logger = logging.getLogger("app.worker")


async def run_worker(
    settings: Settings,
    embedder: Embedder | None = None,
    stop: asyncio.Event | None = None,
) -> None:
    stop = stop or asyncio.Event()
    engine = build_engine(settings)
    session_factory = build_session_factory(engine)
    storage = LocalFileStorage(settings.upload_dir)
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    cache = Cache(redis, settings)
    # Load the model before taking work, so a missing model fails fast at startup.
    embedder = embedder or await asyncio.to_thread(build_embedder, settings)
    logger.info("worker ready (embedding model: %s)", embedder.model_name)

    try:
        while not stop.is_set():
            try:
                status = await process_next(session_factory, storage, embedder, settings, cache)
            except Exception:
                logger.exception("worker loop error; backing off")
                status = None
            if status is None:  # queue empty (or error): housekeeping, then wait
                with contextlib.suppress(Exception):
                    await requeue_stale(session_factory, settings)
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(stop.wait(), settings.worker_poll_interval_s)
    finally:
        await redis.aclose()
        await engine.dispose()
        logger.info("worker stopped")


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    async def runner() -> None:
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            # Finish the current document, then exit (graceful container shutdown).
            with contextlib.suppress(NotImplementedError):  # not available on Windows
                loop.add_signal_handler(sig, stop.set)
        await run_worker(settings, stop=stop)

    asyncio.run(runner())


if __name__ == "__main__":
    main()
