import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from redis.asyncio import Redis

from app.api import ask, documents, health, search
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.middleware import add_request_context
from app.db.session import build_engine, build_session_factory
from app.generation.answer import AnswerService
from app.generation.extract import ExtractionService
from app.generation.gateway import build_gateway
from app.ingestion.embeddings import build_embedder
from app.ingestion.storage import LocalFileStorage
from app.retrieval.rerank import build_reranker
from app.retrieval.search import Retriever


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)

    # Connection pools are created once per process and shared by all requests.
    app.state.db_engine = build_engine(settings)
    app.state.session_factory = build_session_factory(app.state.db_engine)
    app.state.redis = Redis.from_url(settings.redis_url, decode_responses=True)
    app.state.storage = LocalFileStorage(settings.upload_dir)

    # Models load once at startup (blocking, so off the event loop), not per request.
    embedder, reranker = await asyncio.gather(
        asyncio.to_thread(build_embedder, settings),
        asyncio.to_thread(build_reranker, settings),
    )
    app.state.retriever = Retriever(embedder, reranker, settings)

    gateway = build_gateway(settings, embedder)
    app.state.answer_service = app.state.extraction_service = None
    if gateway is not None:
        app.state.answer_service = AnswerService(
            app.state.retriever, gateway, app.state.session_factory, settings
        )
        app.state.extraction_service = ExtractionService(
            app.state.retriever, gateway, app.state.session_factory, settings
        )
    try:
        yield
    finally:
        await app.state.redis.aclose()
        await app.state.db_engine.dispose()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="RAG over enterprise documents with access control, citations and evaluation.",
        lifespan=lifespan,
    )
    add_request_context(app)
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(search.router)
    app.include_router(ask.router)
    return app


app = create_app()
