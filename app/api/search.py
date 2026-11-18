import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.cache import Cache, access_scope, normalize_query
from app.core.config import get_settings
from app.core.security import PrincipalDep
from app.core.tracing import Trace
from app.db.session import get_session
from app.retrieval.search import Retriever, SearchMode

router = APIRouter(tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=8, ge=1, le=50)
    mode: SearchMode = Field(
        default="hybrid", description="hybrid (default), or dense / keyword only for comparison"
    )
    rerank: bool = Field(default=True, description="Re-score candidates with the cross-encoder")
    document_ids: list[uuid.UUID] | None = Field(
        default=None, max_length=100, description="Only search these documents"
    )

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


class SearchHitOut(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    chunk_index: int
    content: str
    section_path: str | None
    page_start: int | None
    page_end: int | None
    score: float = Field(description="Reranker relevance (0-1) if reranked, else RRF score")
    rrf_score: float
    dense_rank: int | None
    keyword_rank: int | None
    rerank_score: float | None


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    reranked: bool
    results: list[SearchHitOut]
    timings_ms: dict[str, float]
    cached: bool = False


@router.post("/search", response_model=SearchResponse)
async def search(
    body: SearchRequest,
    request: Request,
    principal: PrincipalDep,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> SearchResponse:
    """Find the passages most relevant to a query, among documents the caller may read.

    Send `Cache-Control: no-cache` to skip the result cache (used by the evals, so they
    measure real latency).
    """
    started = time.perf_counter()
    cache: Cache = request.app.state.cache
    version = await cache.index_version()
    key = None
    response = None
    if version >= 0:
        key = cache.key(
            "search",
            version,
            access_scope(principal),
            normalize_query(body.query),
            body.top_k,
            body.mode,
            body.rerank,
            sorted(map(str, body.document_ids or [])),
        )
        if not wants_fresh(request):
            hit = await cache.get_json(key)
            if hit is not None:
                response = SearchResponse(**{**hit, "cached": True})

    if response is None:
        retriever: Retriever = request.app.state.retriever
        result = await retriever.search(
            session,
            body.query,
            principal,
            top_k=body.top_k,
            mode=body.mode,
            rerank=body.rerank,
            document_ids=body.document_ids,
        )
        response = SearchResponse(
            query=body.query,
            mode=body.mode,
            reranked=result.reranked,
            results=[SearchHitOut.model_validate(h, from_attributes=True) for h in result.hits],
            timings_ms=result.timings_ms,
        )
        if key is not None:
            await cache.set_json(
                key, response.model_dump(mode="json"), get_settings().search_cache_ttl_s
            )

    await request.app.state.tracer.record(
        Trace(
            endpoint="search",
            status="ok" if response.results else "no_results",
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            user_id=principal.user_id,
            request_id=getattr(request.state, "request_id", None),
            query=body.query,
            cached=response.cached,
            timings_ms=response.timings_ms,
            result_count=len(response.results),
            top_score=response.results[0].score if response.results else None,
            source_chunk_ids=[str(r.chunk_id) for r in response.results],
        )
    )
    return response


def wants_fresh(request: Request) -> bool:
    return "no-cache" in request.headers.get("cache-control", "").lower()
