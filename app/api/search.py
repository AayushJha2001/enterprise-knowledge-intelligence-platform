import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import PrincipalDep
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


@router.post("/search", response_model=SearchResponse)
async def search(
    body: SearchRequest,
    request: Request,
    principal: PrincipalDep,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> SearchResponse:
    """Find the passages most relevant to a query, among documents the caller may read."""
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
    return SearchResponse(
        query=body.query,
        mode=body.mode,
        reranked=result.reranked,
        results=[SearchHitOut.model_validate(hit, from_attributes=True) for hit in result.hits],
        timings_ms=result.timings_ms,
    )
