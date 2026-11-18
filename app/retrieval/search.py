"""Hybrid retrieval: vector search + keyword search, fused with RRF, then reranked.

Why both retrievers:
- Vector (dense) search matches meaning: "time off for new parents" finds "parental leave".
- Keyword (full-text) search matches exact terms that embeddings blur: error codes,
  product names, policy numbers like "HR-204".

Access control is applied inside both SQL queries, before ranking. A chunk the user may
not read is never a candidate, so it can't leak through scores, snippets or the LLM.
"""

import asyncio
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.security import Principal
from app.ingestion.embeddings import Embedder
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.rerank import Reranker

SearchMode = Literal["hybrid", "dense", "keyword"]

# A chunk is visible to admins, or to members of a group its document is shared with.
_ACL_FILTER = """
    (CAST(:is_admin AS boolean) OR EXISTS (
        SELECT 1 FROM document_access da
         WHERE da.document_id = c.document_id
           AND da.group_id = ANY(CAST(:group_ids AS uuid[]))
    ))
"""
_DOCUMENT_FILTER = """
    (CAST(:document_ids AS uuid[]) IS NULL OR c.document_id = ANY(CAST(:document_ids AS uuid[])))
"""

_DENSE_SQL = text(
    f"""
    SELECT c.id
      FROM chunks c
     WHERE c.embedding IS NOT NULL
       AND {_ACL_FILTER}
       AND {_DOCUMENT_FILTER}
     ORDER BY c.embedding <=> CAST(:query_vector AS vector)
     LIMIT :limit
    """
)

# plainto_tsquery ANDs every word, which is too strict for natural-language questions
# ("how do I reset my vpn password" rarely contains all five stems). Turning & into |
# matches any term; ts_rank_cd still ranks chunks with more (and closer) terms higher.
_KEYWORD_SQL = text(
    f"""
    WITH q AS (
        SELECT CAST(replace(CAST(plainto_tsquery('english', :query_text) AS text), '&', '|')
                    AS tsquery) AS query
    )
    SELECT c.id
      FROM chunks c, q
     WHERE c.tsv @@ q.query
       AND {_ACL_FILTER}
       AND {_DOCUMENT_FILTER}
     ORDER BY ts_rank_cd(c.tsv, q.query, 32) DESC, c.id
     LIMIT :limit
    """
)

_DETAILS_SQL = text(
    """
    SELECT c.id, c.document_id, d.title, c.chunk_index, c.content, c.section_path,
           c.page_number, c.metadata
      FROM chunks c
      JOIN documents d ON d.id = c.document_id
     WHERE c.id = ANY(CAST(:ids AS uuid[]))
    """
)


@dataclass
class SearchHit:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    chunk_index: int
    content: str
    section_path: str | None
    page_start: int | None
    page_end: int | None
    score: float  # reranker relevance if reranked, otherwise the RRF score
    rrf_score: float
    dense_rank: int | None
    keyword_rank: int | None
    rerank_score: float | None = None


@dataclass
class SearchResult:
    hits: list[SearchHit]
    reranked: bool
    timings_ms: dict[str, float] = field(default_factory=dict)


class _Timer:
    """Collects how long each step took, for the response and (Phase 6) tracing."""

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    @contextmanager
    def track(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = round((time.perf_counter() - start) * 1000, 2)


class Retriever:
    def __init__(self, embedder: Embedder, reranker: Reranker | None, settings: Settings) -> None:
        self.embedder = embedder
        self.reranker = reranker
        self.settings = settings

    async def search(
        self,
        session: AsyncSession,
        query: str,
        principal: Principal,
        *,
        top_k: int = 8,
        mode: SearchMode = "hybrid",
        rerank: bool = True,
        document_ids: list[uuid.UUID] | None = None,
    ) -> SearchResult:
        timer = _Timer()
        params = {
            "is_admin": principal.is_admin,
            "group_ids": list(principal.group_ids),
            "document_ids": document_ids,
            "limit": self.settings.retrieval_candidates,
        }
        ranked_lists: dict[str, list[uuid.UUID]] = {}

        if mode in ("hybrid", "dense"):
            with timer.track("embed"):
                vector = await asyncio.to_thread(self.embedder.embed_query, query)
            with timer.track("dense"):
                ef_search = int(self.settings.hnsw_ef_search)
                # Iterative scan: if the ACL filter discards most nearest neighbours, keep
                # walking the HNSW graph instead of returning too few results (pgvector 0.8+).
                await session.execute(text(f"SET LOCAL hnsw.ef_search = {ef_search}"))
                await session.execute(text("SET LOCAL hnsw.iterative_scan = strict_order"))
                rows = await session.execute(_DENSE_SQL, {**params, "query_vector": str(vector)})
                ranked_lists["dense"] = list(rows.scalars())

        if mode in ("hybrid", "keyword"):
            with timer.track("keyword"):
                rows = await session.execute(_KEYWORD_SQL, {**params, "query_text": query})
                ranked_lists["keyword"] = list(rows.scalars())

        fused = reciprocal_rank_fusion(ranked_lists, k=self.settings.rrf_k)
        use_reranker = rerank and self.reranker is not None
        keep = self.settings.rerank_candidates if use_reranker else top_k
        fused = fused[: max(keep, top_k)]
        if not fused:
            return SearchResult(hits=[], reranked=use_reranker, timings_ms=timer.timings)

        with timer.track("fetch"):
            rows = await session.execute(_DETAILS_SQL, {"ids": [item.id for item in fused]})
            details = {row.id: row for row in rows}

        hits = []
        for item in fused:
            row = details.get(item.id)
            if row is None:  # chunk deleted between queries (document re-indexed)
                continue
            hits.append(
                SearchHit(
                    chunk_id=row.id,
                    document_id=row.document_id,
                    document_title=row.title,
                    chunk_index=row.chunk_index,
                    content=row.content,
                    section_path=row.section_path,
                    page_start=row.page_number,
                    page_end=(row.metadata or {}).get("page_end", row.page_number),
                    score=item.score,
                    rrf_score=item.score,
                    dense_rank=item.ranks.get("dense"),
                    keyword_rank=item.ranks.get("keyword"),
                )
            )

        if use_reranker and hits:
            with timer.track("rerank"):
                passages = [_rerank_text(hit, self.settings.rerank_max_chars) for hit in hits]
                scores = await asyncio.to_thread(self.reranker.score, query, passages)
            for hit, relevance in zip(hits, scores, strict=True):
                hit.rerank_score = relevance
                hit.score = relevance
            hits.sort(key=lambda h: h.score, reverse=True)

        timer.timings["total"] = round(sum(timer.timings.values()), 2)
        return SearchResult(hits=hits[:top_k], reranked=use_reranker, timings_ms=timer.timings)


def _rerank_text(hit: SearchHit, max_chars: int) -> str:
    """Location header + the start of the chunk, cut at a word boundary.

    The header gives the cross-encoder the same context the embedding saw. Truncation
    bounds latency; the cost is that the reranker can't see the end of long chunks.
    """
    header = " > ".join(p for p in (hit.document_title, hit.section_path) if p)
    text = f"{header}\n{hit.content}" if header else hit.content
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0]
