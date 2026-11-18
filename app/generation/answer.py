"""Question answering: retrieve passages the user may read, then stream a cited answer."""

import time
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.cache import Cache, access_scope, normalize_query
from app.core.config import Settings
from app.core.security import Principal
from app.generation.gateway import CitationDelta, ClaudeGateway, Completed, TextDelta
from app.generation.pricing import Usage, estimate_cost_usd
from app.generation.prompts import ANSWER_SYSTEM_PROMPT, Source, user_message
from app.retrieval.search import Retriever, SearchResult

NO_ANSWER_TEXT = "I couldn't find information about this in the documents you have access to."
REFUSED_TEXT = "This request was declined, so no answer was generated."

AnswerStatus = Literal["answered", "no_relevant_sources", "refused", "truncated"]


@dataclass
class CitationRef:
    source: int  # Source.number
    cited_text: str


@dataclass
class Segment:
    """A run of answer text and the sources that support it."""

    text: str = ""
    citations: list[CitationRef] = field(default_factory=list)

    def markdown(self) -> str:
        numbers = list(dict.fromkeys(c.source for c in self.citations))
        if not numbers:
            return self.text
        markers = "".join(f"[{n}]" for n in numbers)
        stripped = self.text.rstrip()
        return f"{stripped}{markers}{self.text[len(stripped) :]}"


@dataclass
class AnswerResult:
    status: AnswerStatus
    answer: str  # Markdown with [n] citation markers
    segments: list[Segment]
    grounded: bool  # at least one claim carries a citation
    model: str | None
    usage: Usage
    cost_usd: float | None
    timings_ms: dict[str, float]
    request_id: str | None = None
    refusal_category: str | None = None
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sources_from_search(search: SearchResult) -> list[Source]:
    return [
        Source(
            number=i,
            chunk_id=hit.chunk_id,
            document_id=hit.document_id,
            document_title=hit.document_title,
            section_path=hit.section_path,
            page_start=hit.page_start,
            page_end=hit.page_end,
            score=hit.score,
            content=hit.content,
        )
        for i, hit in enumerate(search.hits, start=1)
    ]


def source_payload(source: Source) -> dict[str, Any]:
    return {**asdict(source), "label": source.label}


async def retrieve_for_generation(
    retriever: Retriever,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    query: str,
    principal: Principal,
    document_ids: list[uuid.UUID] | None,
) -> tuple[list[Source], SearchResult, bool]:
    """Retrieve passages; returns (sources, search, relevant_enough)."""
    # A short-lived session: the DB connection goes back to the pool before the
    # (much slower) model call starts.
    async with session_factory() as session:
        search = await retriever.search(
            session,
            query,
            principal,
            top_k=settings.answer_context_chunks,
            document_ids=document_ids,
        )
    sources = sources_from_search(search)
    relevant = bool(sources) and (
        not search.reranked or sources[0].score >= settings.answer_min_relevance
    )
    return sources, search, relevant


class AnswerService:
    def __init__(
        self,
        retriever: Retriever,
        gateway: ClaudeGateway,
        session_factory: async_sessionmaker[AsyncSession],
        settings: Settings,
        cache: Cache | None = None,
    ) -> None:
        self.retriever = retriever
        self.gateway = gateway
        self.session_factory = session_factory
        self.settings = settings
        self.cache = cache

    async def stream(
        self,
        question: str,
        principal: Principal,
        document_ids: list[uuid.UUID] | None = None,
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Yields (event, payload): sources, then text/citation deltas, then done.

        Answers are cached per access scope, generator and index version, and a cache hit
        is replayed through the same events, so clients can't tell the difference except
        for `cached: true` (and the speed).
        """
        key = await self._cache_key(question, principal, document_ids)
        if key is not None:
            started = time.perf_counter()
            hit = await self.cache.get_json(key)
            if hit is not None:
                for event in _replay(hit, _ms_since(started)):
                    yield event
                return

        sources: dict[str, Any] = {}
        async for name, payload in self._generate(question, principal, document_ids):
            if name == "sources":
                sources = payload
            elif (
                name == "done"
                and key is not None
                and payload["status"] in ("answered", "no_relevant_sources")
            ):
                await self.cache.set_json(
                    key, {"sources": sources, "done": payload}, self.settings.answer_cache_ttl_s
                )
            yield name, payload

    async def _cache_key(
        self, question: str, principal: Principal, document_ids: list[uuid.UUID] | None
    ) -> str | None:
        if self.cache is None:
            return None
        version = await self.cache.index_version()
        if version < 0:
            return None
        return self.cache.key(
            "answer",
            version,
            access_scope(principal),
            getattr(self.gateway, "cache_id", type(self.gateway).__name__),
            normalize_query(question),
            sorted(map(str, document_ids or [])),
        )

    async def _generate(
        self,
        question: str,
        principal: Principal,
        document_ids: list[uuid.UUID] | None,
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        sources, search, relevant = await retrieve_for_generation(
            self.retriever, self.session_factory, self.settings, question, principal, document_ids
        )
        timings = {f"retrieval_{k}": v for k, v in search.timings_ms.items()}
        yield "sources", {"sources": [source_payload(s) for s in sources]}

        if not relevant:
            yield (
                "done",
                AnswerResult(
                    status="no_relevant_sources",
                    answer=NO_ANSWER_TEXT,
                    segments=[Segment(NO_ANSWER_TEXT)],
                    grounded=False,
                    model=None,
                    usage=Usage(),
                    cost_usd=0.0,
                    timings_ms=timings,
                ).to_dict(),
            )
            return

        segments: dict[int, Segment] = {}
        completed: Completed | None = None
        started = time.perf_counter()
        async for event in self.gateway.stream(
            system=ANSWER_SYSTEM_PROMPT,
            messages=[user_message(sources, question, citations=True)],
        ):
            if isinstance(event, TextDelta):
                if "llm_first_token" not in timings:
                    timings["llm_first_token"] = _ms_since(started)
                segments.setdefault(event.block_index, Segment()).text += event.text
                yield "text", {"block": event.block_index, "text": event.text}
            elif isinstance(event, CitationDelta):
                number = event.search_result_index + 1
                if not 1 <= number <= len(sources):
                    continue  # defensive: never point at a source we didn't send
                ref = CitationRef(source=number, cited_text=event.cited_text)
                segments.setdefault(event.block_index, Segment()).citations.append(ref)
                yield "citation", {"block": event.block_index, **asdict(ref)}
            elif isinstance(event, Completed):
                completed = event
        timings["llm_total"] = _ms_since(started)

        assert completed is not None, "gateway stream ended without a Completed event"
        ordered = [segments[i] for i in sorted(segments)]
        status: AnswerStatus = "answered"
        if completed.stop_reason == "refusal":
            status = "refused"
            if not any(s.text.strip() for s in ordered):
                ordered = [Segment(REFUSED_TEXT)]
        elif completed.stop_reason == "max_tokens":
            status = "truncated"

        yield (
            "done",
            AnswerResult(
                status=status,
                answer="".join(s.markdown() for s in ordered).strip(),
                segments=ordered,
                grounded=any(s.citations for s in ordered),
                model=completed.model,
                usage=completed.usage,
                cost_usd=estimate_cost_usd(completed.model, completed.usage),
                timings_ms=timings,
                request_id=completed.request_id,
                refusal_category=completed.refusal_category,
            ).to_dict(),
        )


def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def _replay(hit: dict[str, Any], lookup_ms: float) -> Iterator[tuple[str, dict[str, Any]]]:
    yield "sources", hit["sources"]
    done = hit["done"]
    for block, segment in enumerate(done["segments"]):
        yield "text", {"block": block, "text": segment["text"]}
        for citation in segment["citations"]:
            yield "citation", {"block": block, **citation}
    yield (
        "done",
        {
            **done,
            "cached": True,
            "usage": asdict(Usage()),  # this request spent nothing
            "cost_usd": 0.0,
            "timings_ms": {"cache_lookup": lookup_ms},
            "request_id": None,
        },
    )
