"""Question answering and structured extraction over the documents a user may read."""

import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Annotated, Any

import anthropic
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import AfterValidator, BaseModel, Field, field_validator

from app.core.security import PrincipalDep
from app.generation.answer import AnswerService
from app.generation.extract import ExtractionService, FieldSpec, FieldType

logger = logging.getLogger(__name__)

router = APIRouter(tags=["answers"])


def _not_blank(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("must not be blank")
    return value


NonBlank = Annotated[str, AfterValidator(_not_blank)]


class AskRequest(BaseModel):
    question: NonBlank = Field(min_length=1, max_length=2000)
    stream: bool = Field(
        default=True, description="Server-sent events (default) or one JSON response"
    )
    document_ids: list[uuid.UUID] | None = Field(default=None, max_length=100)


class FieldIn(BaseModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,49}$")
    description: str = Field(min_length=1, max_length=300)
    type: FieldType = "string"


class ExtractRequest(BaseModel):
    query: NonBlank = Field(min_length=1, max_length=1000, description="What to search for")
    fields: list[FieldIn] = Field(min_length=1, max_length=20)
    document_ids: list[uuid.UUID] | None = Field(default=None, max_length=100)

    @field_validator("fields")
    @classmethod
    def unique_names(cls, fields: list[FieldIn]) -> list[FieldIn]:
        names = [f.name for f in fields]
        if len(names) != len(set(names)):
            raise ValueError("field names must be unique")
        return fields


def _service(request: Request, name: str):
    service = getattr(request.app.state, name, None)
    if service is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Answer generation is not configured: LLM_PROVIDER=anthropic needs "
            "ANTHROPIC_API_KEY (or use LLM_PROVIDER=extractive)",
        )
    return service


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _upstream_error(exc: Exception) -> tuple[int, str]:
    if isinstance(exc, anthropic.RateLimitError):
        return (
            status.HTTP_429_TOO_MANY_REQUESTS,
            "The language model is rate limited; retry shortly",
        )
    if isinstance(exc, anthropic.APIError):
        return status.HTTP_502_BAD_GATEWAY, "The language model request failed"
    return status.HTTP_500_INTERNAL_SERVER_ERROR, "Internal error while generating the answer"


@router.post(
    "/ask",
    responses={200: {"description": "JSON answer, or text/event-stream when stream=true"}},
)
async def ask(body: AskRequest, request: Request, principal: PrincipalDep):
    """Answer a question from the caller's documents, with citations.

    Streaming events, in order: `sources` (the numbered passages), many `text` and
    `citation` deltas, then `done` with the full answer, usage, cost and timings.
    An `error` event replaces `done` if generation fails mid-stream.
    """
    service: AnswerService = _service(request, "answer_service")
    events = service.stream(body.question, principal, body.document_ids)

    if body.stream:

        async def event_stream() -> AsyncIterator[str]:
            try:
                async for name, payload in events:
                    yield _sse(name, payload)
            except Exception as exc:
                logger.exception("answer generation failed")
                code, message = _upstream_error(exc)
                yield _sse("error", {"status": code, "message": message})

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    sources: list[dict] = []
    try:
        async for name, payload in events:
            if name == "sources":
                sources = payload["sources"]
            elif name == "done":
                return {**payload, "sources": sources}
    except Exception as exc:
        logger.exception("answer generation failed")
        code, message = _upstream_error(exc)
        raise HTTPException(code, message) from exc
    raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Answer stream ended early")


@router.post("/extract")
async def extract(body: ExtractRequest, request: Request, principal: PrincipalDep) -> dict:
    """Fill named fields from the caller's documents as schema-validated JSON.

    Each field returns `value` (null if the documents don't state it) and `sources`, the
    numbers of the passages that support it.
    """
    service: ExtractionService = _service(request, "extraction_service")
    fields = [FieldSpec(f.name, f.description, f.type) for f in body.fields]
    try:
        result = await service.extract(body.query, fields, principal, body.document_ids)
    except Exception as exc:
        logger.exception("extraction failed")
        code, message = _upstream_error(exc)
        raise HTTPException(code, message) from exc
    return result.to_dict()
