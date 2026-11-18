"""The only module that talks to the Anthropic SDK.

It turns SDK stream events into a few plain events the rest of the app understands, so
answer logic can be tested with a fake gateway and the SDK can be upgraded in one place.
"""

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import anthropic

from app.core.config import Settings
from app.generation.pricing import Usage
from app.ingestion.embeddings import Embedder

logger = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass(frozen=True)
class TextDelta:
    block_index: int
    text: str


@dataclass(frozen=True)
class CitationDelta:
    block_index: int
    search_result_index: int
    start_block_index: int
    end_block_index: int
    cited_text: str


@dataclass(frozen=True)
class Completed:
    model: str
    stop_reason: str | None
    usage: Usage
    request_id: str | None = None
    refusal_category: str | None = None


StreamEvent = TextDelta | CitationDelta | Completed


@dataclass(frozen=True)
class JsonResult:
    data: dict[str, Any]
    model: str
    stop_reason: str | None
    usage: Usage
    request_id: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class ClaudeGateway:
    def __init__(self, client: anthropic.AsyncAnthropic, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    @property
    def cache_id(self) -> str:
        """Identifies the generator configuration in answer-cache keys."""
        return f"{self._settings.answer_model}:{self._settings.answer_effort}"

    def _common(self, system: str, messages: list[dict], max_tokens: int) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self._settings.answer_model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
            "output_config": {"effort": self._settings.answer_effort},
        }
        if self._settings.llm_fallbacks_enabled:
            params["betas"] = [FALLBACK_BETA]
            params["fallbacks"] = "default"
        return params

    async def stream(self, *, system: str, messages: list[dict]) -> AsyncIterator[StreamEvent]:
        params = self._common(system, messages, self._settings.answer_max_tokens)
        async with self._client.beta.messages.stream(**params) as stream:
            async for event in stream:
                if event.type != "content_block_delta":
                    continue  # thinking, block start/stop, message-level events
                delta = event.delta
                if delta.type == "text_delta":
                    yield TextDelta(event.index, delta.text)
                elif delta.type == "citations_delta":
                    citation = delta.citation
                    if citation.type == "search_result_location":
                        yield CitationDelta(
                            block_index=event.index,
                            search_result_index=citation.search_result_index,
                            start_block_index=citation.start_block_index,
                            end_block_index=citation.end_block_index,
                            cited_text=citation.cited_text,
                        )
            final = await stream.get_final_message()
        yield Completed(
            model=final.model,
            stop_reason=final.stop_reason,
            usage=Usage.from_api(final.usage),
            request_id=getattr(final, "_request_id", None),
            refusal_category=_refusal_category(final),
        )

    async def create_json(
        self, *, system: str, messages: list[dict], schema: dict[str, Any]
    ) -> JsonResult:
        params = self._common(system, messages, max_tokens=16000)
        params["output_config"]["format"] = {"type": "json_schema", "schema": schema}
        response = await self._client.beta.messages.create(**params)
        data: dict[str, Any] = {}
        if response.stop_reason not in ("refusal", "max_tokens"):
            text = next((b.text for b in response.content if b.type == "text"), "{}")
            data = json.loads(text)
        return JsonResult(
            data=data,
            model=response.model,
            stop_reason=response.stop_reason,
            usage=Usage.from_api(response.usage),
            request_id=getattr(response, "_request_id", None),
            extras={"refusal_category": _refusal_category(response)},
        )


def _refusal_category(message: object) -> str | None:
    if getattr(message, "stop_reason", None) != "refusal":
        return None
    details = getattr(message, "stop_details", None)
    return getattr(details, "category", None) if details else None


def build_gateway(settings: Settings, embedder: Embedder):
    """Claude if configured, else the offline extractive answerer (see LLM_PROVIDER)."""
    provider = settings.llm_provider
    if provider == "auto":
        provider = "anthropic" if settings.anthropic_api_key else "extractive"
    if provider == "extractive":
        from app.generation.extractive import ExtractiveGateway

        logger.info("answers use the offline extractive generator (no LLM)")
        return ExtractiveGateway(embedder)
    if settings.anthropic_api_key is None:
        logger.warning("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set: /ask disabled")
        return None
    logger.info("answers use Claude (%s)", settings.answer_model)
    client = anthropic.AsyncAnthropic(
        api_key=settings.anthropic_api_key.get_secret_value(),
        timeout=settings.llm_timeout_s,
        max_retries=2,
    )
    return ClaudeGateway(client, settings)
