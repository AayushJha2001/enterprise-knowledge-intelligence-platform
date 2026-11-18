"""Structured extraction: fill named fields from retrieved passages, as validated JSON.

Uses structured outputs (a JSON schema the response is guaranteed to match). The API does
not allow citations together with structured outputs, so grounding comes from each field
listing the numbers of the passages that support it.
"""

import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Literal

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.core.security import Principal
from app.generation.answer import retrieve_for_generation, source_payload
from app.generation.gateway import ClaudeGateway
from app.generation.pricing import Usage, estimate_cost_usd
from app.generation.prompts import EXTRACT_SYSTEM_PROMPT, user_message
from app.retrieval.search import Retriever

FieldType = Literal["string", "number", "integer", "boolean", "string_list"]

_JSON_TYPES: dict[str, dict[str, Any]] = {
    "string": {"type": "string"},
    "number": {"type": "number"},
    "integer": {"type": "integer"},
    "boolean": {"type": "boolean"},
    "string_list": {"type": "array", "items": {"type": "string"}},
}


@dataclass(frozen=True)
class FieldSpec:
    name: str
    description: str
    type: FieldType = "string"


def build_schema(fields: list[FieldSpec]) -> dict[str, Any]:
    def field_schema(spec: FieldSpec) -> dict[str, Any]:
        return {
            "type": "object",
            "description": spec.description,
            "properties": {
                "value": {"anyOf": [_JSON_TYPES[spec.type], {"type": "null"}]},
                "sources": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["value", "sources"],
            "additionalProperties": False,
        }

    return {
        "type": "object",
        "properties": {spec.name: field_schema(spec) for spec in fields},
        "required": [spec.name for spec in fields],
        "additionalProperties": False,
    }


def _instructions(query: str, fields: list[FieldSpec]) -> str:
    lines = [f"Topic: {query}", "", "Fields to fill:"]
    lines += [f"- {f.name} ({f.type}): {f.description}" for f in fields]
    return "\n".join(lines)


@dataclass
class ExtractResult:
    status: Literal["extracted", "no_relevant_sources", "refused", "truncated"]
    fields: dict[str, dict[str, Any]]  # name -> {"value": ..., "sources": [n, ...]}
    sources: list[dict[str, Any]]
    model: str | None
    usage: Usage
    cost_usd: float | None
    timings_ms: dict[str, float]
    request_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExtractionService:
    def __init__(
        self,
        retriever: Retriever,
        gateway: ClaudeGateway,
        session_factory: async_sessionmaker[AsyncSession],
        settings: Settings,
    ) -> None:
        self.retriever = retriever
        self.gateway = gateway
        self.session_factory = session_factory
        self.settings = settings

    async def extract(
        self,
        query: str,
        fields: list[FieldSpec],
        principal: Principal,
        document_ids: list[uuid.UUID] | None = None,
    ) -> ExtractResult:
        sources, search, relevant = await retrieve_for_generation(
            self.retriever, self.session_factory, self.settings, query, principal, document_ids
        )
        timings = {f"retrieval_{k}": v for k, v in search.timings_ms.items()}
        empty = {f.name: {"value": None, "sources": []} for f in fields}
        payload = [source_payload(s) for s in sources]

        if not relevant:
            return ExtractResult("no_relevant_sources", empty, payload, None, Usage(), 0.0, timings)

        started = time.perf_counter()
        result = await self.gateway.create_json(
            system=EXTRACT_SYSTEM_PROMPT,
            messages=[user_message(sources, _instructions(query, fields), citations=False)],
            schema=build_schema(fields),
        )
        timings["llm_total"] = round((time.perf_counter() - started) * 1000, 2)

        status = {"refusal": "refused", "max_tokens": "truncated"}.get(
            result.stop_reason or "", "extracted"
        )
        extracted = dict(empty)
        if status == "extracted":
            valid = range(1, len(sources) + 1)
            for spec in fields:
                item = result.data.get(spec.name) or {}
                numbers = [n for n in item.get("sources", []) if n in valid]
                extracted[spec.name] = {
                    "value": item.get("value"),
                    "sources": list(dict.fromkeys(numbers)),
                }
        return ExtractResult(
            status=status,
            fields=extracted,
            sources=payload,
            model=result.model,
            usage=result.usage,
            cost_usd=estimate_cost_usd(result.model, result.usage),
            timings_ms=timings,
            request_id=result.request_id,
        )
