"""Offline answer generator: no LLM, no API key, no cost.

Implements the same interface as AnthropicGateway and receives the exact messages the
LLM would (search_result blocks + the question). Instead of writing an answer, it *selects*
the sentences from those passages that best match the question, using the local
embedding model, and returns each with a citation to its passage and paragraph.

What it can't do, compared with an LLM: combine facts across sentences, resolve
conditions ("220 EUR in Paris" vs "150 EUR elsewhere"), rephrase, or say "no" in its own
words. It is a free stand-in that keeps the whole pipeline working end to end; set
ANTHROPIC_API_KEY (or LLM_PROVIDER=anthropic) to use the Anthropic API instead.
"""

import asyncio
import math
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from app.generation.gateway import CitationDelta, Completed, JsonResult, StreamEvent, TextDelta
from app.generation.pricing import Usage
from app.ingestion.embeddings import Embedder

MODEL_NAME = "extractive-v1"
NO_DIRECT_ANSWER = (
    "I found related passages, but none of them answers this directly. See the sources below."
)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")


@dataclass(frozen=True)
class Candidate:
    search_result_index: int  # which passage (0-based, as in the API's citations)
    paragraph_index: int  # which text block inside it
    sentence: str
    paragraph: str


def parse_messages(messages: list[dict]) -> tuple[list[list[str]], str]:
    """Paragraphs of each search_result block, and the trailing text (the question)."""
    passages: list[list[str]] = []
    text = ""
    for block in messages[-1]["content"]:
        if block["type"] == "search_result":
            passages.append([part["text"] for part in block["content"]])
        elif block["type"] == "text":
            text = block["text"]
    return passages, text


_ITEM_START = re.compile(r"^\s*(?:[-*+]\s|\d+[.)]\s)")


def _logical_lines(paragraph: str) -> list[str]:
    """Undo hard line wrapping, keeping list items and table rows as separate lines.

    Markdown and plain text are often wrapped at ~80-100 columns, so a line break inside
    a paragraph is usually not the end of a sentence.
    """
    lines: list[str] = []
    for raw in paragraph.splitlines():
        line = raw.strip()
        if not line:
            continue
        starts_new = not lines or _ITEM_START.match(raw) or " | " in line or " | " in lines[-1]
        if starts_new:
            lines.append(line)
        else:
            lines[-1] = f"{lines[-1]} {line}"
    return lines


def candidate_sentences(passages: list[list[str]]) -> list[Candidate]:
    candidates = []
    seen: set[str] = set()
    for sri, paragraphs in enumerate(passages):
        for pi, paragraph in enumerate(paragraphs):
            if paragraph.lstrip().startswith(("```", "~~~")):
                continue  # code blocks aren't answers
            for line in _logical_lines(paragraph):
                for sentence in _SENTENCE_END.split(line.strip()):
                    sentence = sentence.strip(" -*")
                    key = " ".join(sentence.lower().split())
                    # Skip fragments, and repeats from overlapping chunks.
                    if len(sentence.split()) < 4 or key in seen:
                        continue
                    seen.add(key)
                    candidates.append(Candidate(sri, pi, sentence, paragraph))
    return candidates


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class ExtractiveGateway:
    cache_id = MODEL_NAME

    def __init__(self, embedder: Embedder, max_sentences: int = 3, margin: float = 0.06) -> None:
        self.embedder = embedder
        self.max_sentences = max_sentences
        # Include further sentences only if they score nearly as well as the best one.
        self.margin = margin

    def _scores(self, queries: list[str], candidates: list[Candidate]) -> list[list[float]]:
        sentence_vectors = self.embedder.embed_passages([c.sentence for c in candidates])
        rows = []
        for query in queries:
            query_vector = self.embedder.embed_query(query)
            # Slight preference for higher-ranked passages (retrieval order is a signal).
            rows.append(
                [
                    _cosine(query_vector, vector) - 0.005 * c.search_result_index
                    for c, vector in zip(candidates, sentence_vectors, strict=True)
                ]
            )
        return rows

    def select(self, candidates: list[Candidate], scores: list[float]) -> list[Candidate]:
        ranked = sorted(zip(scores, range(len(candidates)), strict=True), reverse=True)
        if not ranked:
            return []
        best = ranked[0][0]
        chosen = [i for score, i in ranked if score >= best - self.margin][: self.max_sentences]
        return [candidates[i] for i in sorted(chosen)]  # document order reads naturally

    async def stream(self, *, system: str, messages: list[dict]) -> AsyncIterator[StreamEvent]:
        passages, question = parse_messages(messages)
        candidates = candidate_sentences(passages)
        chosen: list[Candidate] = []
        if candidates:
            [scores] = await asyncio.to_thread(self._scores, [question], candidates)
            chosen = self.select(candidates, scores)

        if not chosen:
            yield TextDelta(0, NO_DIRECT_ANSWER)
        for block, candidate in enumerate(chosen):
            # Leading space, so a citation marker lands right after its sentence:
            # "First.[1] Second.[2]"
            yield TextDelta(block, (" " if block else "") + candidate.sentence)
            yield CitationDelta(
                block_index=block,
                search_result_index=candidate.search_result_index,
                start_block_index=candidate.paragraph_index,
                end_block_index=candidate.paragraph_index + 1,
                cited_text=candidate.paragraph,
            )
        yield Completed(model=MODEL_NAME, stop_reason="end_turn", usage=Usage())

    async def create_json(
        self, *, system: str, messages: list[dict], schema: dict[str, Any]
    ) -> JsonResult:
        """Best-effort field filling: the best-matching sentence per field.

        Numbers are the first number in that sentence and booleans are never inferred,
        so expect misses on anything conditional. An LLM is needed for real extraction.
        """
        passages, instructions = parse_messages(messages)
        topic = instructions.splitlines()[0].removeprefix("Topic: ") if instructions else ""
        candidates = candidate_sentences(passages)
        fields = schema["properties"]
        data: dict[str, Any] = {name: {"value": None, "sources": []} for name in fields}
        if not candidates:
            return JsonResult(data, MODEL_NAME, "end_turn", Usage())

        queries = [f"{topic}: {spec.get('description', name)}" for name, spec in fields.items()]
        all_scores = await asyncio.to_thread(self._scores, queries, candidates)
        for (name, spec), scores in zip(fields.items(), all_scores, strict=True):
            best = candidates[max(range(len(scores)), key=scores.__getitem__)]
            value = _coerce(best.sentence, spec["properties"]["value"]["anyOf"][0])
            if value is not None:
                data[name] = {"value": value, "sources": [best.search_result_index + 1]}
        return JsonResult(data, MODEL_NAME, "end_turn", Usage())


def _coerce(sentence: str, value_schema: dict[str, Any]) -> Any:
    kind = value_schema.get("type")
    if kind in ("number", "integer"):
        match = _NUMBER.search(sentence)
        if match is None:
            return None
        number = float(match.group().replace(",", ""))
        return int(number) if kind == "integer" else number
    if kind == "array":
        return [sentence]
    if kind == "string":
        return sentence
    return None  # booleans: can't be inferred by sentence matching
