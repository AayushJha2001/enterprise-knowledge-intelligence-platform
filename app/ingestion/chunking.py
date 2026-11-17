"""Heading-aware chunking.

Rules:
1. A chunk never spans two sections, so every chunk has exactly one heading path to cite.
2. Within a section, paragraphs are packed until the target size; no chunk exceeds the max.
3. Consecutive chunks in the same section overlap by a few sentences' worth of words, so
   a fact split across a boundary is still retrievable from one chunk.
4. Oversized paragraphs are split on sentence boundaries, and as a last resort on words.

Sizes are *estimated* tokens (see `estimate_tokens`). The estimate is deliberately
conservative so chunks stay under the embedding model's 512-token input limit.
"""

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import groupby

from app.ingestion.parsers import Block

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def estimate_tokens(text: str) -> int:
    """Upper-leaning estimate for English text with WordPiece/BPE tokenizers.

    Words * 1.3 covers prose; chars / 4 covers code, numbers and identifiers, which
    tokenize into many short pieces. Taking the max keeps both kinds under the limit.
    """
    if not text:
        return 0
    return max(1, math.ceil(max(len(text.split()) * 1.3, len(text) / 4)))


@dataclass(frozen=True)
class ChunkDraft:
    index: int
    text: str
    token_count: int
    section_path: tuple[str, ...]
    page_start: int | None
    page_end: int | None


@dataclass(frozen=True)
class _Piece:
    text: str
    page: int | None


def _split_words(text: str, max_tokens: int) -> list[str]:
    words = text.split()
    parts: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if current and estimate_tokens(candidate) > max_tokens:
            parts.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        parts.append(" ".join(current))
    return parts


def _split_oversized(text: str, max_tokens: int) -> list[str]:
    """Split one long paragraph into parts that each fit in max_tokens."""
    parts: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(text):
        if estimate_tokens(sentence) > max_tokens:
            if current:
                parts.append(current)
                current = ""
            parts.extend(_split_words(sentence, max_tokens))
            continue
        candidate = f"{current} {sentence}".strip()
        if current and estimate_tokens(candidate) > max_tokens:
            parts.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def _to_pieces(blocks: Iterable[Block], max_tokens: int) -> list[_Piece]:
    pieces = []
    for block in blocks:
        if estimate_tokens(block.text) <= max_tokens:
            pieces.append(_Piece(block.text, block.page))
        else:
            for part in _split_oversized(block.text, max_tokens):
                pieces.append(_Piece(part, block.page))
    return pieces


def _overlap_tail(text: str, overlap_tokens: int) -> str:
    """The last few words of a chunk, at most overlap_tokens long."""
    if overlap_tokens <= 0:
        return ""
    words = text.split()
    tail: list[str] = []
    for word in reversed(words):
        if estimate_tokens(" ".join([word, *tail])) > overlap_tokens:
            break
        tail.insert(0, word)
    # Don't repeat the whole previous chunk when it was tiny.
    return "" if len(tail) == len(words) else " ".join(tail)


def chunk_blocks(
    blocks: list[Block],
    *,
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> list[ChunkDraft]:
    if not 0 <= overlap_tokens < target_tokens <= max_tokens:
        raise ValueError("Expected 0 <= overlap < target <= max")

    drafts: list[ChunkDraft] = []

    def join(prefix: str, parts: list[_Piece]) -> str:
        text = "\n\n".join(p.text for p in parts)
        return f"{prefix} {text}" if prefix else text

    def emit(parts: list[_Piece], section: tuple[str, ...], prefix: str) -> str:
        text = join(prefix, parts)
        pages = [p.page for p in parts if p.page is not None]
        drafts.append(
            ChunkDraft(
                index=len(drafts),
                text=text,
                token_count=estimate_tokens(text),
                section_path=section,
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
            )
        )
        return text

    for section, section_blocks in groupby(blocks, key=lambda b: b.section_path):
        current: list[_Piece] = []
        prefix = ""  # overlap carried from the previous chunk in this section

        for piece in _to_pieces(section_blocks, max_tokens):
            if current and estimate_tokens(join(prefix, [*current, piece])) > target_tokens:
                previous = emit(current, section, prefix)
                current = []
                prefix = _overlap_tail(previous, overlap_tokens)
                if estimate_tokens(join(prefix, [piece])) > max_tokens:
                    prefix = ""
            current.append(piece)

        if current:
            emit(current, section, prefix)

    return drafts


def embedding_text(chunk: ChunkDraft, document_title: str | None) -> str:
    """What gets embedded: the chunk plus a short header saying where it comes from.

    A chunk like "It is capped at 20 days." means little on its own; with the header
    "Employee Handbook > Leave > Annual leave" it matches queries about annual leave.
    """
    header = " > ".join(part for part in (document_title, *chunk.section_path) if part)
    return f"{header}\n\n{chunk.text}" if header else chunk.text
