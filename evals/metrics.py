"""Retrieval and answer metrics. Pure functions, no I/O.

Relevance is judged at the *section* level: a retrieved chunk is relevant if its heading
path ends with one of the question's expected sections. Several chunks can share a
section; each expected section counts once.
"""

import math
import re
from collections.abc import Sequence


def last_section(section_path: str | None) -> str:
    return (section_path or "").split(" > ")[-1].strip().casefold()


def is_relevant(section_path: str | None, expected: Sequence[str]) -> bool:
    return last_section(section_path) in {e.strip().casefold() for e in expected}


def first_relevant_rank(section_paths: Sequence[str | None], expected: Sequence[str]) -> int | None:
    """1-based rank of the first relevant result, or None if none is relevant."""
    for rank, path in enumerate(section_paths, start=1):
        if is_relevant(path, expected):
            return rank
    return None


def hit_at_k(rank: int | None, k: int) -> float:
    return 1.0 if rank is not None and rank <= k else 0.0


def reciprocal_rank(rank: int | None) -> float:
    return 1.0 / rank if rank else 0.0


def ndcg_at_k(section_paths: Sequence[str | None], expected: Sequence[str], k: int) -> float:
    """Binary-relevance nDCG: rewards finding every expected section, and finding it early."""
    wanted = {e.strip().casefold() for e in expected}
    seen: set[str] = set()
    dcg = 0.0
    for rank, path in enumerate(section_paths[:k], start=1):
        section = last_section(path)
        if section in wanted and section not in seen:
            seen.add(section)
            dcg += 1.0 / math.log2(rank + 1)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(wanted), k) + 1))
    return dcg / ideal if ideal else 0.0


_SPACES = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _SPACES.sub(" ", text.casefold()).strip()


def fact_recall(answer: str, facts: Sequence[str]) -> float:
    """Share of expected facts that appear in the answer (case/whitespace-insensitive)."""
    if not facts:
        return 1.0
    normalized = _normalize(answer)
    return sum(_normalize(f) in normalized for f in facts) / len(facts)


def citation_precision(
    cited_sections: Sequence[str | None], expected: Sequence[str]
) -> float | None:
    """Share of cited passages that come from an expected section (None: nothing cited)."""
    if not cited_sections:
        return None
    return sum(is_relevant(s, expected) for s in cited_sections) / len(cited_sections)


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile (p in 0-100), like numpy's default."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
