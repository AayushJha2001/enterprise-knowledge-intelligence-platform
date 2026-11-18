"""Reciprocal Rank Fusion (RRF): merge ranked lists without comparing their scores.

Vector distances and full-text ranks live on different scales, so adding them is
meaningless. RRF only uses each item's *position*: score = sum(1 / (k + rank)). An item
ranked well by both retrievers beats one ranked first by only one of them.
(Cormack, Clarke & Buettcher, SIGIR 2009.)
"""

from collections.abc import Hashable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class FusedItem[T: Hashable]:
    id: T
    score: float
    ranks: dict[str, int]  # retriever name -> 1-based rank in that retriever's list


def reciprocal_rank_fusion[T: Hashable](
    ranked_lists: dict[str, Sequence[T]], k: int = 60
) -> list[FusedItem[T]]:
    scores: dict[T, float] = {}
    ranks: dict[T, dict[str, int]] = {}
    for name, items in ranked_lists.items():
        for rank, item in enumerate(items, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(item, {})[name] = rank
    ordered = sorted(scores, key=lambda item: (-scores[item], min(ranks[item].values())))
    return [FusedItem(id=item, score=scores[item], ranks=ranks[item]) for item in ordered]
