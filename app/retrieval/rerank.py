"""Cross-encoder reranking.

The embedding model encodes query and chunk *separately*, which is fast enough to search
everything but loses detail. A cross-encoder reads query and chunk *together* and scores
how well the chunk answers the query: far more accurate, but too slow to run on every
chunk. So retrieval narrows thousands of chunks to ~30 candidates and the reranker orders
those.
"""

import logging
import math
from typing import Protocol

from app.core.config import Settings

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    model_name: str

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Relevance of each passage to the query, in [0, 1]. Blocking."""
        ...


class FastEmbedReranker:
    def __init__(self, model_name: str, cache_dir: str) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        logger.info("loading reranker model %s", model_name)
        self._model = TextCrossEncoder(model_name=model_name, cache_dir=cache_dir)
        self.model_name = model_name

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        # The model returns raw logits; a sigmoid maps them to a 0-1 relevance that can
        # be thresholded later (e.g. "no passage is relevant enough to answer").
        logits = self._model.rerank(query, passages, batch_size=32)
        return [1.0 / (1.0 + math.exp(-float(logit))) for logit in logits]


def build_reranker(settings: Settings) -> Reranker | None:
    if settings.reranker_provider == "none":
        return None
    return FastEmbedReranker(settings.reranker_model, str(settings.model_cache_dir))
