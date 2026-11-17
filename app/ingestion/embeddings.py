"""Text embedding models behind one small interface, so they can be swapped and faked."""

import hashlib
import logging
import math
from typing import Protocol

from app.core.config import Settings

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    model_name: str
    dim: int

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    """Local ONNX embedding model (default BAAI/bge-small-en-v1.5, 384 dimensions).

    Runs on CPU with no API key. The model (~70 MB) is downloaded once into
    `model_cache_dir` on first use. Calls are blocking: run them in a thread from async code.
    """

    def __init__(self, model_name: str, dim: int, cache_dir: str, batch_size: int) -> None:
        from fastembed import TextEmbedding  # heavy import, only when actually used

        logger.info("loading embedding model %s", model_name)
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self.model_name = model_name
        self.dim = dim
        self._batch_size = batch_size

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        vectors = [
            v.tolist() for v in self._model.passage_embed(texts, batch_size=self._batch_size)
        ]
        self._check_dim(vectors)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        # bge models expect queries and passages to be encoded differently.
        vectors = [v.tolist() for v in self._model.query_embed(text)]
        self._check_dim(vectors)
        return vectors[0]

    def _check_dim(self, vectors: list[list[float]]) -> None:
        if vectors and len(vectors[0]) != self.dim:
            raise ValueError(
                f"{self.model_name} returned {len(vectors[0])} dimensions; "
                f"EMBEDDING_DIM is {self.dim}"
            )


class HashingEmbedder:
    """Deterministic bag-of-words vectors for tests and offline development.

    Texts sharing words get similar vectors, which is enough to test the pipeline,
    but it has no understanding of meaning. Never use it for real retrieval.
    """

    model_name = "hashing-bow"

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for word in text.lower().split():
            digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "hash":
        return HashingEmbedder(settings.embedding_dim)
    return FastEmbedEmbedder(
        model_name=settings.embedding_model,
        dim=settings.embedding_dim,
        cache_dir=str(settings.model_cache_dir),
        batch_size=settings.embed_batch_size,
    )
