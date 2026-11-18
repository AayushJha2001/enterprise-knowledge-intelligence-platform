from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEV_JWT_SECRET = "dev-only-insecure-jwt-secret-change-me"  # noqa: S105 - documented dev default


class Settings(BaseSettings):
    """All runtime configuration, read from environment variables (or a local .env file)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Enterprise Knowledge Intelligence Platform"
    environment: str = "local"
    log_level: str = "INFO"

    database_url: str = "postgresql+asyncpg://ekip:ekip@localhost:5433/ekip"
    db_pool_size: int = 5
    db_max_overflow: int = 10

    redis_url: str = "redis://localhost:6380/0"

    # How long a readiness probe waits on each dependency before reporting it down.
    health_check_timeout_s: float = 2.0

    # --- Uploads ---
    upload_dir: Path = Path("data/uploads")
    max_upload_bytes: int = 25 * 1024 * 1024

    # --- Embeddings ---
    # "fastembed" runs a local ONNX model; "hash" is a deterministic stand-in for tests
    # and offline development (no model download, no semantic meaning).
    embedding_provider: Literal["fastembed", "hash"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Changing the dimension requires a migration for chunks.embedding.
    embedding_dim: int = 384
    embed_batch_size: int = 32
    model_cache_dir: Path = Path("data/models")

    # --- Chunking (estimated tokens) ---
    # bge-small reads at most 512 tokens; the target leaves room for the title/section
    # header that is prepended to each chunk before embedding.
    chunk_target_tokens: int = 350
    chunk_max_tokens: int = 450
    chunk_overlap_tokens: int = 50

    # --- Retrieval ---
    # Candidates taken from each retriever (vector and keyword) before fusion.
    retrieval_candidates: int = 40
    # Reciprocal Rank Fusion constant: higher flattens the advantage of top ranks.
    rrf_k: int = 60
    # The reranker is the slowest step: its cost grows with candidates x passage length.
    # On a laptop CPU, 20 candidates x 1200 chars takes ~1 s; a server CPU or GPU is far
    # faster. Tune both against the Phase 6 retrieval evals, not by feel.
    rerank_candidates: int = 20
    rerank_max_chars: int = 1200
    # HNSW search breadth; higher = better recall, slower. pgvector default is 40.
    hnsw_ef_search: int = 100
    # "fastembed" runs a local cross-encoder; "none" skips reranking (tests, ablations).
    reranker_provider: Literal["fastembed", "none"] = "fastembed"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # --- Authentication ---
    # HS256 signing key for access tokens. The default is only for local development;
    # the app refuses to start with it when ENVIRONMENT=production.
    jwt_secret: SecretStr = SecretStr(DEV_JWT_SECRET)
    access_token_ttl_minutes: int = 60
    # Lock an email out after this many failed logins within the window (per client IP).
    login_max_failures: int = 5
    login_lockout_s: int = 900

    # --- Caching (Redis) ---
    cache_enabled: bool = True
    search_cache_ttl_s: int = 600
    answer_cache_ttl_s: int = 3600
    query_embedding_cache_ttl_s: int = 86400

    # --- Answer generation ---
    # "auto": Claude when ANTHROPIC_API_KEY is set, otherwise the free offline extractive
    # answerer. "anthropic" requires the key (else /ask returns 503). "extractive" never
    # calls an API.
    llm_provider: Literal["auto", "anthropic", "extractive"] = "auto"
    anthropic_api_key: SecretStr | None = None
    answer_model: str = "claude-opus-5-5"
    # Thinking depth. "low" keeps grounded Q&A fast; raise it if the evals show answers
    # missing multi-step reasoning across sources. (Opus 5.5's own default is "medium".)
    answer_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    answer_max_tokens: int = 64000
    # Passages given to Claude as search results.
    answer_context_chunks: int = 8
    # If the best reranked passage scores below this, answer "not found" without calling
    # Claude: cheaper, and it can't be tempted to answer from general knowledge.
    answer_min_relevance: float = 0.02
    # Server-side fallback: if a request is declined by a safety classifier, the API
    # retries it on another model in the same call.
    llm_fallbacks_enabled: bool = True
    llm_timeout_s: float = 120.0

    @field_validator("anthropic_api_key", mode="before")
    @classmethod
    def empty_key_means_unset(cls, value: object) -> object:
        # docker compose passes "" when the variable isn't set anywhere.
        return None if isinstance(value, str) and not value.strip() else value

    # --- Ingestion worker ---
    worker_poll_interval_s: float = 2.0
    # A document stuck in "processing" this long (worker crashed) is put back in the queue.
    worker_stale_after_s: int = 900
    worker_max_attempts: int = 3

    def check_production_safety(self) -> None:
        """Fail fast on settings that must never reach production."""
        if self.environment == "production" and (
            self.jwt_secret.get_secret_value() == DEV_JWT_SECRET
            or len(self.jwt_secret.get_secret_value()) < 32
        ):
            raise RuntimeError("Set JWT_SECRET to a random value of 32+ characters in production")


@lru_cache
def get_settings() -> Settings:
    return Settings()
