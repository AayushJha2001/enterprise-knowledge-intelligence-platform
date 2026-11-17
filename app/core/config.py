from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # --- Ingestion worker ---
    worker_poll_interval_s: float = 2.0
    # A document stuck in "processing" this long (worker crashed) is put back in the queue.
    worker_stale_after_s: int = 900
    worker_max_attempts: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()
