from functools import lru_cache

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

    # Dimension of the embedding model chosen in Phase 2 (bge-small-en-v1.5 = 384).
    # Changing it requires a new migration for the chunks.embedding column.
    embedding_dim: int = 384


@lru_cache
def get_settings() -> Settings:
    return Settings()
