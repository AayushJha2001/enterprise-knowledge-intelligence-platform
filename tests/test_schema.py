"""Checks that the migrated database matches what the app expects."""

import asyncio

from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import get_settings
from app.db.models import EMBEDDING_DIM, Base


def _run(query_fn):
    async def runner():
        engine = create_async_engine(get_settings().database_url, poolclass=NullPool)
        try:
            async with engine.connect() as conn:
                return await query_fn(conn)
        finally:
            await engine.dispose()

    return asyncio.run(runner())


def test_pgvector_extension_installed() -> None:
    async def query(conn):
        result = await conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'"))
        return result.scalar_one_or_none()

    assert _run(query) == 1


def test_search_indexes_exist() -> None:
    async def query(conn):
        result = await conn.execute(
            text("SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'chunks'")
        )
        return dict(result.all())

    indexes = _run(query)
    assert "USING hnsw" in indexes["ix_chunks_embedding_hnsw"]
    assert "vector_cosine_ops" in indexes["ix_chunks_embedding_hnsw"]
    assert "USING gin" in indexes["ix_chunks_tsv"]


def test_embedding_column_dimension_matches_settings() -> None:
    async def query(conn):
        result = await conn.execute(
            text(
                "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
            )
        )
        return result.scalar_one()

    assert _run(query) == f"vector({EMBEDDING_DIM})"
    assert get_settings().embedding_dim == EMBEDDING_DIM


def test_migrations_match_models() -> None:
    """Fails if someone changes app/db/models.py without writing a migration."""

    def diff(sync_conn):
        return compare_metadata(MigrationContext.configure(sync_conn), Base.metadata)

    async def query(conn):
        return await conn.run_sync(diff)

    assert _run(query) == []
