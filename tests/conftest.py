"""Shared fixtures.

Integration tests run against a separate `<db>_test` database (created and migrated
automatically) so they never touch development data. Run them with:

    docker compose run --rm api pytest
"""

import os
import tempfile
from pathlib import Path

from sqlalchemy.engine import make_url

# --- Point the app at test resources BEFORE any app module reads its settings. ---
_url = make_url(
    os.environ.get("DATABASE_URL", "postgresql+asyncpg://ekip:ekip@localhost:5433/ekip")
)
if not _url.database.endswith("_test"):
    _url = _url.set(database=f"{_url.database}_test")
os.environ["DATABASE_URL"] = _url.render_as_string(hide_password=False)
os.environ["UPLOAD_DIR"] = tempfile.mkdtemp(prefix="ekip-test-uploads-")
os.environ["EMBEDDING_PROVIDER"] = "hash"
os.environ["RERANKER_PROVIDER"] = "none"
os.environ["ENVIRONMENT"] = "test"

from collections.abc import Iterator  # noqa: E402

import asyncpg  # noqa: E402
import pytest  # noqa: E402
from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.main import create_app  # noqa: E402
from tests.helpers import run_db  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


async def _ensure_test_database() -> None:
    conn = await asyncpg.connect(
        host=_url.host,
        port=_url.port or 5432,
        user=_url.username,
        password=_url.password,
        database="postgres",
    )
    try:
        exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", _url.database)
        if not exists:
            await conn.execute(f'CREATE DATABASE "{_url.database}"')
    finally:
        await conn.close()


@pytest.fixture(scope="session", autouse=True)
def test_database() -> None:
    import asyncio

    get_settings.cache_clear()
    asyncio.run(_ensure_test_database())
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    command.upgrade(config, "head")


@pytest.fixture(autouse=True)
def clean_tables(test_database: None) -> Iterator[None]:
    yield

    async def truncate(conn) -> None:
        await conn.execute(text("TRUNCATE users, groups, documents RESTART IDENTITY CASCADE"))
        await conn.commit()

    run_db(truncate)


@pytest.fixture(scope="session")
def client(test_database: None) -> Iterator[TestClient]:
    # The `with` block runs the app's lifespan, so DB and Redis pools are created once.
    with TestClient(create_app()) as test_client:
        yield test_client
