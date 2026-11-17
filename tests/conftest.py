"""Shared fixtures.

These are integration tests: they need the Postgres and Redis from docker compose, with
migrations applied. Run them with `docker compose run --rm api pytest`.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    # The `with` block runs the app's lifespan, so DB and Redis pools are created once.
    with TestClient(create_app()) as test_client:
        yield test_client
