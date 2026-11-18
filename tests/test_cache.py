"""Redis caching: hits, access-scope isolation, invalidation, and failure tolerance."""

import os
from collections.abc import Iterator

import pytest
import redis
from fastapi.testclient import TestClient
from redis.exceptions import RedisError

from tests.helpers import bearer, create_group, create_user, ingest

HR_DOC = b"# HR Handbook\n\n## Leave\n\nParents receive sixteen weeks of paid parental leave."
ENG_DOC = b"# Runbook\n\n## Deploys\n\nNever deploy on Fridays after four in the afternoon."


@pytest.fixture
def cached(client: TestClient) -> Iterator[TestClient]:
    cache = client.app.state.cache
    cache.enabled = True
    yield client
    cache.enabled = False


@pytest.fixture
def docs(cached: TestClient) -> dict:
    hr, eng = create_group("hr"), create_group("eng")
    return {
        "hr": hr,
        "hr_doc": ingest(cached, "hr.md", HR_DOC, group_ids=[hr]),
        "eng_doc": ingest(cached, "eng.md", ENG_DOC, group_ids=[eng]),
        "hana": create_user("hana@example.com", [hr]),
        "harry": create_user("harry@example.com", [hr]),
        "eli": create_user("eli@example.com", [eng]),
    }


def search(client: TestClient, query: str, user=None) -> dict:
    headers = bearer(user) if user else {}
    response = client.post("/search", json={"query": query}, headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


def ask(client: TestClient, question: str, user=None) -> dict:
    headers = bearer(user) if user else {}
    response = client.post("/ask", json={"question": question, "stream": False}, headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


class TestSearchCache:
    def test_repeat_query_is_served_from_cache(self, cached: TestClient, docs) -> None:
        first = search(cached, "paid parental leave")
        second = search(cached, "  Paid   PARENTAL leave ")  # normalised to the same key
        assert first["cached"] is False
        assert second["cached"] is True
        assert second["results"] == first["results"]

    def test_users_with_same_groups_share_entries(self, cached: TestClient, docs) -> None:
        assert search(cached, "parental leave", docs["hana"])["cached"] is False
        assert search(cached, "parental leave", docs["harry"])["cached"] is True

    def test_users_with_different_access_never_share(self, cached: TestClient, docs) -> None:
        hr_view = search(cached, "parental leave deploy", docs["hana"])
        eng_view = search(cached, "parental leave deploy", docs["eli"])
        assert eng_view["cached"] is False
        assert {r["document_id"] for r in hr_view["results"]} == {docs["hr_doc"]}
        assert {r["document_id"] for r in eng_view["results"]} == {docs["eng_doc"]}

    def test_sharing_change_invalidates(self, cached: TestClient, docs) -> None:
        before = search(cached, "deploy fridays", docs["hana"])
        assert docs["eng_doc"] not in {r["document_id"] for r in before["results"]}

        cached.put(
            f"/admin/documents/{docs['eng_doc']}/groups", json={"group_ids": [str(docs["hr"])]}
        )
        after = search(cached, "deploy fridays", docs["hana"])
        assert after["cached"] is False
        assert docs["eng_doc"] in {r["document_id"] for r in after["results"]}

    def test_new_document_invalidates(self, cached: TestClient, docs) -> None:
        search(cached, "expense claims")
        ingest(cached, "expenses.md", b"# Expenses\n\n## Claims\n\nSubmit expense claims monthly.")
        assert search(cached, "expense claims")["cached"] is False

    def test_deleting_a_document_invalidates(self, cached: TestClient, docs) -> None:
        search(cached, "parental leave")
        cached.delete(f"/documents/{docs['hr_doc']}")
        after = search(cached, "parental leave")
        assert after["cached"] is False
        assert docs["hr_doc"] not in {r["document_id"] for r in after["results"]}

    def test_query_embedding_is_cached(self, cached: TestClient, docs) -> None:
        search(cached, "parental leave")
        with redis.Redis.from_url(os.environ["REDIS_URL"]) as r:
            assert list(r.scan_iter("ekip:embedding:*"))


class TestAnswerCache:
    def test_repeat_question_is_replayed(self, cached: TestClient, docs) -> None:
        first = ask(cached, "How many weeks of paid parental leave?")
        second = ask(cached, "how many weeks of paid parental leave?")
        assert first["cached"] is False and second["cached"] is True
        assert second["answer"] == first["answer"]
        assert second["segments"] == first["segments"]
        assert second["sources"] == first["sources"]
        assert second["cost_usd"] == 0.0
        assert set(second["timings_ms"]) == {"cache_lookup"}

    def test_streamed_replay_has_the_same_events(self, cached: TestClient, docs) -> None:
        def event_names() -> list[str]:
            raw = cached.post("/ask", json={"question": "paid parental leave weeks"}).text
            return [line[7:] for line in raw.splitlines() if line.startswith("event: ")]

        assert event_names() == event_names()

    def test_not_shared_across_access_scopes(self, cached: TestClient, docs) -> None:
        ask(cached, "paid parental leave", docs["hana"])
        other = ask(cached, "paid parental leave", docs["eli"])
        assert other["cached"] is False
        assert all(s["document_id"] != docs["hr_doc"] for s in other["sources"])


def test_redis_failure_degrades_to_uncached(cached: TestClient, docs, monkeypatch) -> None:
    async def broken(*args, **kwargs):
        raise RedisError("connection refused")

    redis_client = cached.app.state.cache.redis
    monkeypatch.setattr(redis_client, "get", broken)
    monkeypatch.setattr(redis_client, "set", broken)
    first = search(cached, "parental leave")
    second = search(cached, "parental leave")
    assert first["results"] and second["cached"] is False
    assert ask(cached, "paid parental leave")["status"] == "answered"
