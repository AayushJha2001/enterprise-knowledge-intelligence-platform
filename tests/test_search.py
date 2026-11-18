"""Search API: hybrid retrieval, access control, filters and reranking.

Uses the hashing embedder (word overlap, no meaning), so these tests check plumbing and
security, not semantic quality. Retrieval quality is measured by the Phase 6 evals.
"""

import uuid
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from tests.helpers import bearer, create_group, create_user, ingest, run_db

HR_DOC = b"""# HR Handbook

## Parental leave

Parents receive 16 weeks of paid parental leave.

## Expenses

Submit expense reports within 30 days of purchase.
"""

ENG_DOC = b"""# Engineering Runbook

## Deploys

Deploy with the release pipeline. Error code E-4021 means the migration lock is held.
"""

BOARD_DOC = b"""# Board minutes

## Acquisition

The board approved the acquisition of Contoso for parental leave policy alignment.
"""


@dataclass
class Corpus:
    hr_doc: str
    eng_doc: str
    board_doc: str
    hr_user: uuid.UUID
    eng_user: uuid.UUID
    no_group_user: uuid.UUID
    admin_user: uuid.UUID


@pytest.fixture
def corpus(client: TestClient) -> Corpus:
    hr, eng = create_group("hr"), create_group("eng")
    return Corpus(
        hr_doc=ingest(client, "hr.md", HR_DOC, group_ids=[hr]),
        eng_doc=ingest(client, "eng.md", ENG_DOC, group_ids=[eng]),
        board_doc=ingest(client, "board.md", BOARD_DOC),  # no groups: admins only
        hr_user=create_user("hr@example.com", [hr]),
        eng_user=create_user("eng@example.com", [eng]),
        no_group_user=create_user("new@example.com"),
        admin_user=create_user("second-admin@example.com", is_admin=True),
    )


def search(client: TestClient, query: str, user: uuid.UUID | None = None, **options):
    headers = bearer(user) if user else {}
    response = client.post("/search", json={"query": query, **options}, headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


def doc_ids(body: dict) -> set[str]:
    return {r["document_id"] for r in body["results"]}


def test_hybrid_search_finds_section_with_both_retrievers(
    client: TestClient, corpus: Corpus
) -> None:
    body = search(client, "parental leave weeks")
    top = body["results"][0]
    assert top["section_path"] == "HR Handbook > Parental leave"
    assert top["dense_rank"] is not None and top["keyword_rank"] is not None
    assert body["mode"] == "hybrid"
    assert body["reranked"] is False  # reranker disabled in tests
    assert {"embed", "dense", "keyword", "fetch", "total"} <= body["timings_ms"].keys()


def test_keyword_search_matches_exact_codes(client: TestClient, corpus: Corpus) -> None:
    body = search(client, "E-4021", mode="keyword")
    assert body["results"][0]["document_id"] == corpus.eng_doc
    assert body["results"][0]["dense_rank"] is None


def test_dense_only_mode(client: TestClient, corpus: Corpus) -> None:
    body = search(client, "expense reports", mode="dense")
    assert body["results"]
    assert all(r["keyword_rank"] is None for r in body["results"])
    assert "keyword" not in body["timings_ms"]


class TestAccessControl:
    def test_admin_sees_everything(self, client: TestClient, corpus: Corpus) -> None:
        body = search(client, "parental leave deploy", top_k=50)
        assert doc_ids(body) == {corpus.hr_doc, corpus.eng_doc, corpus.board_doc}

    def test_group_member_sees_only_shared_documents(
        self, client: TestClient, corpus: Corpus
    ) -> None:
        # The board minutes match this query best, but the HR user may not read them.
        body = search(client, "acquisition of Contoso parental leave", corpus.hr_user, top_k=50)
        assert doc_ids(body) == {corpus.hr_doc}

    def test_restriction_applies_to_every_mode(self, client: TestClient, corpus: Corpus) -> None:
        for mode in ("hybrid", "dense", "keyword"):
            body = search(client, "deploy E-4021 pipeline", corpus.hr_user, mode=mode, top_k=50)
            assert corpus.eng_doc not in doc_ids(body), mode

    def test_user_without_groups_sees_nothing(self, client: TestClient, corpus: Corpus) -> None:
        assert search(client, "parental leave", corpus.no_group_user)["results"] == []

    def test_admin_user_sees_unshared_documents(self, client: TestClient, corpus: Corpus) -> None:
        body = search(client, "Contoso acquisition", corpus.admin_user, mode="keyword")
        assert body["results"][0]["document_id"] == corpus.board_doc

    def test_document_filter_cannot_widen_access(self, client: TestClient, corpus: Corpus) -> None:
        body = search(client, "Contoso", corpus.hr_user, document_ids=[corpus.board_doc])
        assert body["results"] == []

    def test_token_for_unknown_user_is_rejected(self, client: TestClient, corpus: Corpus) -> None:
        response = client.post("/search", json={"query": "leave"}, headers=bearer(uuid.uuid4()))
        assert response.status_code == 401

    def test_missing_token_is_rejected(self, client: TestClient, corpus: Corpus) -> None:
        response = client.post("/search", json={"query": "leave"}, headers={"Authorization": ""})
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"

    def test_group_change_applies_without_new_token(
        self, client: TestClient, corpus: Corpus
    ) -> None:
        from sqlalchemy import text

        assert search(client, "parental leave", corpus.no_group_user)["results"] == []

        async def add_to_hr(conn):
            await conn.execute(
                text(
                    "INSERT INTO user_groups (user_id, group_id) "
                    "SELECT :u, id FROM groups WHERE name = 'hr'"
                ),
                {"u": corpus.no_group_user},
            )
            await conn.commit()

        run_db(add_to_hr)
        # Same token as before: permissions are read from the database per request.
        assert doc_ids(search(client, "parental leave", corpus.no_group_user)) == {corpus.hr_doc}


def test_document_filter(client: TestClient, corpus: Corpus) -> None:
    body = search(client, "leave deploy", document_ids=[corpus.eng_doc], top_k=50)
    assert doc_ids(body) == {corpus.eng_doc}


def test_top_k_limits_results(client: TestClient, corpus: Corpus) -> None:
    assert len(search(client, "the", top_k=1, mode="dense")["results"]) == 1


def test_no_matches_returns_empty_list(client: TestClient, corpus: Corpus) -> None:
    assert search(client, "zebra quantum", mode="keyword")["results"] == []


@pytest.mark.parametrize(
    "payload",
    [{"query": ""}, {"query": "   "}, {"query": "x", "top_k": 0}, {"query": "x", "mode": "bm25"}],
)
def test_invalid_requests(client: TestClient, payload: dict) -> None:
    assert client.post("/search", json=payload).status_code == 422


class PrefersExpenses:
    """Stand-in cross-encoder: rates any passage about expenses as most relevant."""

    model_name = "fake"

    def score(self, query: str, passages: list[str]) -> list[float]:
        return [0.9 if "expense" in p.lower() else 0.1 for p in passages]


def test_reranker_reorders_candidates(client: TestClient, corpus: Corpus) -> None:
    retriever = client.app.state.retriever
    before = search(client, "parental leave weeks")
    assert before["results"][0]["section_path"] == "HR Handbook > Parental leave"

    retriever.reranker = PrefersExpenses()
    try:
        after = search(client, "parental leave weeks")
        skipped = search(client, "parental leave weeks", rerank=False)
    finally:
        retriever.reranker = None

    assert after["reranked"] is True
    assert after["results"][0]["section_path"] == "HR Handbook > Expenses"
    assert after["results"][0]["score"] == after["results"][0]["rerank_score"] == 0.9
    assert "rerank" in after["timings_ms"]
    assert skipped["reranked"] is False
    assert skipped["results"][0]["section_path"] == "HR Handbook > Parental leave"
