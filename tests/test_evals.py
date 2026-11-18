"""Eval harness: metric maths, dataset validity, and an end-to-end run against the test app."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from evals.metrics import (
    citation_precision,
    fact_recall,
    first_relevant_rank,
    hit_at_k,
    ndcg_at_k,
    percentile,
    reciprocal_rank,
)
from evals.run import DEFAULT_DATASET, Api, check_thresholds, load_dataset, run
from tests.helpers import create_group, create_user, ingest

PATHS = ["Handbook > Travel", "Handbook > Leave > Parental leave", "Handbook > Travel"]


class TestMetrics:
    def test_rank_hit_and_mrr(self) -> None:
        rank = first_relevant_rank(PATHS, ["parental LEAVE"])
        assert rank == 2
        assert hit_at_k(rank, 1) == 0.0 and hit_at_k(rank, 3) == 1.0
        assert reciprocal_rank(rank) == 0.5
        assert first_relevant_rank(PATHS, ["Claims"]) is None
        assert reciprocal_rank(None) == 0.0

    def test_relevance_uses_the_last_heading_only(self) -> None:
        # "Leave" is an ancestor here, not the section itself.
        assert first_relevant_rank(PATHS, ["Leave"]) is None

    def test_ndcg(self) -> None:
        assert ndcg_at_k(PATHS, ["Travel"], 10) == 1.0  # found at rank 1
        assert ndcg_at_k(PATHS, ["Parental leave"], 10) == pytest.approx(1 / 1.58496, 1e-4)
        # Two expected sections at ranks 1 and 2: ideal ordering.
        assert ndcg_at_k(PATHS, ["Travel", "Parental leave"], 10) == pytest.approx(1.0)
        assert ndcg_at_k(PATHS, ["Claims"], 10) == 0.0
        assert ndcg_at_k(PATHS, ["Parental leave"], 1) == 0.0  # outside the cut-off

    def test_fact_recall(self) -> None:
        answer = "Parents receive  16 Weeks of leave, requested 8 weeks ahead."
        assert fact_recall(answer, ["16 weeks", "8 weeks"]) == 1.0
        assert fact_recall(answer, ["16 weeks", "three blocks"]) == 0.5
        assert fact_recall(answer, []) == 1.0

    def test_citation_precision(self) -> None:
        assert citation_precision(PATHS, ["Travel"]) == pytest.approx(2 / 3)
        assert citation_precision([], ["Travel"]) is None

    def test_percentile(self) -> None:
        assert percentile([10, 20, 30, 40], 50) == 25
        assert percentile([10, 20, 30, 40], 100) == 40
        assert percentile([], 95) == 0.0


def test_shipped_dataset_is_valid() -> None:
    questions = load_dataset(DEFAULT_DATASET)
    assert len(questions) >= 50
    categories = {q.category for q in questions}
    assert categories <= {"paraphrase", "identifier", "table", "unanswerable", "acl"}
    for q in questions:
        if q.category == "unanswerable":
            assert not q.expected_sections and not q.expected_facts, q.id
        elif q.category != "acl" or q.expected_sections:
            assert q.expected_sections and q.expected_facts, q.id
        if q.category == "acl":
            assert q.forbidden_documents and q.user != "eli", q.id


def test_dataset_rejects_duplicate_ids(tmp_path: Path) -> None:
    line = json.dumps({"id": "a", "category": "paraphrase", "question": "q"})
    (tmp_path / "d.jsonl").write_text(f"{line}\n{line}\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_dataset(tmp_path / "d.jsonl")


class InProcessApi(Api):
    """Routes the harness's HTTP calls into the in-process test app."""

    def __init__(self, client: TestClient) -> None:
        super().__init__("http://testserver")
        self.client = client

    def _send(self, method, path, data, headers):
        response = self.client.request(
            method, path, content=data, headers={"Authorization": "", **headers}
        )
        try:
            return response.status_code, response.json()
        except ValueError:
            return response.status_code, response.text


HR_DOC = b"""# HR Handbook

## Parental leave

Parents receive sixteen weeks of paid parental leave.

## Travel

Hotel stays are capped at 150 EUR per night.
"""

SECRET_DOC = b"# Board minutes\n\n## Deals\n\nThe board approved buying Contoso for parental care."


def test_end_to_end_run(client: TestClient, tmp_path: Path) -> None:
    hr = create_group("hr")
    ingest(client, "hr.md", HR_DOC, group_ids=[hr])
    ingest(client, "board.md", SECRET_DOC)  # admin-only
    create_user("hana@example.com", [hr], password="hana-password-123")

    rows = [
        {
            "id": "leave",
            "category": "paraphrase",
            "user": "hana",
            "question": "weeks of paid parental leave",
            "expected_sections": ["Parental leave"],
            "expected_facts": ["sixteen weeks"],
            "forbidden_documents": ["Board minutes"],
        },
        {
            "id": "hotel",
            "category": "paraphrase",
            "user": "hana",
            "question": "hotel stays capped per night",
            "expected_sections": ["Travel"],
            "expected_facts": ["150 EUR"],
        },
        {
            "id": "board",
            "category": "acl",
            "user": "hana",
            "question": "Did the board approve buying Contoso?",
            "forbidden_documents": ["Board minutes"],
        },
    ]
    dataset = tmp_path / "mini.jsonl"
    dataset.write_text("\n".join(json.dumps(r) for r in rows))

    report = run(
        InProcessApi(client),
        dataset,
        {"retrieval", "answers"},
        {"hana": ("hana@example.com", "hana-password-123")},
    )

    hybrid = report["retrieval"]["hybrid"]
    assert hybrid["questions"] == 2
    assert hybrid["hit@3"] == 1.0
    assert hybrid["acl_leaks"] == 0
    assert set(report["retrieval"]) == {"dense", "keyword", "hybrid", "hybrid+rerank"}

    answers = report["answers"]
    assert answers["fact_recall"] == 1.0
    assert answers["grounded_rate"] == 1.0
    assert answers["acl_leaks"] == 0
    assert answers["model"] == "extractive-v1"
    assert answers["total_cost_usd"] == 0.0

    assert check_thresholds(report, {"answers": {"fact_recall": 0.9, "acl_leaks_max": 0}}) == []
    violations = check_thresholds(
        report,
        {
            "retrieval": {"hybrid": {"hit@3": 1.1}},
            "answers": {"acl_leaks_max": -1, "nonexistent": 0.5},
        },
    )
    assert len(violations) == 3
    assert any("hybrid.hit@3" in v for v in violations)
    assert any("metric missing" in v for v in violations)
