"""End-to-end ingestion: upload through the API, then run the worker's job."""

import asyncio

from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.pool import NullPool

from app.core.config import get_settings
from app.db.session import build_session_factory
from app.ingestion.embeddings import HashingEmbedder
from app.ingestion.pipeline import requeue_stale
from tests.helpers import SAMPLE_MARKDOWN, make_docx, make_pdf, run_db, run_worker_once


def upload(client: TestClient, name: str, data: bytes) -> dict:
    response = client.post("/documents", files={"file": (name, data)})
    assert response.status_code == 202, response.text
    return response.json()


def test_empty_queue_returns_none() -> None:
    assert run_worker_once() is None


def test_markdown_document_is_chunked_and_embedded(client: TestClient) -> None:
    doc = upload(client, "handbook.md", SAMPLE_MARKDOWN)
    assert run_worker_once() == "ready"

    body = client.get(f"/documents/{doc['id']}").json()
    assert body["status"] == "ready"
    assert body["title"] == "Employee Handbook"  # taken from the file's first heading
    assert body["chunk_count"] >= 3
    assert body["embedding_model"] == "hashing-bow"
    assert body["processed_at"] is not None
    assert body["attempts"] == 1

    chunks = client.get(f"/documents/{doc['id']}/chunks").json()
    assert chunks["total"] == body["chunk_count"]
    sections = {c["section_path"] for c in chunks["items"]}
    assert "Employee Handbook > Leave > Annual leave" in sections

    async def check_db(conn):
        row = await conn.execute(
            text(
                "SELECT count(*) FILTER (WHERE embedding IS NULL), "
                "       count(*) FILTER (WHERE tsv @@ plainto_tsquery('english', 'parental')) "
                "FROM chunks WHERE document_id = :id"
            ),
            {"id": doc["id"]},
        )
        return row.one()

    missing_embeddings, keyword_hits = run_db(check_db)
    assert missing_embeddings == 0
    assert keyword_hits == 1  # full-text search column is populated by Postgres


def test_pdf_chunks_keep_page_numbers(client: TestClient) -> None:
    doc = upload(client, "policy.pdf", make_pdf(["Intro on page one.", "Details on page two."]))
    assert run_worker_once() == "ready"
    chunks = client.get(f"/documents/{doc['id']}/chunks").json()["items"]
    assert chunks[0]["page_number"] == 1
    assert chunks[0]["metadata"] == {"page_end": 2}
    assert client.get(f"/documents/{doc['id']}").json()["page_count"] == 2


def test_docx_is_ingested(client: TestClient) -> None:
    doc = upload(client, "travel.docx", make_docx())
    assert run_worker_once() == "ready"
    sections = {
        c["section_path"] for c in client.get(f"/documents/{doc['id']}/chunks").json()["items"]
    }
    assert "Travel Policy > Hotels" in sections


def test_unparseable_file_fails_without_retry(client: TestClient) -> None:
    doc = upload(client, "scan.pdf", make_pdf([""]))  # valid PDF, but no text in it
    assert run_worker_once() == "failed"
    body = client.get(f"/documents/{doc['id']}").json()
    assert body["status"] == "failed"
    assert "No text found" in body["error"]
    assert body["attempts"] == 1
    assert run_worker_once() is None  # not requeued


class FlakyEmbedder(HashingEmbedder):
    def __init__(self, dim: int, failures: int) -> None:
        super().__init__(dim)
        self.failures = failures

    def embed_passages(self, texts):
        if self.failures:
            self.failures -= 1
            raise RuntimeError("embedding backend unavailable")
        return super().embed_passages(texts)


def test_transient_error_is_retried_then_succeeds(client: TestClient) -> None:
    doc = upload(client, "a.md", b"# A\n\nalpha text")
    embedder = FlakyEmbedder(get_settings().embedding_dim, failures=1)
    assert run_worker_once(embedder) == "pending"
    assert "embedding backend unavailable" in client.get(f"/documents/{doc['id']}").json()["error"]
    assert run_worker_once(embedder) == "ready"
    body = client.get(f"/documents/{doc['id']}").json()
    assert body["attempts"] == 2
    assert body["error"] is None


def test_transient_error_gives_up_after_max_attempts(client: TestClient) -> None:
    doc = upload(client, "a.md", b"# A\n\nalpha text")
    embedder = FlakyEmbedder(get_settings().embedding_dim, failures=99)
    results = [run_worker_once(embedder) for _ in range(get_settings().worker_max_attempts)]
    assert results[-1] == "failed"
    assert results[:-1] == ["pending"] * (len(results) - 1)
    assert client.get(f"/documents/{doc['id']}").json()["status"] == "failed"


def test_reindex_replaces_chunks(client: TestClient) -> None:
    doc = upload(client, "handbook.md", SAMPLE_MARKDOWN)
    run_worker_once()
    first_ids = {c["id"] for c in client.get(f"/documents/{doc['id']}/chunks").json()["items"]}

    assert client.post(f"/documents/{doc['id']}/reindex").status_code == 202
    assert run_worker_once() == "ready"
    second = client.get(f"/documents/{doc['id']}/chunks").json()
    assert second["total"] == len(first_ids)
    assert first_ids.isdisjoint({c["id"] for c in second["items"]})


def test_stale_processing_document_is_requeued(client: TestClient) -> None:
    doc = upload(client, "a.md", b"# A\n\nalpha text")

    async def simulate_crashed_worker(conn):
        await conn.execute(
            text(
                "UPDATE documents SET status = 'processing', "
                "updated_at = now() - interval '1 hour' WHERE id = :id"
            ),
            {"id": doc["id"]},
        )
        await conn.commit()

    run_db(simulate_crashed_worker)
    assert run_worker_once() is None  # nothing pending yet

    async def requeue():
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(get_settings().database_url, poolclass=NullPool)
        try:
            return await requeue_stale(build_session_factory(engine), get_settings())
        finally:
            await engine.dispose()

    assert asyncio.run(requeue()) == 1
    assert run_worker_once() == "ready"
