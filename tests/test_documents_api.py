import hashlib
import uuid

from fastapi.testclient import TestClient
from sqlalchemy import text

from tests.helpers import SAMPLE_MARKDOWN, make_pdf, run_db


def upload(client: TestClient, name: str, data: bytes, **form):
    return client.post("/documents", files={"file": (name, data)}, data=form)


def create_group(name: str) -> uuid.UUID:
    async def insert(conn):
        group_id = uuid.uuid4()
        await conn.execute(
            text("INSERT INTO groups (id, name) VALUES (:id, :name)"),
            {"id": group_id, "name": name},
        )
        await conn.commit()
        return group_id

    return run_db(insert)


def test_upload_queues_document(client: TestClient) -> None:
    response = upload(client, "handbook.md", SAMPLE_MARKDOWN)
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["status"] == "pending"
    assert body["title"] == "handbook"  # replaced by the parsed title during ingestion
    assert body["mime_type"] == "text/markdown"
    assert body["size_bytes"] == len(SAMPLE_MARKDOWN)
    assert body["chunk_count"] == 0

    fetched = client.get(f"/documents/{body['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["id"] == body["id"]


def test_upload_with_title_and_groups(client: TestClient) -> None:
    hr = create_group("hr")
    response = upload(
        client, "a.pdf", make_pdf(["hello"]), title="HR Policies", group_ids=[str(hr)]
    )
    assert response.status_code == 202, response.text
    assert response.json()["title"] == "HR Policies"
    assert response.json()["group_ids"] == [str(hr)]


def test_unknown_group_is_rejected(client: TestClient) -> None:
    missing = str(uuid.uuid4())
    response = upload(client, "a.md", b"# hi\n\ntext", group_ids=[missing])
    assert response.status_code == 422
    assert response.json()["detail"]["group_ids"] == [missing]


def test_duplicate_upload_returns_existing_id(client: TestClient) -> None:
    first = upload(client, "one.md", SAMPLE_MARKDOWN).json()
    second = upload(client, "copy-with-other-name.md", SAMPLE_MARKDOWN)
    assert second.status_code == 409
    assert second.json()["detail"]["document_id"] == first["id"]


def test_unsupported_type_rejected(client: TestClient) -> None:
    assert upload(client, "setup.exe", b"MZ\x90\x00").status_code == 415
    assert upload(client, "renamed.pdf", b"MZ\x90\x00").status_code == 415


def test_empty_file_rejected(client: TestClient) -> None:
    assert upload(client, "empty.txt", b"").status_code == 422


def test_too_large_rejected(client: TestClient, monkeypatch) -> None:
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "max_upload_bytes", 10)
    assert upload(client, "big.txt", b"x" * 11).status_code == 413


def test_list_and_filter_by_status(client: TestClient) -> None:
    upload(client, "a.md", b"# A\n\nalpha")
    upload(client, "b.md", b"# B\n\nbeta")
    listing = client.get("/documents").json()
    assert listing["total"] == 2
    assert {d["original_filename"] for d in listing["items"]} == {"a.md", "b.md"}
    assert client.get("/documents?status=ready").json()["total"] == 0
    assert client.get("/documents?limit=1").json()["items"].__len__() == 1


def test_get_unknown_document_404(client: TestClient) -> None:
    assert client.get(f"/documents/{uuid.uuid4()}").status_code == 404


def test_reindex_pending_document_conflicts(client: TestClient) -> None:
    doc = upload(client, "a.md", b"# A\n\nalpha").json()
    assert client.post(f"/documents/{doc['id']}/reindex").status_code == 409


def test_delete_removes_document_and_file(client: TestClient) -> None:
    data = b"# Delete me\n\nunique content for the delete test"
    doc = upload(client, "a.md", data).json()
    checksum = hashlib.sha256(data).hexdigest()
    stored = client.app.state.storage.root / checksum[:2] / f"{checksum}.md"
    assert stored.exists()

    assert client.delete(f"/documents/{doc['id']}").status_code == 204
    assert client.get(f"/documents/{doc['id']}").status_code == 404
    assert not stored.exists()
