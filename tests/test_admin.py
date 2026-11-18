"""Admin endpoints and document visibility for non-admins."""

import uuid

import pytest
from fastapi.testclient import TestClient

from tests.helpers import bearer, create_group, create_user, ingest

PASSWORD = "correct horse battery staple"


def login(client: TestClient, email: str, password: str = PASSWORD):
    return client.post(
        "/auth/token", data={"username": email, "password": password}, headers={"Authorization": ""}
    )


class TestPermissions:
    def test_non_admin_gets_403(self, client: TestClient) -> None:
        member = create_user("member@example.com")
        for method, path in [
            ("get", "/admin/users"),
            ("post", "/admin/groups"),
            ("get", "/admin/groups"),
            ("delete", f"/admin/users/{uuid.uuid4()}"),
        ]:
            response = client.request(method, path, json={}, headers=bearer(member))
            assert response.status_code == 403, path

    def test_anonymous_gets_401(self, client: TestClient) -> None:
        response = client.get("/admin/users", headers={"Authorization": ""})
        assert response.status_code == 401


class TestGroups:
    def test_create_list_delete(self, client: TestClient) -> None:
        created = client.post("/admin/groups", json={"name": "finance", "description": "Money"})
        assert created.status_code == 201
        group_id = created.json()["id"]
        create_user("fin@example.com", [uuid.UUID(group_id)])

        listed = {g["name"]: g for g in client.get("/admin/groups").json()}
        assert listed["finance"]["member_count"] == 1
        assert listed["finance"]["document_count"] == 0

        assert client.delete(f"/admin/groups/{group_id}").status_code == 204
        assert "finance" not in {g["name"] for g in client.get("/admin/groups").json()}

    def test_duplicate_name(self, client: TestClient) -> None:
        client.post("/admin/groups", json={"name": "finance"})
        assert client.post("/admin/groups", json={"name": "finance"}).status_code == 409

    @pytest.mark.parametrize("name", ["", "Has Spaces", "UPPER", "-leading"])
    def test_invalid_name(self, client: TestClient, name: str) -> None:
        assert client.post("/admin/groups", json={"name": name}).status_code == 422


class TestUsers:
    def test_create_user_who_can_then_log_in(self, client: TestClient) -> None:
        hr = create_group("hr")
        response = client.post(
            "/admin/users",
            json={
                "email": "New.Person@Example.com",
                "display_name": "New Person",
                "password": PASSWORD,
                "group_ids": [str(hr)],
            },
        )
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["email"] == "new.person@example.com"
        assert [g["name"] for g in body["groups"]] == ["hr"]
        assert "password" not in str(body).lower()
        assert login(client, "new.person@example.com").status_code == 200

    @pytest.mark.parametrize(
        "change, status",
        [
            ({"email": "not-an-email"}, 422),
            ({"password": "short"}, 422),
            ({"group_ids": [str(uuid.uuid4())]}, 422),
            ({"email": "admin@example.com"}, 409),  # the test admin's email
        ],
    )
    def test_create_user_validation(self, client: TestClient, change: dict, status: int) -> None:
        payload = {"email": "x@example.com", "display_name": "X", "password": PASSWORD}
        assert client.post("/admin/users", json={**payload, **change}).status_code == status

    def test_update_groups(self, client: TestClient) -> None:
        hr, eng = create_group("hr"), create_group("eng")
        user = create_user("u@example.com", [hr])
        response = client.patch(f"/admin/users/{user}", json={"group_ids": [str(eng)]})
        assert [g["name"] for g in response.json()["groups"]] == ["eng"]

    def test_deactivation_revokes_tokens(self, client: TestClient) -> None:
        user = create_user("u@example.com")
        assert client.get("/auth/me", headers=bearer(user)).status_code == 200
        client.patch(f"/admin/users/{user}", json={"is_active": False})
        assert client.get("/auth/me", headers=bearer(user, token_version=1)).status_code == 401
        assert client.get("/auth/me", headers=bearer(user)).status_code == 401

    def test_password_reset_revokes_tokens(self, client: TestClient) -> None:
        user = create_user("u@example.com", password=PASSWORD)
        client.patch(f"/admin/users/{user}", json={"password": "a brand new password"})
        assert client.get("/auth/me", headers=bearer(user)).status_code == 401
        assert login(client, "u@example.com", "a brand new password").status_code == 200

    def test_admin_cannot_lock_themselves_out(self, client: TestClient, admin_id) -> None:
        assert client.patch(f"/admin/users/{admin_id}", json={"is_admin": False}).status_code == 409
        assert (
            client.patch(f"/admin/users/{admin_id}", json={"is_active": False}).status_code == 409
        )
        assert client.delete(f"/admin/users/{admin_id}").status_code == 409

    def test_delete_user(self, client: TestClient) -> None:
        user = create_user("gone@example.com")
        assert client.delete(f"/admin/users/{user}").status_code == 204
        assert client.get(f"/admin/users/{user}").status_code == 404
        assert client.get("/auth/me", headers=bearer(user)).status_code == 401


HR_DOC = b"# HR Handbook\n\n## Leave\n\nParents receive 16 weeks of paid parental leave."
BOARD_DOC = b"# Board minutes\n\n## Deals\n\nThe board approved the acquisition of Contoso."


class TestDocumentVisibility:
    @pytest.fixture
    def setup(self, client: TestClient) -> dict:
        hr = create_group("hr")
        return {
            "hr": hr,
            "hr_doc": ingest(client, "hr.md", HR_DOC, group_ids=[hr]),
            "board_doc": ingest(client, "board.md", BOARD_DOC),
            "member": create_user("member@example.com", [hr]),
        }

    def test_member_lists_only_shared_documents(self, client: TestClient, setup) -> None:
        listing = client.get("/documents", headers=bearer(setup["member"])).json()
        assert [d["id"] for d in listing["items"]] == [setup["hr_doc"]]
        assert listing["total"] == 1
        assert client.get("/documents").json()["total"] == 2  # admin sees both

    def test_unshared_document_is_not_found(self, client: TestClient, setup) -> None:
        headers = bearer(setup["member"])
        board = setup["board_doc"]
        assert client.get(f"/documents/{board}", headers=headers).status_code == 404
        assert client.get(f"/documents/{board}/chunks", headers=headers).status_code == 404
        assert client.get(f"/documents/{setup['hr_doc']}", headers=headers).status_code == 200

    def test_only_admins_manage_documents(self, client: TestClient, setup) -> None:
        headers = bearer(setup["member"])
        hr_doc = setup["hr_doc"]
        upload = client.post(
            "/documents", files={"file": ("x.md", b"# X\n\ntext")}, headers=headers
        )
        assert upload.status_code == 403
        assert client.delete(f"/documents/{hr_doc}", headers=headers).status_code == 403
        assert client.post(f"/documents/{hr_doc}/reindex", headers=headers).status_code == 403

    def test_sharing_a_document(self, client: TestClient, setup) -> None:
        headers = bearer(setup["member"])
        board = setup["board_doc"]
        response = client.put(
            f"/admin/documents/{board}/groups", json={"group_ids": [str(setup["hr"])]}
        )
        assert [g["name"] for g in response.json()] == ["hr"]
        assert client.get(f"/documents/{board}", headers=headers).status_code == 200
        search = client.post("/search", json={"query": "Contoso acquisition"}, headers=headers)
        assert board in {r["document_id"] for r in search.json()["results"]}

        client.put(f"/admin/documents/{board}/groups", json={"group_ids": []})
        assert client.get(f"/documents/{board}", headers=headers).status_code == 404

    def test_sharing_unknown_document(self, client: TestClient) -> None:
        response = client.put(f"/admin/documents/{uuid.uuid4()}/groups", json={"group_ids": []})
        assert response.status_code == 404
