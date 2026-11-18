"""Login, tokens, revocation and password changes."""

import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi.testclient import TestClient

from app.core.config import DEV_JWT_SECRET, Settings, get_settings
from app.core.security import hash_password, verify_password
from tests.helpers import bearer, create_group, create_user

PASSWORD = "correct horse battery staple"
NO_AUTH = {"Authorization": ""}


def login(client: TestClient, email: str, password: str):
    return client.post(
        "/auth/token", data={"username": email, "password": password}, headers=NO_AUTH
    )


def token_header(response) -> dict[str, str]:
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture
def user(client: TestClient) -> uuid.UUID:
    return create_user("Hana@Example.com", [create_group("hr")], password=PASSWORD)


class TestLogin:
    def test_success_returns_working_token(self, client: TestClient, user) -> None:
        response = login(client, "hana@example.com", PASSWORD)
        body = response.json()
        assert body["token_type"] == "bearer"
        assert body["expires_in"] == get_settings().access_token_ttl_minutes * 60

        me = client.get("/auth/me", headers=token_header(response)).json()
        assert me["id"] == str(user)
        assert me["is_admin"] is False
        assert [g["name"] for g in me["groups"]] == ["hr"]

    def test_email_is_case_insensitive(self, client: TestClient, user) -> None:
        assert login(client, "  HANA@example.COM ", PASSWORD).status_code == 200

    def test_wrong_password_and_unknown_email_look_the_same(self, client: TestClient, user) -> None:
        wrong = login(client, "hana@example.com", "wrong password!!")
        unknown = login(client, "nobody@example.com", PASSWORD)
        assert wrong.status_code == unknown.status_code == 401
        assert wrong.json() == unknown.json()

    def test_lockout_after_repeated_failures(self, client: TestClient, user) -> None:
        for _ in range(get_settings().login_max_failures):
            assert login(client, "hana@example.com", "wrong password!!").status_code == 401
        locked = login(client, "hana@example.com", PASSWORD)  # even the right password
        assert locked.status_code == 429
        assert locked.headers["retry-after"] == str(get_settings().login_lockout_s)

    def test_success_resets_failure_count(self, client: TestClient, user) -> None:
        for _ in range(get_settings().login_max_failures - 1):
            login(client, "hana@example.com", "wrong password!!")
        assert login(client, "hana@example.com", PASSWORD).status_code == 200
        assert login(client, "hana@example.com", "wrong password!!").status_code == 401
        assert login(client, "hana@example.com", PASSWORD).status_code == 200

    def test_user_without_password_cannot_log_in(self, client: TestClient) -> None:
        create_user("sso-only@example.com")
        assert login(client, "sso-only@example.com", "").status_code in (401, 422)
        assert login(client, "sso-only@example.com", PASSWORD).status_code == 401

    def test_deactivated_user_cannot_log_in(self, client: TestClient, user) -> None:
        client.patch(f"/admin/users/{user}", json={"is_active": False})
        assert login(client, "hana@example.com", PASSWORD).status_code == 401


class TestTokens:
    def test_expired_token(self, client: TestClient, user, monkeypatch) -> None:
        monkeypatch.setattr(get_settings(), "access_token_ttl_minutes", -1)
        response = client.get("/auth/me", headers=bearer(user))
        assert response.status_code == 401
        assert response.json()["detail"] == "Token expired"

    def test_tampered_signature(self, client: TestClient, user) -> None:
        token = bearer(user)["Authorization"].removeprefix("Bearer ")
        forged = jwt.encode(
            jwt.decode(token, options={"verify_signature": False}),
            "attacker-controlled-key-of-sufficient-length",
            "HS256",
        )
        response = client.get("/auth/me", headers={"Authorization": f"Bearer {forged}"})
        assert response.status_code == 401

    def test_unsigned_token_is_rejected(self, client: TestClient, user) -> None:
        now = datetime.now(UTC)
        claims = {
            "sub": str(user),
            "ver": 0,
            "type": "access",
            "iat": now,
            "exp": now + timedelta(hours=1),
        }
        unsigned = jwt.encode(claims, key=None, algorithm="none")
        response = client.get("/auth/me", headers={"Authorization": f"Bearer {unsigned}"})
        assert response.status_code == 401

    def test_garbage_token(self, client: TestClient) -> None:
        response = client.get("/auth/me", headers={"Authorization": "Bearer not-a-jwt"})
        assert response.status_code == 401

    def test_logout_everywhere_revokes_existing_tokens(self, client: TestClient, user) -> None:
        old = token_header(login(client, "hana@example.com", PASSWORD))
        assert client.post("/auth/logout-everywhere", headers=old).status_code == 204
        assert client.get("/auth/me", headers=old).status_code == 401
        new = token_header(login(client, "hana@example.com", PASSWORD))
        assert client.get("/auth/me", headers=new).status_code == 200


class TestPasswordChange:
    def test_change_password(self, client: TestClient, user) -> None:
        old = token_header(login(client, "hana@example.com", PASSWORD))
        response = client.post(
            "/auth/password",
            json={"current_password": PASSWORD, "new_password": "a much longer new secret"},
            headers=old,
        )
        assert response.status_code == 204
        assert client.get("/auth/me", headers=old).status_code == 401  # signed out everywhere
        assert login(client, "hana@example.com", PASSWORD).status_code == 401
        assert login(client, "hana@example.com", "a much longer new secret").status_code == 200

    def test_wrong_current_password(self, client: TestClient, user) -> None:
        response = client.post(
            "/auth/password",
            json={"current_password": "nope nope nope", "new_password": "a much longer new one"},
            headers=bearer(user),
        )
        assert response.status_code == 400

    def test_new_password_too_short(self, client: TestClient, user) -> None:
        response = client.post(
            "/auth/password",
            json={"current_password": PASSWORD, "new_password": "short"},
            headers=bearer(user),
        )
        assert response.status_code == 422


def test_password_hashing() -> None:
    hashed = hash_password(PASSWORD)
    assert hashed.startswith("$argon2id$")
    assert verify_password(hashed, PASSWORD)
    assert not verify_password(hashed, PASSWORD + "!")
    assert not verify_password(None, PASSWORD)
    assert not verify_password("not-a-hash", PASSWORD)


def test_production_refuses_default_jwt_secret() -> None:
    with pytest.raises(RuntimeError, match="JWT_SECRET"):
        Settings(environment="production", jwt_secret=DEV_JWT_SECRET).check_production_safety()
    with pytest.raises(RuntimeError):
        Settings(environment="production", jwt_secret="too-short").check_production_safety()
    Settings(environment="production", jwt_secret="x" * 40).check_production_safety()
    Settings(environment="local").check_production_safety()  # dev default is fine locally
