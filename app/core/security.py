"""Authentication: password hashing, access tokens, and who is making a request.

Design:
- Passwords are hashed with argon2id (memory-hard, the current OWASP recommendation).
- Access tokens are short-lived HS256 JWTs carrying only the user id and a token version.
- Admin rights and group memberships are read from the database on every request, never
  trusted from the token, so revoking access takes effect immediately, not at expiry.
- Bumping a user's token_version (password change, deactivation, "log out everywhere")
  invalidates every token issued before it.
"""

import contextlib
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Annotated, Any

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import Settings, get_settings
from app.db.models import User
from app.db.session import get_session

ALGORITHM = "HS256"
_hasher = PasswordHasher()


# ---------------------------------------------------------------- passwords


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password_hash: str | None, password: str) -> bool:
    if not password_hash:
        _dummy_verify(password)
        return False
    try:
        return _hasher.verify(password_hash, password)
    except (VerificationError, InvalidHashError):
        return False


def password_needs_rehash(password_hash: str) -> bool:
    """True when the hash uses outdated parameters (upgrade it on next successful login)."""
    return _hasher.check_needs_rehash(password_hash)


@lru_cache
def _dummy_hash() -> str:
    return _hasher.hash("timing-equalizer-not-a-real-password")


def _dummy_verify(password: str) -> None:
    """Spend the same time as a real check, so response timing doesn't reveal which
    email addresses have accounts."""
    with contextlib.suppress(VerificationError):
        _hasher.verify(_dummy_hash(), password)


# ---------------------------------------------------------------- tokens


def create_access_token(user_id: uuid.UUID, token_version: int, settings: Settings) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "ver": token_version,
        "type": "access",
        "iat": now,
        "exp": now + timedelta(minutes=settings.access_token_ttl_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret.get_secret_value(), algorithm=ALGORITHM)


def decode_access_token(token: str, settings: Settings) -> dict[str, Any]:
    payload = jwt.decode(
        token,
        settings.jwt_secret.get_secret_value(),
        algorithms=[ALGORITHM],  # never accept "none" or an algorithm the client picks
        options={"require": ["exp", "iat", "sub", "ver"]},
    )
    if payload.get("type") != "access":
        raise jwt.InvalidTokenError("not an access token")
    return payload


# ---------------------------------------------------------------- request principal


@dataclass(frozen=True)
class Principal:
    user_id: uuid.UUID
    is_admin: bool
    group_ids: frozenset[uuid.UUID] = field(default_factory=frozenset)


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status.HTTP_401_UNAUTHORIZED, detail, headers={"WWW-Authenticate": "Bearer"}
    )


async def load_user(session: AsyncSession, user_id: uuid.UUID) -> User | None:
    return await session.scalar(
        select(User).where(User.id == user_id).options(selectinload(User.groups))
    )


async def get_principal(
    session: Annotated[AsyncSession, Depends(get_session)],
    token: Annotated[str | None, Depends(oauth2_scheme)],
) -> Principal:
    if not token:
        raise _unauthorized("Not authenticated")
    try:
        payload = decode_access_token(token, get_settings())
        user_id = uuid.UUID(payload["sub"])
    except jwt.ExpiredSignatureError as exc:
        raise _unauthorized("Token expired") from exc
    except (jwt.InvalidTokenError, ValueError) as exc:
        raise _unauthorized("Invalid token") from exc

    user = await load_user(session, user_id)
    if user is None or not user.is_active or user.token_version != payload["ver"]:
        raise _unauthorized("Token is no longer valid")
    return Principal(
        user_id=user.id,
        is_admin=user.is_admin,
        group_ids=frozenset(g.id for g in user.groups),
    )


async def require_admin(principal: Annotated[Principal, Depends(get_principal)]) -> Principal:
    if not principal.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin rights required")
    return principal


PrincipalDep = Annotated[Principal, Depends(get_principal)]
AdminDep = Annotated[Principal, Depends(require_admin)]
