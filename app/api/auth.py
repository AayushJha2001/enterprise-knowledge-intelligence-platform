"""Login, current user, password change and token revocation."""

import contextlib
import logging
import uuid
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from redis.exceptions import RedisError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.security import (
    PrincipalDep,
    create_access_token,
    hash_password,
    load_user,
    password_needs_rehash,
    verify_password,
)
from app.db.models import User
from app.db.session import get_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
MIN_PASSWORD_LENGTH = 12


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class GroupRef(BaseModel):
    id: uuid.UUID
    name: str


class MeResponse(BaseModel):
    id: uuid.UUID
    email: str
    display_name: str
    is_admin: bool
    groups: list[GroupRef]


class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(min_length=MIN_PASSWORD_LENGTH, max_length=256)


def _failure_key(email: str, request: Request) -> str:
    client = request.client.host if request.client else "unknown"
    return f"ekip:login:failures:{email}:{client}"


@router.post("/token", response_model=TokenResponse)
async def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    request: Request,
    session: SessionDep,
) -> TokenResponse:
    """Exchange email + password for an access token (OAuth2 password flow).

    Use the email as `username`. Repeated failures lock the email out for a while,
    per client address, to slow down password guessing.
    """
    settings = get_settings()
    email = form.username.strip().lower()
    redis = request.app.state.redis
    failure_key = _failure_key(email, request)

    try:
        failures = int(await redis.get(failure_key) or 0)
    except RedisError:
        failures = 0  # fail open: an outage shouldn't lock everyone out
    if failures >= settings.login_max_failures:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Too many failed logins; try again later",
            headers={"Retry-After": str(settings.login_lockout_s)},
        )

    user = await session.scalar(select(User).where(func.lower(User.email) == email))
    # verify_password runs even for unknown emails, so timing reveals nothing.
    valid = verify_password(user.password_hash if user else None, form.password)
    if not valid or user is None or not user.is_active:
        with contextlib.suppress(RedisError):
            count = await redis.incr(failure_key)
            if count == 1:
                await redis.expire(failure_key, settings.login_lockout_s)
        logger.info("failed login")
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    with contextlib.suppress(RedisError):
        await redis.delete(failure_key)
    if password_needs_rehash(user.password_hash):
        user.password_hash = hash_password(form.password)
    user.last_login_at = datetime.now(UTC)
    await session.commit()
    return TokenResponse(
        access_token=create_access_token(user.id, user.token_version, settings),
        expires_in=settings.access_token_ttl_minutes * 60,
    )


@router.get("/me", response_model=MeResponse)
async def me(principal: PrincipalDep, session: SessionDep) -> MeResponse:
    user = await load_user(session, principal.user_id)
    return MeResponse(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        is_admin=user.is_admin,
        groups=[GroupRef(id=g.id, name=g.name) for g in sorted(user.groups, key=lambda g: g.name)],
    )


@router.post("/password", status_code=status.HTTP_204_NO_CONTENT)
async def change_password(body: PasswordChange, principal: PrincipalDep, session: SessionDep):
    """Change your password. Signs you out everywhere (all existing tokens stop working)."""
    user = await load_user(session, principal.user_id)
    if not verify_password(user.password_hash, body.current_password):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Current password is incorrect")
    user.password_hash = hash_password(body.new_password)
    user.token_version += 1
    await session.commit()


@router.post("/logout-everywhere", status_code=status.HTTP_204_NO_CONTENT)
async def logout_everywhere(principal: PrincipalDep, session: SessionDep):
    """Revoke every access token issued to you so far, including this one."""
    user = await load_user(session, principal.user_id)
    user.token_version += 1
    await session.commit()
