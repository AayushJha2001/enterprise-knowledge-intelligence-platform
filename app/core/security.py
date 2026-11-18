"""Who is making the request, and which groups they belong to.

TEMPORARY (until Phase 5 adds JWT login): in non-production environments a caller can act
as a specific user with the `X-User-Id` header; without it, the request runs as a local
admin. In production this dependency refuses every request, so the dev shortcut can never
ship by accident.
"""

import uuid
from dataclasses import dataclass, field
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import get_settings
from app.db.models import User
from app.db.session import get_session


@dataclass(frozen=True)
class Principal:
    user_id: uuid.UUID | None
    is_admin: bool
    group_ids: frozenset[uuid.UUID] = field(default_factory=frozenset)


LOCAL_ADMIN = Principal(user_id=None, is_admin=True)


async def get_principal(
    session: Annotated[AsyncSession, Depends(get_session)],
    x_user_id: Annotated[uuid.UUID | None, Header()] = None,
) -> Principal:
    if get_settings().environment == "production":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Authentication is not configured")
    if x_user_id is None:
        return LOCAL_ADMIN

    user = await session.scalar(
        select(User).where(User.id == x_user_id).options(selectinload(User.groups))
    )
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Unknown user")
    return Principal(
        user_id=user.id,
        is_admin=user.is_admin,
        group_ids=frozenset(g.id for g in user.groups),
    )


PrincipalDep = Annotated[Principal, Depends(get_principal)]
