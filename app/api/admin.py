"""Administration: users, groups, and which groups can read which documents."""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.auth import MIN_PASSWORD_LENGTH, GroupRef
from app.core.security import AdminDep, hash_password
from app.db.models import Document, Group, User, document_access, user_groups
from app.db.session import get_session

router = APIRouter(prefix="/admin", tags=["admin"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"


# ---------------------------------------------------------------- schemas


class GroupIn(BaseModel):
    name: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    description: str | None = Field(default=None, max_length=1000)


class GroupOut(BaseModel):
    id: uuid.UUID
    name: str
    description: str | None
    member_count: int
    document_count: int


class UserIn(BaseModel):
    email: str = Field(max_length=320, pattern=EMAIL_PATTERN)
    display_name: str = Field(min_length=1, max_length=200)
    password: str = Field(min_length=MIN_PASSWORD_LENGTH, max_length=256)
    is_admin: bool = False
    group_ids: list[uuid.UUID] = Field(default_factory=list, max_length=100)


class UserUpdate(BaseModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=200)
    password: str | None = Field(default=None, min_length=MIN_PASSWORD_LENGTH, max_length=256)
    is_admin: bool | None = None
    is_active: bool | None = None
    group_ids: list[uuid.UUID] | None = Field(default=None, max_length=100)


class UserOut(BaseModel):
    id: uuid.UUID
    email: str
    display_name: str
    is_admin: bool
    is_active: bool
    groups: list[GroupRef]
    last_login_at: datetime | None
    created_at: datetime

    @classmethod
    def from_model(cls, user: User) -> "UserOut":
        return cls(
            id=user.id,
            email=user.email,
            display_name=user.display_name,
            is_admin=user.is_admin,
            is_active=user.is_active,
            groups=[
                GroupRef(id=g.id, name=g.name) for g in sorted(user.groups, key=lambda g: g.name)
            ],
            last_login_at=user.last_login_at,
            created_at=user.created_at,
        )


class DocumentGroupsIn(BaseModel):
    group_ids: list[uuid.UUID] = Field(max_length=100)


# ---------------------------------------------------------------- helpers


async def _groups_by_id(session: AsyncSession, group_ids: list[uuid.UUID]) -> list[Group]:
    wanted = set(group_ids)
    if not wanted:
        return []
    groups = list(await session.scalars(select(Group).where(Group.id.in_(wanted))))
    missing = wanted - {g.id for g in groups}
    if missing:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            {"message": "Unknown group ids", "group_ids": sorted(map(str, missing))},
        )
    return groups


async def _user(session: AsyncSession, user_id: uuid.UUID) -> User:
    user = await session.scalar(
        select(User).where(User.id == user_id).options(selectinload(User.groups))
    )
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    return user


# ---------------------------------------------------------------- groups


@router.get("/groups", response_model=list[GroupOut])
async def list_groups(admin: AdminDep, session: SessionDep) -> list[GroupOut]:
    members = (
        select(func.count())
        .select_from(user_groups)
        .where(user_groups.c.group_id == Group.id)
        .scalar_subquery()
    )
    documents = (
        select(func.count())
        .select_from(document_access)
        .where(document_access.c.group_id == Group.id)
        .scalar_subquery()
    )
    rows = await session.execute(
        select(Group, members.label("members"), documents.label("documents")).order_by(Group.name)
    )
    return [
        GroupOut(
            id=group.id,
            name=group.name,
            description=group.description,
            member_count=member_count,
            document_count=document_count,
        )
        for group, member_count, document_count in rows
    ]


@router.post("/groups", response_model=GroupOut, status_code=status.HTTP_201_CREATED)
async def create_group(body: GroupIn, admin: AdminDep, session: SessionDep) -> GroupOut:
    group = Group(name=body.name, description=body.description)
    session.add(group)
    try:
        await session.commit()
    except IntegrityError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, "A group with this name exists") from exc
    return GroupOut(
        id=group.id,
        name=group.name,
        description=group.description,
        member_count=0,
        document_count=0,
    )


@router.delete("/groups/{group_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_group(
    group_id: uuid.UUID, request: Request, admin: AdminDep, session: SessionDep
) -> None:
    """Delete a group. Its members lose access to documents shared only with it."""
    group = await session.get(Group, group_id)
    if group is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Group not found")
    await session.delete(group)
    await session.commit()
    await request.app.state.cache.bump_index_version()


# ---------------------------------------------------------------- users


@router.get("/users", response_model=list[UserOut])
async def list_users(
    admin: AdminDep,
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[UserOut]:
    users = await session.scalars(
        select(User)
        .options(selectinload(User.groups))
        .order_by(User.email)
        .limit(limit)
        .offset(offset)
    )
    return [UserOut.from_model(u) for u in users]


@router.post("/users", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def create_user(body: UserIn, admin: AdminDep, session: SessionDep) -> UserOut:
    user = User(
        email=body.email.strip().lower(),
        display_name=body.display_name,
        password_hash=hash_password(body.password),
        is_admin=body.is_admin,
        groups=await _groups_by_id(session, body.group_ids),
    )
    session.add(user)
    try:
        await session.commit()
    except IntegrityError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, "A user with this email exists") from exc
    return UserOut.from_model(await _user(session, user.id))


@router.get("/users/{user_id}", response_model=UserOut)
async def get_user(user_id: uuid.UUID, admin: AdminDep, session: SessionDep) -> UserOut:
    return UserOut.from_model(await _user(session, user_id))


@router.patch("/users/{user_id}", response_model=UserOut)
async def update_user(
    user_id: uuid.UUID, body: UserUpdate, admin: AdminDep, session: SessionDep
) -> UserOut:
    user = await _user(session, user_id)
    if user.id == admin.user_id and (body.is_admin is False or body.is_active is False):
        # Prevents the last admin from locking everyone out by accident.
        raise HTTPException(status.HTTP_409_CONFLICT, "You can't remove your own admin access")

    if body.display_name is not None:
        user.display_name = body.display_name
    if body.is_admin is not None:
        user.is_admin = body.is_admin
    if body.group_ids is not None:
        user.groups = await _groups_by_id(session, body.group_ids)
    if body.password is not None:
        user.password_hash = hash_password(body.password)
        user.token_version += 1  # signs the user out everywhere
    if body.is_active is not None:
        if user.is_active and not body.is_active:
            user.token_version += 1
        user.is_active = body.is_active
    await session.commit()
    return UserOut.from_model(await _user(session, user_id))


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: uuid.UUID, admin: AdminDep, session: SessionDep) -> None:
    if user_id == admin.user_id:
        raise HTTPException(status.HTTP_409_CONFLICT, "You can't delete your own account")
    user = await _user(session, user_id)
    await session.delete(user)
    await session.commit()


# ---------------------------------------------------------------- document access


@router.put("/documents/{document_id}/groups", response_model=list[GroupRef])
async def set_document_groups(
    document_id: uuid.UUID,
    body: DocumentGroupsIn,
    request: Request,
    admin: AdminDep,
    session: SessionDep,
) -> list[GroupRef]:
    """Replace the set of groups that can read a document. An empty list makes it
    admin-only."""
    document = await session.scalar(
        select(Document).where(Document.id == document_id).options(selectinload(Document.groups))
    )
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    document.groups = await _groups_by_id(session, body.group_ids)
    await session.commit()
    await request.app.state.cache.bump_index_version()  # cached results may now leak or miss
    return [GroupRef(id=g.id, name=g.name) for g in sorted(document.groups, key=lambda g: g.name)]
