"""Document upload and management.

Uploading only stores the file and queues it (202 Accepted); the ingestion worker parses,
chunks and embeds it in the background. Poll GET /documents/{id} until status is "ready".

Authentication arrives in Phase 5; until then these endpoints are open, so only run the
stack locally.
"""

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime
from pathlib import PurePath
from typing import Annotated, Literal

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import get_settings
from app.db.models import Chunk, Document, Group
from app.db.session import get_session
from app.ingestion.parsers import MIME_TYPES, detect_mime_type
from app.ingestion.storage import FileStorage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
DocumentStatus = Literal["pending", "processing", "ready", "failed"]


class DocumentOut(BaseModel):
    id: uuid.UUID
    title: str
    original_filename: str
    mime_type: str
    size_bytes: int
    status: DocumentStatus
    error: str | None
    attempts: int
    chunk_count: int
    page_count: int | None
    embedding_model: str | None
    group_ids: list[uuid.UUID]
    created_at: datetime
    updated_at: datetime
    processed_at: datetime | None

    @classmethod
    def from_model(cls, doc: Document) -> "DocumentOut":
        return cls(
            id=doc.id,
            title=doc.title,
            original_filename=doc.original_filename,
            mime_type=doc.mime_type,
            size_bytes=doc.size_bytes,
            status=doc.status,
            error=doc.error,
            attempts=doc.attempts,
            chunk_count=doc.chunk_count,
            page_count=doc.page_count,
            embedding_model=doc.embedding_model,
            group_ids=sorted(g.id for g in doc.groups),
            created_at=doc.created_at,
            updated_at=doc.updated_at,
            processed_at=doc.processed_at,
        )


class DocumentList(BaseModel):
    items: list[DocumentOut]
    total: int
    limit: int
    offset: int


class ChunkOut(BaseModel):
    id: uuid.UUID
    chunk_index: int
    content: str
    token_count: int
    page_number: int | None
    section_path: str | None
    metadata: dict


class ChunkList(BaseModel):
    items: list[ChunkOut]
    total: int
    limit: int
    offset: int


def _storage(request: Request) -> FileStorage:
    return request.app.state.storage


async def _get_document(session: AsyncSession, document_id: uuid.UUID) -> Document:
    doc = await session.scalar(
        select(Document).where(Document.id == document_id).options(selectinload(Document.groups))
    )
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return doc


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=DocumentOut,
    responses={
        409: {"description": "The same file was already uploaded"},
        413: {"description": "File too large"},
        415: {"description": "Unsupported file type"},
    },
)
async def upload_document(
    request: Request,
    session: SessionDep,
    file: Annotated[UploadFile, File(description="PDF, DOCX, Markdown, HTML or plain text")],
    title: Annotated[str | None, Form(max_length=500)] = None,
    group_ids: Annotated[
        list[uuid.UUID] | None, Form(description="Groups allowed to read this document")
    ] = None,
) -> DocumentOut:
    settings = get_settings()
    filename = PurePath(file.filename or "upload").name

    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"File exceeds {settings.max_upload_bytes // (1024 * 1024)} MB",
        )
    if not data:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "File is empty")

    mime_type = detect_mime_type(filename, data[:8])
    if mime_type is None:
        supported = ", ".join(sorted(MIME_TYPES))
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported or mislabelled file. Supported extensions: {supported}",
        )

    checksum = hashlib.sha256(data).hexdigest()
    existing = await session.scalar(select(Document.id).where(Document.checksum_sha256 == checksum))
    if existing is not None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            {"message": "This file was already uploaded", "document_id": str(existing)},
        )

    groups: list[Group] = []
    if group_ids:
        wanted = set(group_ids)
        groups = list(await session.scalars(select(Group).where(Group.id.in_(wanted))))
        missing = wanted - {g.id for g in groups}
        if missing:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                {"message": "Unknown group ids", "group_ids": sorted(map(str, missing))},
            )

    storage_key = f"{checksum[:2]}/{checksum}{PurePath(filename).suffix.lower()}"
    storage = _storage(request)
    await asyncio.to_thread(storage.save, storage_key, data)

    doc = Document(
        title=(title or "").strip() or PurePath(filename).stem,
        original_filename=filename,
        storage_key=storage_key,
        mime_type=mime_type,
        checksum_sha256=checksum,
        size_bytes=len(data),
        status="pending",
        groups=groups,
    )
    session.add(doc)
    try:
        await session.commit()
    except IntegrityError as exc:
        # Two identical uploads raced past the duplicate check; the other one won.
        # The stored file is identical (same checksum), so keep it.
        await session.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "This file was already uploaded") from exc

    logger.info(
        "document queued",
        extra={
            "request_id": getattr(request.state, "request_id", None),
            "document_id": str(doc.id),
        },
    )
    return DocumentOut.from_model(await _get_document(session, doc.id))


@router.get("", response_model=DocumentList)
async def list_documents(
    session: SessionDep,
    status_filter: Annotated[DocumentStatus | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentList:
    query = select(Document)
    count = select(func.count()).select_from(Document)
    if status_filter:
        query = query.where(Document.status == status_filter)
        count = count.where(Document.status == status_filter)
    docs = await session.scalars(
        query.options(selectinload(Document.groups))
        .order_by(Document.created_at.desc(), Document.id)
        .limit(limit)
        .offset(offset)
    )
    return DocumentList(
        items=[DocumentOut.from_model(d) for d in docs],
        total=await session.scalar(count) or 0,
        limit=limit,
        offset=offset,
    )


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(document_id: uuid.UUID, session: SessionDep) -> DocumentOut:
    return DocumentOut.from_model(await _get_document(session, document_id))


@router.get("/{document_id}/chunks", response_model=ChunkList)
async def list_chunks(
    document_id: uuid.UUID,
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ChunkList:
    """Inspect how a document was chunked (useful when tuning chunk sizes)."""
    await _get_document(session, document_id)
    chunks = await session.scalars(
        select(Chunk)
        .where(Chunk.document_id == document_id)
        .order_by(Chunk.chunk_index)
        .limit(limit)
        .offset(offset)
    )
    total = await session.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.document_id == document_id)
    )
    return ChunkList(
        items=[
            ChunkOut(
                id=c.id,
                chunk_index=c.chunk_index,
                content=c.content,
                token_count=c.token_count,
                page_number=c.page_number,
                section_path=c.section_path,
                metadata=c.metadata_,
            )
            for c in chunks
        ],
        total=total or 0,
        limit=limit,
        offset=offset,
    )


@router.post(
    "/{document_id}/reindex", status_code=status.HTTP_202_ACCEPTED, response_model=DocumentOut
)
async def reindex_document(document_id: uuid.UUID, session: SessionDep) -> DocumentOut:
    """Queue a document for processing again (e.g. after changing the chunking settings)."""
    doc = await _get_document(session, document_id)
    if doc.status in ("pending", "processing"):
        raise HTTPException(status.HTTP_409_CONFLICT, f"Document is already {doc.status}")
    doc.status = "pending"
    doc.attempts = 0
    doc.error = None
    await session.commit()
    return DocumentOut.from_model(await _get_document(session, document_id))


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(request: Request, document_id: uuid.UUID, session: SessionDep) -> None:
    doc = await _get_document(session, document_id)
    storage_key = doc.storage_key
    await session.delete(doc)  # chunks and access rows cascade in the database
    await session.commit()
    # Delete the file only after the row is gone, so a failure never leaves a
    # document pointing at a missing file.
    await asyncio.to_thread(_storage(request).delete, storage_key)
