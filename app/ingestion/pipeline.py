"""The ingestion job: claim a pending document, parse, chunk, embed, store.

The queue is the documents table itself. A worker claims the oldest pending row with
`FOR UPDATE SKIP LOCKED`, so any number of workers can run side by side without two of
them taking the same document, and no separate message broker is needed. A document
whose worker crashed is put back in the queue once it has been "processing" too long.
"""

import asyncio
import logging
import time
import uuid
from datetime import UTC, datetime

from sqlalchemy import delete, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.db.models import Chunk, Document
from app.ingestion.chunking import chunk_blocks, embedding_text
from app.ingestion.embeddings import Embedder
from app.ingestion.parsers import ParseError, parse_document
from app.ingestion.storage import FileStorage

logger = logging.getLogger(__name__)

_CLAIM_NEXT = text(
    """
    UPDATE documents
       SET status = 'processing', attempts = attempts + 1, updated_at = now()
     WHERE id = (
            SELECT id FROM documents
             WHERE status = 'pending'
             ORDER BY created_at
             FOR UPDATE SKIP LOCKED
             LIMIT 1
           )
    RETURNING id
    """
)

_REQUEUE_STALE = text(
    """
    UPDATE documents
       SET status = 'pending', updated_at = now()
     WHERE status = 'processing'
       AND updated_at < now() - make_interval(secs => :stale_after)
    RETURNING id
    """
)


async def requeue_stale(
    session_factory: async_sessionmaker[AsyncSession], settings: Settings
) -> int:
    async with session_factory() as session:
        result = await session.execute(
            _REQUEUE_STALE, {"stale_after": settings.worker_stale_after_s}
        )
        ids = result.scalars().all()
        await session.commit()
    for doc_id in ids:
        logger.warning("requeued stale document", extra={"document_id": str(doc_id)})
    return len(ids)


async def claim_next(session_factory: async_sessionmaker[AsyncSession]) -> uuid.UUID | None:
    async with session_factory() as session:
        doc_id = (await session.execute(_CLAIM_NEXT)).scalar_one_or_none()
        await session.commit()
        return doc_id


async def process_document(
    document_id: uuid.UUID,
    session_factory: async_sessionmaker[AsyncSession],
    storage: FileStorage,
    embedder: Embedder,
    settings: Settings,
) -> str:
    """Ingest one claimed document. Returns its final status."""
    started = time.perf_counter()
    async with session_factory() as session:
        doc = await session.get(Document, document_id)
        if doc is None:  # deleted after being claimed
            return "deleted"
        log_extra = {"document_id": str(doc.id), "attempt": doc.attempts}

        try:
            data = await asyncio.to_thread(storage.read, doc.storage_key)
            parsed = await asyncio.to_thread(parse_document, data, doc.mime_type)
            drafts = chunk_blocks(
                parsed.blocks,
                target_tokens=settings.chunk_target_tokens,
                max_tokens=settings.chunk_max_tokens,
                overlap_tokens=settings.chunk_overlap_tokens,
            )
            # Use the document's own title for chunk headers when the user didn't give one.
            title = doc.title
            if parsed.title and doc.title == doc.original_filename.rsplit(".", 1)[0]:
                title = parsed.title
            vectors = await asyncio.to_thread(
                embedder.embed_passages, [embedding_text(d, title) for d in drafts]
            )

            # Replace all chunks in one transaction: readers see old chunks or new, never half.
            await session.execute(delete(Chunk).where(Chunk.document_id == doc.id))
            session.add_all(
                Chunk(
                    document_id=doc.id,
                    chunk_index=draft.index,
                    content=draft.text,
                    token_count=draft.token_count,
                    page_number=draft.page_start,
                    section_path=" > ".join(draft.section_path) or None,
                    metadata_={"page_end": draft.page_end} if draft.page_end else {},
                    embedding=vector,
                )
                for draft, vector in zip(drafts, vectors, strict=True)
            )
            doc.title = title
            doc.status = "ready"
            doc.error = None
            doc.chunk_count = len(drafts)
            doc.page_count = parsed.page_count
            doc.embedding_model = embedder.model_name
            doc.processed_at = datetime.now(UTC)
            await session.commit()
            logger.info(
                "document ingested in %.0f ms",
                (time.perf_counter() - started) * 1000,
                extra={**log_extra, "document_status": "ready", "chunk_count": len(drafts)},
            )
            return "ready"

        except ParseError as exc:
            # The file itself is the problem; retrying gives the same result.
            await session.rollback()
            return await _mark(session, document_id, "failed", str(exc), log_extra)
        except Exception as exc:
            # Possibly transient (DB blip, out of memory): retry until attempts run out.
            await session.rollback()
            logger.exception("ingestion error", extra=log_extra)
            attempts = await session.scalar(
                select(Document.attempts).where(Document.id == document_id)
            )
            if attempts is None:
                return "deleted"
            final = "failed" if attempts >= settings.worker_max_attempts else "pending"
            return await _mark(
                session, document_id, final, f"{type(exc).__name__}: {exc}"[:2000], log_extra
            )


async def _mark(
    session: AsyncSession, document_id: uuid.UUID, status: str, error: str, log_extra: dict
) -> str:
    await session.execute(
        update(Document).where(Document.id == document_id).values(status=status, error=error)
    )
    await session.commit()
    logger.warning(
        "ingestion %s: %s", status, error, extra={**log_extra, "document_status": status}
    )
    return status


async def process_next(
    session_factory: async_sessionmaker[AsyncSession],
    storage: FileStorage,
    embedder: Embedder,
    settings: Settings,
) -> str | None:
    """Claim and process one document. Returns its status, or None if the queue is empty."""
    document_id = await claim_next(session_factory)
    if document_id is None:
        return None
    return await process_document(document_id, session_factory, storage, embedder, settings)
