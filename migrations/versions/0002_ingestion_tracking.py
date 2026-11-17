"""Ingestion tracking: file storage key, attempts, chunk/page counts, embedding model.

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # No documents can exist yet (Phase 1 had no upload endpoint), so NOT NULL is safe.
    op.add_column("documents", sa.Column("original_filename", sa.String(500), nullable=False))
    op.add_column("documents", sa.Column("storage_key", sa.String(255), nullable=False))
    op.add_column(
        "documents", sa.Column("attempts", sa.Integer, server_default="0", nullable=False)
    )
    op.add_column(
        "documents", sa.Column("chunk_count", sa.Integer, server_default="0", nullable=False)
    )
    op.add_column("documents", sa.Column("page_count", sa.Integer))
    op.add_column("documents", sa.Column("embedding_model", sa.String(200)))
    op.add_column("documents", sa.Column("processed_at", sa.DateTime(timezone=True)))
    op.create_index("ix_documents_status_created_at", "documents", ["status", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_documents_status_created_at", table_name="documents")
    for column in (
        "processed_at",
        "embedding_model",
        "page_count",
        "chunk_count",
        "attempts",
        "storage_key",
        "original_filename",
    ):
        op.drop_column("documents", column)
