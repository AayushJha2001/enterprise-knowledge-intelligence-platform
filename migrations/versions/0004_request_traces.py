"""Request traces: latency, tokens, cost and outcome of every search/answer/extraction.

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "request_traces",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("endpoint", sa.String(20), nullable=False),
        sa.Column("request_id", sa.String(64)),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
        ),
        sa.Column("query", sa.Text),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("cached", sa.Boolean, server_default="false", nullable=False),
        sa.Column("model", sa.String(100)),
        sa.Column("input_tokens", sa.Integer, server_default="0", nullable=False),
        sa.Column("output_tokens", sa.Integer, server_default="0", nullable=False),
        sa.Column("cache_read_tokens", sa.Integer, server_default="0", nullable=False),
        sa.Column("cost_usd", sa.Float),
        sa.Column("latency_ms", sa.Float, nullable=False),
        sa.Column("timings_ms", postgresql.JSONB, server_default="{}", nullable=False),
        sa.Column("result_count", sa.Integer, server_default="0", nullable=False),
        sa.Column("top_score", sa.Float),
        sa.Column("source_chunk_ids", postgresql.JSONB, server_default="[]", nullable=False),
        sa.Column("error", sa.Text),
    )
    op.create_index("ix_request_traces_created_at", "request_traces", ["created_at"])
    op.create_index(
        "ix_request_traces_endpoint_created_at", "request_traces", ["endpoint", "created_at"]
    )


def downgrade() -> None:
    op.drop_table("request_traces")
