"""User authentication: active flag, token version (revocation), last login.

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "users", sa.Column("is_active", sa.Boolean, server_default="true", nullable=False)
    )
    op.add_column(
        "users", sa.Column("token_version", sa.Integer, server_default="0", nullable=False)
    )
    op.add_column("users", sa.Column("last_login_at", sa.DateTime(timezone=True)))


def downgrade() -> None:
    op.drop_column("users", "last_login_at")
    op.drop_column("users", "token_version")
    op.drop_column("users", "is_active")
