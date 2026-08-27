"""add needs_clarification to ingest_cache

Revision ID: 41b134c55432
Revises: bf271c02f743
Create Date: 2026-08-27 10:00:52.632685

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "41b134c55432"
down_revision: str | Sequence[str] | None = "bf271c02f743"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("ingest_cache") as batch_op:
        batch_op.add_column(sa.Column("needs_clarification", sa.Boolean(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("ingest_cache") as batch_op:
        batch_op.drop_column("needs_clarification")
