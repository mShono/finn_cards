"""add lookup_cache table

Revision ID: 3f9a6c2e8b15
Revises: 8e4c1a7f2d90
Create Date: 2026-11-24 18:00:00.000000

Plan 2026-11-20-kirjakaveri-reader, "П3": the reader's word lookups,
cached so a second tap on the same word in the same sentence costs no
OpenAI call. Rows hold text derived from the book, so they go with its
chapter (ON DELETE CASCADE).
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op
from kielikaveri.db.models import UTCDateTime

# revision identifiers, used by Alembic.
revision: str = "3f9a6c2e8b15"
down_revision: str | Sequence[str] | None = "8e4c1a7f2d90"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "lookup_cache",
        sa.Column("key", sa.String(), nullable=False),
        sa.Column("chapter_id", sa.String(), nullable=False),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["chapter_id"],
            ["chapters.id"],
            name="fk_lookup_cache_chapter_id_chapters",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("key"),
    )
    with op.batch_alter_table("lookup_cache", schema=None) as batch_op:
        batch_op.create_index("ix_lookup_cache_chapter_id", ["chapter_id"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("lookup_cache", schema=None) as batch_op:
        batch_op.drop_index("ix_lookup_cache_chapter_id")
    op.drop_table("lookup_cache")
