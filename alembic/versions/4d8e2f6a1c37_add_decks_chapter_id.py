"""add decks.chapter_id

Revision ID: 4d8e2f6a1c37
Revises: dc204c38f9e1
Create Date: 2026-11-23 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "4d8e2f6a1c37"
down_revision: str | Sequence[str] | None = "dc204c38f9e1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Plan И7: a reader chapter deck is found by chapter_id, not by name, and
    # the unique index makes its get-or-create race-free.
    with op.batch_alter_table("decks", schema=None) as batch_op:
        batch_op.add_column(sa.Column("chapter_id", sa.String(), nullable=True))
        batch_op.create_foreign_key(
            "fk_decks_chapter_id_chapters",
            "chapters",
            ["chapter_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_index("uq_decks_chapter_id", ["chapter_id"], unique=True)


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("decks", schema=None) as batch_op:
        batch_op.drop_index("uq_decks_chapter_id")
        batch_op.drop_constraint("fk_decks_chapter_id_chapters", type_="foreignkey")
        batch_op.drop_column("chapter_id")
