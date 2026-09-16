"""drop sources table and notes.source_id - written by /add, never read

Revision ID: 05cddfc3c4b2
Revises: f2b81c6a9d30
Create Date: 2026-09-16 14:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "05cddfc3c4b2"
down_revision: str | Sequence[str] | None = "f2b81c6a9d30"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Same definition as e3a7b5c91d48 created.
NOTE_INDEX_NAME = "uq_notes_user_deck_lemma_pos"
NOTE_INDEX_COLUMNS = [
    "user_id",
    "lemma",
    sa.text("coalesce(deck_id, '')"),
    sa.text("coalesce(pos, '')"),
]


def upgrade() -> None:
    """Upgrade schema."""
    # Batch mode rebuilds `notes` from reflection, and SQLAlchemy can't reflect
    # an expression index ("Skipped unsupported reflection of expression-based
    # index") - left in place, the unique index would silently vanish with the
    # rebuild. Drop it first, recreate it on the rebuilt table.
    op.drop_index(NOTE_INDEX_NAME, table_name="notes")

    # The FK on source_id is unnamed in the initial schema (named after a
    # downgrade) - either way the rebuild drops it along with the column.
    with op.batch_alter_table("notes") as batch_op:
        batch_op.drop_column("source_id")

    op.create_index(NOTE_INDEX_NAME, "notes", NOTE_INDEX_COLUMNS, unique=True)

    op.drop_table("sources")


def downgrade() -> None:
    """Downgrade schema."""
    # Restores the structure only - the dropped rows are gone.
    op.create_table(
        "sources",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column(
            "type",
            sa.Enum(
                "video",
                "audio",
                "article",
                "conversation",
                "book",
                "other",
                name="sourcetype",
                native_enum=False,
            ),
            nullable=False,
        ),
        sa.Column("ref", sa.String(), nullable=False),
        sa.Column("context_fi", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.drop_index(NOTE_INDEX_NAME, table_name="notes")

    # Named, unlike the initial schema's FK - batch mode refuses an unnamed
    # constraint ("Constraint must have a name").
    with op.batch_alter_table("notes") as batch_op:
        batch_op.add_column(sa.Column("source_id", sa.String(), nullable=True))
        batch_op.create_foreign_key("fk_notes_source_id_sources", "sources", ["source_id"], ["id"])

    op.create_index(NOTE_INDEX_NAME, "notes", NOTE_INDEX_COLUMNS, unique=True)
