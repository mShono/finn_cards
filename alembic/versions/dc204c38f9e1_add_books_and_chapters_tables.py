"""add books and chapters tables

Revision ID: dc204c38f9e1
Revises: 9c1a4e7d5b02
Create Date: 2026-11-20 18:07:40.117475

"""

from collections.abc import Sequence

import sqlalchemy as sa

import kielikaveri.db.models
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "dc204c38f9e1"
down_revision: str | Sequence[str] | None = "9c1a4e7d5b02"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Plan 2026-11-20-kirjakaveri-reader, "П1": the boundary between the
    # translator (writes fi_paragraphs) and the reader (reads them).
    op.create_table(
        "books",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("source_lang", sa.String(), nullable=False),
        sa.Column("source_hash", sa.String(), nullable=False),
        sa.Column("front_matter", sa.JSON(), nullable=False),
        sa.Column("glossary", sa.JSON(), nullable=True),
        sa.Column("style_notes", sa.String(), nullable=True),
        sa.Column("created_at", kielikaveri.db.models.UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("books", schema=None) as batch_op:
        batch_op.create_index("uq_books_user_hash", ["user_id", "source_hash"], unique=True)

    op.create_table(
        "chapters",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("book_id", sa.String(), nullable=False),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column("part", sa.String(), nullable=True),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("source_paragraphs", sa.JSON(), nullable=False),
        sa.Column("fi_paragraphs", sa.JSON(), nullable=True),
        sa.Column("fst_flags", sa.JSON(), nullable=True),
        sa.Column("review_issues", sa.JSON(), nullable=True),
        sa.Column(
            "status",
            sa.Enum(
                "pending",
                "queued",
                "translating",
                "done",
                "failed",
                name="chapterstatus",
                native_enum=False,
            ),
            nullable=False,
        ),
        sa.Column("error", sa.String(), nullable=True),
        sa.Column("opened_at", kielikaveri.db.models.UTCDateTime(), nullable=True),
        sa.Column("translated_at", kielikaveri.db.models.UTCDateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["book_id"],
            ["books.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("chapters", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_chapters_status"), ["status"], unique=False)
        batch_op.create_index("uq_chapters_book_number", ["book_id", "number"], unique=True)


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("chapters", schema=None) as batch_op:
        batch_op.drop_index("uq_chapters_book_number")
        batch_op.drop_index(batch_op.f("ix_chapters_status"))

    op.drop_table("chapters")
    with op.batch_alter_table("books", schema=None) as batch_op:
        batch_op.drop_index("uq_books_user_hash")

    op.drop_table("books")
