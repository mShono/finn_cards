"""add decks table and note deck_id

Revision ID: bf271c02f743
Revises: 884334e01440
Create Date: 2026-08-26 11:47:51.950024

"""

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

import sqlalchemy as sa

import kielikaveri.db.models
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "bf271c02f743"
down_revision: str | Sequence[str] | None = "884334e01440"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

DEFAULT_DECK_NAME = "Общая"


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "decks",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("created_at", kielikaveri.db.models.UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("ingest_cache") as batch_op:
        batch_op.add_column(sa.Column("reply_ru", sa.String(), nullable=True))

    with op.batch_alter_table("notes") as batch_op:
        batch_op.add_column(sa.Column("deck_id", sa.String(), nullable=True))
        batch_op.create_foreign_key("fk_notes_deck_id_decks", "decks", ["deck_id"], ["id"])

    with op.batch_alter_table("users") as batch_op:
        batch_op.add_column(sa.Column("last_deck_id", sa.String(), nullable=True))

    # Backfill: every existing note lands in a per-user "Общая" deck, so
    # pre-decks data doesn't just vanish from /learn once queue filtering by
    # deck_id ships (see srs/queue.py).
    decks = sa.table(
        "decks",
        sa.column("id", sa.String),
        sa.column("user_id", sa.Integer),
        sa.column("name", sa.String),
        sa.column("created_at", kielikaveri.db.models.UTCDateTime()),
    )
    notes = sa.table(
        "notes",
        sa.column("id", sa.String),
        sa.column("user_id", sa.Integer),
        sa.column("deck_id", sa.String),
    )
    bind = op.get_bind()
    user_ids = bind.execute(sa.select(notes.c.user_id).distinct()).scalars().all()
    now = datetime.now(UTC)
    for user_id in user_ids:
        deck_id = str(uuid.uuid4())
        bind.execute(
            decks.insert().values(
                id=deck_id, user_id=user_id, name=DEFAULT_DECK_NAME, created_at=now
            )
        )
        bind.execute(notes.update().where(notes.c.user_id == user_id).values(deck_id=deck_id))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("users") as batch_op:
        batch_op.drop_column("last_deck_id")

    with op.batch_alter_table("notes") as batch_op:
        batch_op.drop_constraint("fk_notes_deck_id_decks", type_="foreignkey")
        batch_op.drop_column("deck_id")

    with op.batch_alter_table("ingest_cache") as batch_op:
        batch_op.drop_column("reply_ru")

    op.drop_table("decks")
