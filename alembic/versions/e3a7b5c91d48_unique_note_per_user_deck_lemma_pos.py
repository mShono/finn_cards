"""unique note per user, deck, lemma, pos

Revision ID: e3a7b5c91d48
Revises: b7e2c4a81f39
Create Date: 2026-09-11 18:10:00.000000

Makes the DB the last line of defence behind /add's Python dedup
(ingest.existing_note_keys + canonical_key), which on its own leaves a race:
two concurrent turns both pass the pre-insert lookup and both insert.

COALESCE, not a plain UNIQUE constraint: SQLite counts NULLs as distinct in a
unique index, and both `pos` (NULL for every kind="pattern" note) and
`deck_id` (NULL for pre-decks rows and for import_cards.py) are legitimately
NULL - a plain constraint would leave exactly those free to duplicate.

Refuses to run if the data already violates the key. Nothing is merged or
deleted here: which of two real notes to keep is the learner's call, not a
migration's.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e3a7b5c91d48"
down_revision: str | Sequence[str] | None = "b7e2c4a81f39"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

INDEX_NAME = "uq_notes_user_deck_lemma_pos"

# Mirrors the index expression exactly, so "no rows here" really means "the
# index can be built".
DUPLICATE_QUERY = sa.text(
    """
    SELECT user_id,
           coalesce(deck_id, '') AS deck_key,
           lemma,
           coalesce(pos, '')     AS pos_key,
           COUNT(*)              AS n,
           group_concat(id)      AS ids
    FROM notes
    GROUP BY user_id, lemma, coalesce(deck_id, ''), coalesce(pos, '')
    HAVING COUNT(*) > 1
    ORDER BY n DESC, lemma
    """
)


def _assert_no_duplicates(bind) -> None:
    rows = bind.execute(DUPLICATE_QUERY).all()
    if not rows:
        return
    details = "\n".join(
        f"  user_id={row.user_id} deck_id={row.deck_key or 'NULL'} "
        f"lemma={row.lemma!r} pos={row.pos_key or 'NULL'} "
        f"count={row.n} note_ids={row.ids}"
        for row in rows
    )
    raise RuntimeError(
        f"Cannot create {INDEX_NAME}: {len(rows)} duplicate group(s) already in `notes`.\n"
        f"{details}\n"
        "Nothing was changed. Decide per group which note to keep (e.g. via /delete "
        "in the bot, which shows each note's deck), then re-run this migration."
    )


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()
    _assert_no_duplicates(bind)
    op.create_index(
        INDEX_NAME,
        "notes",
        ["user_id", "lemma", sa.text("coalesce(deck_id, '')"), sa.text("coalesce(pos, '')")],
        unique=True,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(INDEX_NAME, table_name="notes")
