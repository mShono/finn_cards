"""index cards.note_id - /learn looks cards up by note on every session

Revision ID: f2b81c6a9d30
Revises: e3a7b5c91d48
Create Date: 2026-09-15 12:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f2b81c6a9d30"
down_revision: str | Sequence[str] | None = "e3a7b5c91d48"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

INDEX_NAME = "ix_cards_note_id"


def upgrade() -> None:
    """Upgrade schema."""
    # `cards.note_id` is a ForeignKey, and SQLite creates no index for one.
    # Both surviving "cards of this note" queries had to scan the whole table:
    # graduation.cards_by_note (one per /learn, joined through notes) and
    # ensure_card_types (one per rating, WHERE note_id = ?). Nothing else is
    # indexed here on speculation - `cards.status` already has its own.
    op.create_index(INDEX_NAME, "cards", ["note_id"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(INDEX_NAME, table_name="cards")
