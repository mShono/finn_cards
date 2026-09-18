"""drop users.last_deck_id - written on every save, never read

Revision ID: 9c1a4e7d5b02
Revises: 05cddfc3c4b2
Create Date: 2026-09-18 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9c1a4e7d5b02"
down_revision: str | Sequence[str] | None = "05cddfc3c4b2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # The "active deck" it held stopped meaning anything once /add and /learn
    # began asking which deck to use on every turn - only the /decks screen
    # read it back, to describe a behaviour that no longer existed.
    with op.batch_alter_table("users") as batch_op:
        batch_op.drop_column("last_deck_id")


def downgrade() -> None:
    """Downgrade schema."""
    # Restores the column only - the deck ids it held are gone.
    with op.batch_alter_table("users") as batch_op:
        batch_op.add_column(sa.Column("last_deck_id", sa.String(), nullable=True))
