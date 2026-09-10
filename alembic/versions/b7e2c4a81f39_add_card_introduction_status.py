"""card introduction status, separate from the FSRS state

Revision ID: b7e2c4a81f39
Revises: a1f6d3b9c204
Create Date: 2026-09-10 15:05:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b7e2c4a81f39"
down_revision: str | Sequence[str] | None = "a1f6d3b9c204"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("cards") as batch_op:
        batch_op.add_column(sa.Column("status", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("introduced_at", sa.Integer(), nullable=True))

    # Every card that already exists was already in rotation - marking them
    # introduced keeps queues, debt and FSRS state exactly as they were.
    # Nothing is retroactively hidden from the learner by this migration;
    # only cards created from here on start out not_introduced.
    op.execute(sa.text("UPDATE cards SET status = 'introduced' WHERE status IS NULL"))

    # introduced_at stays NULL for them on purpose: it exists to count
    # *today's* introductions against the daily form budget, and these
    # happened before the budget existed.
    with op.batch_alter_table("cards") as batch_op:
        batch_op.alter_column("status", existing_type=sa.String(), nullable=False)
        batch_op.create_index("ix_cards_status", ["status"])


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("cards") as batch_op:
        batch_op.drop_index("ix_cards_status")
        batch_op.drop_column("introduced_at")
        batch_op.drop_column("status")
