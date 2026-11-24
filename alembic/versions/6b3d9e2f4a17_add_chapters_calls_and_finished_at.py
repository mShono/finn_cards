"""add chapters.calls and chapters.finished_at

Revision ID: 6b3d9e2f4a17
Revises: 4d8e2f6a1c37
Create Date: 2026-11-23 18:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

import kielikaveri.db.models
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "6b3d9e2f4a17"
down_revision: str | Sequence[str] | None = "4d8e2f6a1c37"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Plan М15: the translator's daily cap survives a restart - on start it
    # counts the calls of chapters finished in the last 24 hours. Existing
    # chapters stay NULL: their calls are unknown and were made before.
    with op.batch_alter_table("chapters", schema=None) as batch_op:
        batch_op.add_column(sa.Column("calls", sa.Integer(), nullable=True))
        batch_op.add_column(
            sa.Column("finished_at", kielikaveri.db.models.UTCDateTime(), nullable=True)
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("chapters", schema=None) as batch_op:
        batch_op.drop_column("finished_at")
        batch_op.drop_column("calls")
