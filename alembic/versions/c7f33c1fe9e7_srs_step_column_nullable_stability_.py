"""srs step column, nullable stability/difficulty

Revision ID: c7f33c1fe9e7
Revises: d64d09a9e67e
Create Date: 2026-08-24 12:53:27.770583

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c7f33c1fe9e7"
down_revision: str | Sequence[str] | None = "d64d09a9e67e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # batch_alter_table, not op.alter_column directly - SQLite has no ALTER
    # COLUMN at all (any version), so a bare alter_column() emits DDL SQLite
    # can't parse. Batch mode rebuilds the table instead.
    with op.batch_alter_table("cards") as batch_op:
        batch_op.add_column(sa.Column("step", sa.Integer(), nullable=True))
        batch_op.alter_column("stability", existing_type=sa.FLOAT(), nullable=True)
        batch_op.alter_column("difficulty", existing_type=sa.FLOAT(), nullable=True)


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("cards") as batch_op:
        batch_op.alter_column("difficulty", existing_type=sa.FLOAT(), nullable=False)
        batch_op.alter_column("stability", existing_type=sa.FLOAT(), nullable=False)
        batch_op.drop_column("step")
