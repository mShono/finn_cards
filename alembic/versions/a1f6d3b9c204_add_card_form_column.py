"""one inflection card per principal form

Revision ID: a1f6d3b9c204
Revises: 41b134c55432
Create Date: 2026-09-10 13:20:00.000000

"""

import json
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1f6d3b9c204"
down_revision: str | Sequence[str] | None = "41b134c55432"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# A snapshot of kielikaveri.grammar.FORM_TASKS as of this revision, in the
# order forms are generated. Deliberately copied rather than imported: a
# migration must keep doing the same thing to the same rows even after the
# application's tables change.
QUIZZABLE_FORMS = (
    "genetiivi",
    "partitiivi",
    "illatiivi",
    "inessiivi",
    "elatiivi",
    "adessiivi",
    "ablatiivi",
    "allatiivi",
    "essiivi",
    "translatiivi",
    "monikon_genetiivi",
    "monikon_partitiivi",
    "preesens_1s",
    "preesens_3s",
    "imperfekti_3s",
    "konditionaali_1s",
    "imperatiivi_2s",
    "nut_partisiippi",
    "passiivi",
)


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("cards") as batch_op:
        batch_op.add_column(sa.Column("form", sa.String(), nullable=True))

    # Existing inflection cards used to draw a random form each review. Give
    # each one a concrete form so its review history keeps counting for
    # something; the remaining forms get their own cards from
    # ensure_card_types() on the next /learn. Cards whose note has no usable
    # form stay NULL and are adopted there instead - never deleted, the
    # history on them is real.
    bind = op.get_bind()
    rows = bind.execute(
        sa.text(
            "SELECT c.id, n.meta FROM cards c JOIN notes n ON n.id = c.note_id "
            "WHERE c.type = 'inflection' AND c.form IS NULL"
        )
    ).fetchall()

    for card_id, meta in rows:
        forms = (json.loads(meta) if isinstance(meta, str) else meta or {}).get(
            "principal_forms"
        ) or {}
        chosen = next((name for name in QUIZZABLE_FORMS if name in forms), None)
        if chosen is None:
            continue
        bind.execute(
            sa.text("UPDATE cards SET form = :form WHERE id = :id"),
            {"form": chosen, "id": card_id},
        )


def downgrade() -> None:
    """Downgrade schema."""
    # The extra per-form cards this feature creates are left in place: they
    # carry their own review history, and dropping the column is enough to
    # put the schema back.
    with op.batch_alter_table("cards") as batch_op:
        batch_op.drop_column("form")
