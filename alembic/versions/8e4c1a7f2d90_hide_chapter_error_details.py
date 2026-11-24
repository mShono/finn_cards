"""hide exception messages in chapters.error

Revision ID: 8e4c1a7f2d90
Revises: 6b3d9e2f4a17
Create Date: 2026-11-24 12:00:00.000000

Plan М17: chapters.error held "Class: str(exc)", and str() of an API or JSON
error, or of a model's refusal, can quote the book. The translator now keeps
the message only for its own exceptions (worker._SAFE_MESSAGE_TYPES); this
cuts the existing rows to the same shape:

- NULL - left alone;
- "Class: message" with Class on the safe list - left alone;
- "Class: anything else" - "Class: (details hidden)" (the class stays: the bot
  picks its reason by it);
- no ": " at all - left alone if it is just a class name (an identifier),
  otherwise "UnknownError: (details hidden)", since what it holds is unknown;
- a prefix before ": " that is not an identifier - the same as above.

Nothing is printed or logged: only row ids and the new values are handled.

downgrade() restores nothing - the messages are gone, there is nothing to
restore them from.
"""

import re
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8e4c1a7f2d90"
down_revision: str | Sequence[str] | None = "6b3d9e2f4a17"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# By name, frozen here: a migration must not change with worker.py.
_SAFE_CLASSES = frozenset(
    {"MisalignedError", "CircuitOpenError", "BudgetExceededError", "_ChapterGone"}
)
_HIDDEN = "(details hidden)"
_UNKNOWN = "UnknownError"
_CLASS_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")


def hidden_error(error: str) -> str:
    """The value a chapters.error row is cut to (may be the same string)."""
    if ": " not in error:
        return error if _CLASS_RE.fullmatch(error) else f"{_UNKNOWN}: {_HIDDEN}"
    cls, _ = error.split(": ", 1)
    if not _CLASS_RE.fullmatch(cls):
        return f"{_UNKNOWN}: {_HIDDEN}"
    if cls in _SAFE_CLASSES:
        return error
    return f"{cls}: {_HIDDEN}"


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()
    rows = bind.execute(sa.text("SELECT id, error FROM chapters WHERE error IS NOT NULL"))
    updates = [
        {"id": row_id, "error": new}
        for row_id, error in rows.all()
        if (new := hidden_error(error)) != error
    ]
    if updates:
        bind.execute(sa.text("UPDATE chapters SET error = :error WHERE id = :id"), updates)


def downgrade() -> None:
    """Downgrade schema: nothing - the hidden messages cannot be restored."""
