"""Gradual opening of card types per note (see plans/.../kielikaveri-bot.md, 3.3).

recognition opens immediately, production once recognition's interval
(stability) crosses a threshold, inflection once the note's principal_forms
are FST-verified. usage is opened by hand (not implemented here) - it isn't
tied to any automatic condition.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kielikaveri.db.models import Card, CardType, Note

PRODUCTION_STABILITY_THRESHOLD_DAYS = 3.0


async def ensure_card_types(session: AsyncSession, note: Note, now: datetime) -> list[Card]:
    """Create any review-card types `note` has become eligible for.

    Safe to call repeatedly (e.g. after every review, or lazily before
    building a /learn queue) - each type is created at most once per note.
    """
    existing = (await session.scalars(select(Card).where(Card.note_id == note.id))).all()
    by_type = {card.type: card for card in existing}
    created: list[Card] = []

    if CardType.recognition not in by_type:
        card = Card(note_id=note.id, user_id=note.user_id, type=CardType.recognition, due=now)
        session.add(card)
        created.append(card)
        by_type[CardType.recognition] = card

    recognition = by_type.get(CardType.recognition)
    if (
        CardType.production not in by_type
        and recognition is not None
        and (recognition.stability or 0.0) >= PRODUCTION_STABILITY_THRESHOLD_DAYS
    ):
        card = Card(note_id=note.id, user_id=note.user_id, type=CardType.production, due=now)
        session.add(card)
        created.append(card)

    if (
        CardType.inflection not in by_type
        and note.meta.get("forms_verified") is True
        and note.meta.get("principal_forms")
    ):
        card = Card(note_id=note.id, user_id=note.user_id, type=CardType.inflection, due=now)
        session.add(card)
        created.append(card)

    return created


async def sync_user_card_types(session: AsyncSession, user_id: int, now: datetime) -> list[Card]:
    """Run ensure_card_types for every note the user owns. Does not commit."""
    notes = (await session.scalars(select(Note).where(Note.user_id == user_id))).all()
    created: list[Card] = []
    for note in notes:
        created.extend(await ensure_card_types(session, note, now))
    return created
