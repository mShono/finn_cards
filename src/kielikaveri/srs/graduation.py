"""Gradual opening of card types per note (see plans/.../kielikaveri-bot.md, 3.3).

recognition opens immediately, production once recognition's interval
(stability) crosses a threshold, inflection once the note's principal_forms
are FST-verified - and there it is one card per form, not one per note, so
each form carries its own FSRS schedule. usage is opened by hand (not
implemented here) - it isn't tied to any automatic condition.

Opening a dozen inflection cards at once does not flood a session:
build_session_queue admits at most `daily_new_limit` never-reviewed cards
per study day, so the forms are introduced a few at a time.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kielikaveri.db.models import Card, CardType, Note
from kielikaveri.grammar import FORM_TASKS

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

    if note.meta.get("forms_verified") is True:
        created.extend(_ensure_inflection_cards(session, note, existing, now))

    return created


def _ensure_inflection_cards(
    session: AsyncSession, note: Note, existing: Sequence[Card], now: datetime
) -> list[Card]:
    """One inflection card per quizzable principal form, created at most once each."""
    forms: dict = note.meta.get("principal_forms") or {}
    wanted = [name for name in forms if name in FORM_TASKS]
    inflection = [card for card in existing if card.type == CardType.inflection]
    covered = {card.form for card in inflection}
    created: list[Card] = []

    # A card left over from when one inflection card quizzed a random form
    # (or from a note whose forms were still empty at migration time) is
    # given a form rather than replaced - it carries real review history.
    orphans = [card for card in inflection if card.form is None]
    for card, name in zip(orphans, (n for n in wanted if n not in covered), strict=False):
        card.form = name
        covered.add(name)

    for name in wanted:
        if name in covered:
            continue
        card = Card(
            note_id=note.id, user_id=note.user_id, type=CardType.inflection, form=name, due=now
        )
        session.add(card)
        created.append(card)
        covered.add(name)

    return created


async def sync_user_card_types(session: AsyncSession, user_id: int, now: datetime) -> list[Card]:
    """Run ensure_card_types for every note the user owns. Does not commit."""
    notes = (await session.scalars(select(Note).where(Note.user_id == user_id))).all()
    created: list[Card] = []
    for note in notes:
        created.extend(await ensure_card_types(session, note, now))
    return created
