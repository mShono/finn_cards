"""Gradual opening of card types per note (see plans/.../kielikaveri-bot.md, 3.3).

recognition opens immediately, production once recognition's interval
(stability) crosses a threshold, inflection once the note's principal_forms
are FST-verified - and there it is one card per form, not one per note, so
each form carries its own FSRS schedule.

Opening a dozen inflection cards at once does not flood a session:
build_session_queue admits at most `daily_new_limit` never-reviewed cards
per study day, so the forms are introduced a few at a time.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from kielikaveri.db.models import Card, CardStatus, CardType, Note, Review
from kielikaveri.grammar import FORM_TASKS

PRODUCTION_STABILITY_THRESHOLD_DAYS = 3.0


async def cards_by_note(
    session: AsyncSession, user_id: int, deck_id: str | None = None
) -> dict[str, list[Card]]:
    """Every card belonging to the user's notes, grouped by note id, in one query.

    Exactly the rows the per-note `WHERE cards.note_id = ?` lookups used to
    return, fetched once: walking N notes cost N SELECTs (measured on this
    schema: 500 notes -> 501 statements for a single /learn). Joined to
    `notes` by the very predicate that selects those notes, so the grouping
    covers precisely them and nothing else.

    Deliberately no ORDER BY. Cards inside one note come back in the same
    (rowid) order the per-note SELECT gave them, and _ensure_inflection_cards
    pairs form-less orphan cards with form names in that order - sorting by
    Card.id here would reshuffle it, and a uuid4 order means nothing anyway.
    """
    stmt = select(Card).join(Note, Card.note_id == Note.id).where(Note.user_id == user_id)
    if deck_id is not None:
        stmt = stmt.where(Note.deck_id == deck_id)
    grouped: dict[str, list[Card]] = {}
    for card in await session.scalars(stmt):
        grouped.setdefault(card.note_id, []).append(card)
    return grouped


async def ensure_card_types(
    session: AsyncSession,
    note: Note,
    now: datetime,
    existing: Sequence[Card] | None = None,
) -> list[Card]:
    """Create any review-card types `note` has become eligible for.

    Safe to call repeatedly (e.g. after every review, or lazily before
    building a /learn queue) - each type is created at most once per note.

    `existing` is this note's cards, for a caller that already holds them -
    see sync_user_card_types, which fetches every note's in one query rather
    than one SELECT per note. Left out, the function reads them itself and
    behaves exactly as it always did; an empty sequence means "this note has
    no cards", not "go and look".
    """
    if existing is None:
        existing = (await session.scalars(select(Card).where(Card.note_id == note.id))).all()
    by_type = {card.type: card for card in existing}
    created: list[Card] = []

    if CardType.recognition not in by_type:
        card = Card(
            note_id=note.id,
            user_id=note.user_id,
            type=CardType.recognition,
            due=now,
            status=CardStatus.introduced,
            introduced_at=now,
        )
        session.add(card)
        created.append(card)
        by_type[CardType.recognition] = card

    recognition = by_type.get(CardType.recognition)
    if (
        CardType.production not in by_type
        and recognition is not None
        and (recognition.stability or 0.0) >= PRODUCTION_STABILITY_THRESHOLD_DAYS
    ):
        # One card per note, opened by a condition of its own - no need to
        # route it through the curriculum's form budget.
        card = Card(
            note_id=note.id,
            user_id=note.user_id,
            type=CardType.production,
            due=now,
            status=CardStatus.introduced,
            introduced_at=now,
        )
        session.add(card)
        created.append(card)

    if note.meta.get("forms_verified") is True:
        created.extend(_ensure_inflection_cards(session, note, existing, now))

    return created


def _ensure_inflection_cards(
    session: AsyncSession, note: Note, existing: Sequence[Card], now: datetime
) -> list[Card]:
    """One inflection card per quizzable principal form, created at most once each.

    Created `not_introduced`: existing is cheap, being shown is not. Which
    of them the learner actually meets, and when, is srs/curriculum.py's
    call - see CardStatus.
    """
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
            note_id=note.id,
            user_id=note.user_id,
            type=CardType.inflection,
            form=name,
            due=now,
            status=CardStatus.not_introduced,
        )
        session.add(card)
        created.append(card)
        covered.add(name)

    return created


async def resync_inflection_cards(
    session: AsyncSession, note: Note, now: datetime
) -> tuple[list[str], list[Card]]:
    """Match the note's inflection cards to the principal forms it has *now*.

    ensure_card_types only ever adds, which is right while a note's forms
    only ever get filled in. /edit breaks that assumption: a new lemma - or
    a lemma whose part of speech turned out to be another one
    (ingest.resolve_note_pos) - rewrites meta.principal_forms wholesale, and
    the cards keyed by the forms that are gone quiz nothing the note still
    has. learn.render_card falls back to (lemma, lemma) for them, so they
    are unanswerable, yet they keep their place in the queue, spend the
    daily form budget and hold up the curriculum's level gate.

    So they are deleted, with their reviews - and their reviews especially:
    queue.count_new_cards_today counts rows in `reviews` without joining
    `cards` (no ON DELETE CASCADE anywhere, see bot/add.py's delete_confirm),
    so leaving them behind would keep a deleted card eating today's new-card
    budget. Creation stays where it was: _ensure_inflection_cards, under the
    same forms_verified gate ensure_card_types applies.

    FSRS history is deliberately not carried over to the new forms - a
    card's stability describes the form it was rated on, not the slot it sat
    in. A form the note still has keeps its own card untouched, so the
    everyday lemma fix (puhua -> hakea: same POS, same form keys) changes
    nothing at all.

    Returns (deleted card ids, created cards). Does not commit.
    """
    existing = (
        await session.scalars(
            select(Card).where(Card.note_id == note.id, Card.type == CardType.inflection)
        )
    ).all()
    forms: dict = note.meta.get("principal_forms") or {}
    # form is None only on the legacy form-less cards _ensure_inflection_cards
    # adopts into a form below - they are tied to no form key, so no rewrite
    # of the form set can strand them.
    stale = [card for card in existing if card.form is not None and card.form not in forms]
    stale_ids = [card.id for card in stale]
    if stale_ids:
        await session.execute(delete(Review).where(Review.card_id.in_(stale_ids)))
        await session.execute(delete(Card).where(Card.id.in_(stale_ids)))

    if note.meta.get("forms_verified") is not True:
        return stale_ids, []
    kept = [card for card in existing if card.form is None or card.form in forms]
    return stale_ids, _ensure_inflection_cards(session, note, kept, now)


async def sync_user_card_types(session: AsyncSession, user_id: int, now: datetime) -> list[Card]:
    """Run ensure_card_types for every note the user owns. Does not commit.

    Two queries regardless of how many notes there are: the notes, then all
    of their cards at once (see cards_by_note). Each note is still handed
    only its own cards, so ensure_card_types decides exactly what it decided
    before - the loop just no longer pays a SELECT per iteration.
    """
    notes = (await session.scalars(select(Note).where(Note.user_id == user_id))).all()
    by_note = await cards_by_note(session, user_id)
    created: list[Card] = []
    for note in notes:
        created.extend(await ensure_card_types(session, note, now, by_note.get(note.id, ())))
    return created
