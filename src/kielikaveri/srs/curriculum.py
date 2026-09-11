"""Which form the learner meets next - the gate between "card exists" and
"card is being learned".

A noun opens twelve inflection cards the moment its forms are verified.
Creating them is free; showing all twelve is not. This module picks the few
that become `introduced` on a given study day, and nothing else in the
system decides that:

    Card exists (not_introduced)
        -> curriculum picks it        introduce_due_forms()
    Card introduced (learning)
        -> py-fsrs owns it from here  srs.scheduler

It deliberately schedules nothing. Once a card is introduced, its intervals
come from FSRS alone, exactly as before - see srs/scheduler.py.

Policy, all of it data-driven from kielikaveri.grammar:
1. A form only opens once the word itself is known - the note's recognition
   card must have been answered correctly SUCCESSFUL_ANSWERS_TO_UNLOCK times.
2. Forms open by curriculum level: every core form of a note must be known
   before an extended one opens, and every extended one before a `later`
   one. Same "answered right twice" bar, so the ladder is driven by what the
   learner actually got right rather than by how long ago the word was added.
3. Ties break by FORM_ORDER, which keeps a grammatical group (mihin? /
   missä? / mistä?) consecutive instead of scattered across weeks.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from kielikaveri.db.models import Card, CardStatus, CardType, Note, Review
from kielikaveri.grammar import FORM_ORDER, FORM_TASKS, LEVEL_ORDER
from kielikaveri.srs.queue import study_day_bounds
from kielikaveri.srs.scheduler import Rating

logger = logging.getLogger(__name__)

# Correct answers a card needs before the curriculum will build on it.
# Deliberately a count of answers and not an FSRS interval: what a word has
# earned in days depends on which ratings it happened to get and on FSRS's
# own parameters, while "I have recalled this twice" is the same short
# acquaintance for every word. Two, not one, because the first correct
# answer to a brand new word is mostly the echo of having just read it.
SUCCESSFUL_ANSWERS_TO_UNLOCK = 2


def _is_known(card: Card, successes: Mapping[str, int]) -> bool:
    return card.status == CardStatus.introduced and (
        successes.get(card.id, 0) >= SUCCESSFUL_ANSWERS_TO_UNLOCK
    )


def _level_index(form: str) -> int:
    return LEVEL_ORDER.index(FORM_TASKS[form].level)


def eligible_forms(cards: list[Card], successes: Mapping[str, int]) -> list[Card]:
    """The not_introduced inflection cards of one note that may open now.

    `successes` counts correct answers per card id - see
    successful_answer_counts(). Empty while the word itself is still new, or
    while any lower level of this note has a form the learner hasn't
    answered right yet.
    """
    recognition = next((c for c in cards if c.type == CardType.recognition), None)
    if recognition is None or not _is_known(recognition, successes):
        return []

    inflection = [c for c in cards if c.type == CardType.inflection and c.form in FORM_TASKS]
    candidates = [c for c in inflection if c.status == CardStatus.not_introduced]
    if not candidates:
        return []

    # The lowest level that still has anything to open is the only one that
    # may: everything below it must be known first.
    open_level = min(_level_index(c.form) for c in candidates)
    unfinished_below = [
        c for c in inflection if _level_index(c.form) < open_level and not _is_known(c, successes)
    ]
    if unfinished_below:
        return []

    return sorted(
        (c for c in candidates if _level_index(c.form) == open_level),
        key=lambda c: FORM_ORDER[c.form],
    )


async def successful_answer_counts(session: AsyncSession, user_id: int) -> dict[str, int]:
    """How many times each of the user's cards has been answered correctly.

    Read straight off the review log, so the curriculum needs no column of
    its own and every review already recorded counts. `Again` is the one
    rating that isn't a correct answer, so a lapse adds nothing to the
    count - it doesn't subtract either: the bar is "has been recalled
    twice", not "has never been forgotten".
    """
    rows = await session.execute(
        select(Review.card_id, func.count())
        .where(Review.user_id == user_id, Review.rating > Rating.Again.value)
        .group_by(Review.card_id)
    )
    return {card_id: count for card_id, count in rows.all()}


async def count_introduced_today(
    session: AsyncSession, user_id: int, now: datetime, boundary_hour: int
) -> int:
    """Inflection cards the curriculum opened inside today's study window."""
    start, end = study_day_bounds(now, boundary_hour)
    return await session.scalar(
        select(func.count())
        .select_from(Card)
        .where(
            Card.user_id == user_id,
            Card.type == CardType.inflection,
            Card.introduced_at >= start,
            Card.introduced_at < end,
        )
    )


async def introduce_due_forms(
    session: AsyncSession,
    user_id: int,
    now: datetime,
    daily_new_forms: int,
    boundary_hour: int,
    deck_id: str | None = None,
) -> list[Card]:
    """Open at most today's remaining form budget. Does not commit.

    This is a separate budget from `daily_new_limit`: that one caps how many
    never-reviewed cards a *session* admits (all types, the defensive limit),
    while this one caps how fast new grammar opens at all. A word and a case
    are not the same unit of new work.
    """
    budget = daily_new_forms - await count_introduced_today(session, user_id, now, boundary_hour)
    if budget <= 0:
        return []

    successes = await successful_answer_counts(session, user_id)

    stmt = select(Note).where(Note.user_id == user_id)
    if deck_id is not None:
        stmt = stmt.where(Note.deck_id == deck_id)
    notes = (await session.scalars(stmt.order_by(Note.created_at, Note.id))).all()

    introduced: list[Card] = []
    for note in notes:
        if len(introduced) >= budget:
            break
        cards = list((await session.scalars(select(Card).where(Card.note_id == note.id))).all())
        for card in eligible_forms(cards, successes):
            if len(introduced) >= budget:
                break
            card.status = CardStatus.introduced
            card.introduced_at = now
            card.due = now
            introduced.append(card)

    if introduced:
        logger.info(
            "event=curriculum.introduced count=%d budget=%d forms=%s",
            len(introduced),
            budget,
            [c.form for c in introduced],
        )
    return introduced
