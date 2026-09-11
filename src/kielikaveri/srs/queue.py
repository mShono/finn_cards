"""Builds a /learn session's queue: due cards, capped by the daily new-card
limit and the debt (backlog) threshold. DB-facing, no Telegram here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from kielikaveri.db.models import Card, CardStatus, CardType, Note, Review
from kielikaveri.grammar import FORM_ORDER

logger = logging.getLogger(__name__)

# The study day boundary is defined in Europe/Helsinki regardless of where
# the server runs (see plan 3.10) - it's the learner's day that matters, not
# the VPS's.
STUDY_TIMEZONE = ZoneInfo("Europe/Helsinki")


def study_day_bounds(now: datetime, boundary_hour: int) -> tuple[datetime, datetime]:
    """Return [start, end) of the study day `now` falls in, as UTC datetimes."""
    local_now = now.astimezone(STUDY_TIMEZONE)
    boundary_today = local_now.replace(hour=boundary_hour, minute=0, second=0, microsecond=0)
    start = boundary_today if local_now >= boundary_today else boundary_today - timedelta(days=1)
    end = start + timedelta(days=1)
    return start.astimezone(UTC), end.astimezone(UTC)


# A card's place inside its own note: the word itself before any of its
# forms, and the forms in FORM_TASKS order. This is the order
# srs/curriculum.py opens forms in (grammar.FORM_ORDER, notes outer and
# forms inner) - the queue repeats it rather than inventing one, so a
# group like mihin?/missa?/mista? reaches the learner back to back.
_CARD_TYPE_ORDER: dict[CardType, int] = {
    CardType.recognition: 0,
    CardType.production: 1,
    CardType.inflection: 2,
    CardType.usage: 3,
}


def _syllabus_order() -> tuple:
    """SQL ORDER BY terms placing a card in its note's syllabus.

    Without them the only ordering term is `due`, and a whole batch of
    forms opened on one study day carries the *same* due second (see
    curriculum.introduce_due_forms, plus UTCDateTime's epoch-second
    storage). Ties in ORDER BY are not ordered at all: SQLite happened to
    return them in insert order, which is the key order of the note's
    `principal_forms` JSON - so a note whose forms arrived reversed handed
    the learner monikon_genetiivi before genetiivi. Deliberately not
    Card.id: it is a uuid4, deterministic but meaningless.
    """
    return (
        case(_CARD_TYPE_ORDER, value=Card.type, else_=len(_CARD_TYPE_ORDER)),
        case(FORM_ORDER, value=Card.form, else_=len(FORM_ORDER)),
        Card.id,  # last resort, so the order is total and never plan-dependent
    )


def _due_order() -> tuple:
    """The full ordering contract for a session's candidates: oldest due
    first, then note by note in the order they were added, then each note's
    syllabus.
    """
    return (Card.due, Note.created_at, Note.id, *_syllabus_order())


async def due_cards(
    session: AsyncSession,
    user_id: int,
    now: datetime,
    limit: int | None,
    deck_id: str | None = None,
    reviewed_only: bool = False,
) -> list[Card]:
    # Note is joined unconditionally, not just to filter by deck: the
    # ordering below needs its columns, and a card always has a note.
    stmt = (
        select(Card)
        .join(Note, Card.note_id == Note.id)
        .where(Card.user_id == user_id, Card.due <= now, Card.status == CardStatus.introduced)
    )
    if reviewed_only:
        stmt = stmt.where(Card.reps > 0)
    if deck_id is not None:
        stmt = stmt.where(Note.deck_id == deck_id)
    result = await session.scalars(stmt.order_by(*_due_order()).limit(limit))
    return list(result.all())


async def overdue_count(
    session: AsyncSession,
    user_id: int,
    now: datetime,
    deck_id: str | None = None,
    reviewed_only: bool = False,
) -> int:
    """How many of the user's cards are due.

    `reviewed_only` is what the debt (backlog) prompt asks for: debt means
    reviews you owe, and a card you have never seen isn't late - it is
    waiting its turn under the daily new-card limit. Since one note now
    opens an inflection card per form, counting those as debt would fire
    the backlog prompt on day one. The deck screen leaves it off: there
    "к повторению" does mean everything due, new cards included.
    """
    stmt = (
        select(func.count())
        .select_from(Card)
        .where(Card.user_id == user_id, Card.due <= now, Card.status == CardStatus.introduced)
    )
    if reviewed_only:
        stmt = stmt.where(Card.reps > 0)
    if deck_id is not None:
        stmt = stmt.join(Note, Card.note_id == Note.id).where(Note.deck_id == deck_id)
    return await session.scalar(stmt)


@dataclass(frozen=True)
class CardCounters:
    """What a deck actually contains, kept as separate numbers on purpose.

    `total` is not a workload: most of it can be forms the curriculum has
    not opened yet. Only `due` asks anything of the learner today, and only
    `overdue` is debt.
    """

    total: int
    introduced: int
    due: int
    overdue: int
    not_introduced: int


async def card_counters(
    session: AsyncSession, user_id: int, now: datetime, deck_id: str | None = None
) -> CardCounters:
    def count(*conditions) -> object:
        stmt = select(func.count()).select_from(Card).where(Card.user_id == user_id, *conditions)
        if deck_id is not None:
            stmt = stmt.join(Note, Card.note_id == Note.id).where(Note.deck_id == deck_id)
        return stmt

    introduced = Card.status == CardStatus.introduced
    return CardCounters(
        total=await session.scalar(count()),
        introduced=await session.scalar(count(introduced)),
        due=await session.scalar(count(introduced, Card.due <= now)),
        # Debt: a review that came due and was missed. A card that has never
        # been shown is not late, it simply has not come up yet.
        overdue=await session.scalar(count(introduced, Card.due <= now, Card.reps > 0)),
        not_introduced=await session.scalar(count(Card.status == CardStatus.not_introduced)),
    )


async def count_new_cards_today(
    session: AsyncSession, user_id: int, now: datetime, boundary_hour: int
) -> int:
    """How many cards had their first-ever review inside today's study window."""
    start, end = study_day_bounds(now, boundary_hour)
    first_review_at = (
        select(Review.card_id, func.min(Review.reviewed_at).label("first_at"))
        .where(Review.user_id == user_id)
        .group_by(Review.card_id)
        .subquery()
    )
    return await session.scalar(
        select(func.count())
        .select_from(first_review_at)
        .where(first_review_at.c.first_at >= start, first_review_at.c.first_at < end)
    )


async def build_session_queue(
    session: AsyncSession,
    user_id: int,
    now: datetime,
    session_max_cards: int,
    daily_new_limit: int,
    boundary_hour: int,
    deck_id: str | None = None,
) -> list[str]:
    """Card ids for one /learn session, in the order the learner meets them.

    The order is the contract, not an accident of the query plan: oldest due
    first, then - among cards sharing a due second, which every batch the
    curriculum opens on one day does - note by note in the order they were
    added, and inside a note the word before its forms and the forms in
    FORM_TASKS order. See _due_order().

    Only `introduced` cards are candidates: a form the curriculum has not
    opened yet has no due date worth honouring, and a suspended one is
    parked. Opening them is srs/curriculum.py's job, run before this.

    Caps total size at `session_max_cards`, and caps how many never-reviewed
    (reps == 0) cards it admits at whatever's left of `daily_new_limit` for
    today's study window - review cards are never held back by this limit.
    That limit is the defensive one, counted over cards of every type;
    `daily_new_forms` separately governs how fast new grammar opens.
    `deck_id` narrows candidates to one deck; the daily new-card budget stays
    global across decks on purpose - it's a "don't overload the learner today"
    cap, not a per-deck one.
    """
    new_today = await count_new_cards_today(session, user_id, now, boundary_hour)
    new_budget = max(0, daily_new_limit - new_today)

    # Fetch generously past session_max_cards - some candidates may be
    # skipped for being "new" past the daily budget, so a tight limit here
    # could starve the queue with review cards still due.
    candidates = await due_cards(
        session, user_id, now, limit=session_max_cards * 5, deck_id=deck_id
    )

    queue: list[str] = []
    new_used = 0
    for card in candidates:
        if card.reps == 0:
            if new_used >= new_budget:
                continue
            new_used += 1
        queue.append(card.id)
        if len(queue) >= session_max_cards:
            break

    logger.debug(
        "event=learn.queue_built candidates=%d queue=%d new_used=%d new_budget=%d",
        len(candidates),
        len(queue),
        new_used,
        new_budget,
    )
    return queue


async def defer_overdue_tail(
    session: AsyncSession,
    user_id: int,
    now: datetime,
    keep_n: int,
    postpone_days: int,
    deck_id: str | None = None,
) -> int:
    """Push every overdue card past the first `keep_n` (oldest-due) forward
    by `postpone_days`. Returns how many cards were postponed.

    Only cards that have been reviewed, matching the count the debt prompt
    showed - postponing a never-seen card would delay work that was never
    late in the first place.
    """
    cards = await due_cards(session, user_id, now, limit=None, deck_id=deck_id, reviewed_only=True)
    tail = cards[keep_n:]
    for card in tail:
        card.due = now + timedelta(days=postpone_days)
    logger.debug(
        "event=learn.debt_deferred overdue=%d kept=%d deferred=%d", len(cards), keep_n, len(tail)
    )
    return len(tail)
