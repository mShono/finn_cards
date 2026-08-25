"""Builds a /learn session's queue: due cards, capped by the daily new-card
limit and the debt (backlog) threshold. DB-facing, no Telegram here.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from kielikaveri.db.models import Card, Review

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


async def due_cards(
    session: AsyncSession, user_id: int, now: datetime, limit: int | None
) -> list[Card]:
    result = await session.scalars(
        select(Card).where(Card.user_id == user_id, Card.due <= now).order_by(Card.due).limit(limit)
    )
    return list(result.all())


async def overdue_count(session: AsyncSession, user_id: int, now: datetime) -> int:
    return await session.scalar(
        select(func.count()).select_from(Card).where(Card.user_id == user_id, Card.due <= now)
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
) -> list[str]:
    """Card ids for one /learn session, oldest-due first.

    Caps total size at `session_max_cards`, and caps how many never-reviewed
    (reps == 0) cards it admits at whatever's left of `daily_new_limit` for
    today's study window - review cards are never held back by this limit.
    """
    new_today = await count_new_cards_today(session, user_id, now, boundary_hour)
    new_budget = max(0, daily_new_limit - new_today)

    # Fetch generously past session_max_cards - some candidates may be
    # skipped for being "new" past the daily budget, so a tight limit here
    # could starve the queue with review cards still due.
    candidates = await due_cards(session, user_id, now, limit=session_max_cards * 5)

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
    return queue


async def defer_overdue_tail(
    session: AsyncSession,
    user_id: int,
    now: datetime,
    keep_n: int,
    postpone_days: int,
) -> int:
    """Push every overdue card past the first `keep_n` (oldest-due) forward
    by `postpone_days`. Returns how many cards were postponed."""
    cards = await due_cards(session, user_id, now, limit=None)
    tail = cards[keep_n:]
    for card in tail:
        card.due = now + timedelta(days=postpone_days)
    return len(tail)
