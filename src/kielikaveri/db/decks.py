"""Deck management (plan: "колоды", manual and user-named, not automatic).

A deck is purely organizational - it narrows /learn's queue and /add's save
target. It never changes FSRS scheduling itself (see srs/queue.py).

There is no "current" deck: /add and /learn both ask which one to use every
time, so nothing is remembered between turns.
"""

from __future__ import annotations

from sqlalchemy import literal_column, select
from sqlalchemy.ext.asyncio import AsyncSession

from kielikaveri.db.models import Deck

DEFAULT_DECK_NAME = "Общая"


async def list_decks(session: AsyncSession, user_id: int) -> list[Deck]:
    """The user's decks in creation order, oldest first.

    `created_at` alone isn't enough: UTCDateTime stores whole epoch seconds,
    so decks made within the same second tie, and SQLite then returns them
    in whatever order the query plan happens to produce. The tie is broken
    by the table's rowid - its insertion order, i.e. the true creation order
    (the "Общая" auto-created right before the /add or /decks list stays
    ahead of a deck made the same second). Not `Deck.id`: a uuid4 order is
    deterministic but means nothing.

    get_or_create_default_deck() takes element [0] of this as "the user's
    first deck", and the /decks list, the /add deck picker and /learn's deck
    choice all show decks in this order.
    """
    result = await session.scalars(
        select(Deck)
        .where(Deck.user_id == user_id)
        .order_by(Deck.created_at, literal_column("decks.rowid"))
    )
    return list(result.all())


async def create_deck(session: AsyncSession, user_id: int, name: str) -> Deck:
    deck = Deck(user_id=user_id, name=name)
    session.add(deck)
    await session.flush()
    return deck


async def get_or_create_default_deck(session: AsyncSession, user_id: int) -> Deck:
    """The user's first deck, or a freshly created "Общая" if they have none."""
    decks = await list_decks(session, user_id)
    if decks:
        return decks[0]
    return await create_deck(session, user_id, DEFAULT_DECK_NAME)
