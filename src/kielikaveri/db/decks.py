"""Deck management (plan: "колоды", manual and user-named, not automatic).

A deck is purely organizational - it narrows /learn's queue and /add's save
target. It never changes FSRS scheduling itself (see srs/queue.py).

There is no "current" deck: /add and /learn both ask which one to use every
time, so nothing is remembered between turns.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kielikaveri.db.models import Deck

DEFAULT_DECK_NAME = "Общая"


async def list_decks(session: AsyncSession, user_id: int) -> list[Deck]:
    result = await session.scalars(
        select(Deck).where(Deck.user_id == user_id).order_by(Deck.created_at)
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
