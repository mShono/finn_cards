"""Deck management (plan: "колоды", manual and user-named, not automatic).

A deck is purely organizational - it narrows /learn's queue and /add's save
target. It never changes FSRS scheduling itself (see srs/queue.py).
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kielikaveri.db.models import Deck, User

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


async def active_deck(session: AsyncSession, user_id: int) -> Deck:
    """The deck new notes are saved to right now: the last one the user
    picked (see set_active_deck), else their first deck, else a fresh
    default. /add must never block a save on "pick a deck first".
    """
    user = await session.get(User, user_id)
    if user is not None and user.last_deck_id is not None:
        deck = await session.get(Deck, user.last_deck_id)
        if deck is not None:
            return deck
    return await get_or_create_default_deck(session, user_id)


async def set_active_deck(session: AsyncSession, user_id: int, deck_id: str) -> None:
    # Telegram users have no guaranteed `users` row (see import_cards.py -
    # only the CLI importer creates one; the live bot never has, and SQLite's
    # FK isn't enforced here) - create it on first deck pick rather than
    # assuming it exists.
    user = await session.get(User, user_id)
    if user is None:
        user = User(id=user_id)
        session.add(user)
    user.last_deck_id = deck_id
