"""Deck management (plan: "колоды", manual and user-named, not automatic).

A deck is purely organizational - it narrows learning's queue and /add's
save target. It never changes FSRS scheduling itself (see srs/queue.py).

There is no "current" deck: /add and learning in the reader both ask which
one to use every time, so nothing is remembered between turns.
"""

from __future__ import annotations

from sqlalchemy import literal_column, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from kielikaveri.db.models import BookChapter, Deck

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
    first deck", and the /decks list, the /add deck picker and the reader's
    learning deck choice all show decks in this order.
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


def _is_chapter_deck_duplicate_error(error: IntegrityError) -> bool:
    """True only for a violation of the unique index on decks.chapter_id.

    For a plain-column unique index SQLite reports the column, not the index
    name ("UNIQUE constraint failed: decks.chapter_id"). Matched against the
    driver's own message (error.orig), not str(error), which would also echo
    the statement and its parameters.
    """
    return "decks.chapter_id" in str(error.orig)


async def _find_chapter_deck(session: AsyncSession, chapter_id: str) -> Deck | None:
    return await session.scalar(select(Deck).where(Deck.chapter_id == chapter_id))


async def get_or_create_chapter_deck(
    session: AsyncSession, user_id: int, chapter: BookChapter, name: str
) -> Deck:
    """The deck of a reader chapter, created on first use (plan И7).

    The deck is looked up by `chapter_id`, never by `name`: deck names are not
    unique (a user may have several decks with the same name) and a book may
    be renamed later, so `name` ("<book> - гл.N") is only the label given to a
    newly created deck - an existing one is returned as is.

    Race-free: aiohttp handlers run concurrently, so two fast taps on a new
    chapter can both find no deck. The unique index on decks.chapter_id lets
    only one INSERT win; the loser's INSERT runs inside a savepoint, which is
    rolled back on the IntegrityError (leaving the caller's transaction
    intact), and the winner's deck is selected again. Any other
    IntegrityError is re-raised.
    """
    deck = await _find_chapter_deck(session, chapter.id)
    if deck is not None:
        return deck
    try:
        async with session.begin_nested():
            deck = Deck(user_id=user_id, name=name, chapter_id=chapter.id)
            session.add(deck)
    except IntegrityError as error:
        if not _is_chapter_deck_duplicate_error(error):
            raise
        deck = await _find_chapter_deck(session, chapter.id)
        if deck is None:
            raise
    return deck
