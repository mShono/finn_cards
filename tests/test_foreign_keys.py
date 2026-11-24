"""Plan М14: PRAGMA foreign_keys=ON on every connection (db/engine.py).

What it takes for the app to keep working with it: a whitelisted user's
`users` row is created on their first update (bot/middleware.py), /delete
removes reviews and cards before their note (bot/add.py), and deleting a
book leaves its chapter decks in place with chapter_id NULL.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.types import Update
from conftest import add_users, delete_book
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError

from kielikaveri.bot.add import delete_confirm
from kielikaveri.bot.middleware import DECLINE_TEXT, WhitelistMiddleware
from kielikaveri.db.decks import get_or_create_chapter_deck
from kielikaveri.db.engine import create_all, make_engine, make_session_factory
from kielikaveri.db.models import Book, BookChapter, Card, Deck, Note, NoteKind, Review, User

USER_ID = 1


@pytest.fixture(params=["memory", "file"])
async def session_factory(request, tmp_path):
    url = (
        "sqlite+aiosqlite://"
        if request.param == "memory"
        else f"sqlite+aiosqlite:///{tmp_path}/fk.db"
    )
    engine = make_engine(url)
    await create_all(engine)
    yield make_session_factory(engine)
    await engine.dispose()


def _note(note_id: str = "n1", deck_id: str | None = None) -> Note:
    return Note(
        id=note_id,
        user_id=USER_ID,
        lemma="naapuri",
        pos="substantiivi",
        translation_ru="сосед",
        example_fi="Naapuri on kotona.",
        example_ru="Сосед дома.",
        kind=NoteKind.word,
        deck_id=deck_id,
        meta={},
    )


async def _count(session_factory, model) -> int:
    async with session_factory() as session:
        return await session.scalar(select(func.count()).select_from(model))


async def test_every_connection_has_foreign_keys_on(session_factory):
    async with session_factory() as session:
        assert await session.scalar(text("PRAGMA foreign_keys")) == 1


async def test_card_without_its_note_is_rejected(session_factory):
    await add_users(session_factory, USER_ID)
    async with session_factory() as session:
        session.add(Card(id="c1", note_id="missing", user_id=USER_ID, type="recognition"))
        with pytest.raises(IntegrityError, match="FOREIGN KEY"):
            await session.commit()


async def test_deck_without_its_user_is_rejected(session_factory):
    async with session_factory() as session:
        session.add(Deck(user_id=USER_ID, name="Общая"))
        with pytest.raises(IntegrityError, match="FOREIGN KEY"):
            await session.commit()


async def test_deleting_a_book_keeps_its_deck_and_cards_and_nulls_chapter_id(session_factory):
    await add_users(session_factory, USER_ID)
    async with session_factory() as session:
        book = Book(
            user_id=USER_ID, title="Kirja", source_lang="ru", source_hash="h", front_matter=[]
        )
        session.add(book)
        await session.flush()
        chapter = BookChapter(book_id=book.id, number=1, title="Глава 1", source_paragraphs=[])
        session.add(chapter)
        await session.flush()
        deck = await get_or_create_chapter_deck(session, USER_ID, chapter, "Kirja - гл.1")
        session.add(_note(deck_id=deck.id))
        await session.flush()
        session.add(Card(id="c1", note_id="n1", user_id=USER_ID, type="recognition"))
        await session.commit()
        book_id, deck_id = book.id, deck.id

    async with session_factory() as session:
        await delete_book(session, book_id)
        await session.commit()

    async with session_factory() as session:
        deck = await session.get(Deck, deck_id)
        assert deck is not None
        assert deck.chapter_id is None
        assert (await session.get(Note, "n1")).deck_id == deck_id
        assert await session.get(Card, "c1") is not None
        assert await session.get(Book, book_id) is None


def _message_update() -> Update:
    return Update.model_construct(update_id=1, message=SimpleNamespace(answer=AsyncMock()))


async def test_first_update_from_a_whitelisted_user_creates_the_users_row(session_factory):
    middleware = WhitelistMiddleware({USER_ID})
    handler = AsyncMock(return_value="handled")

    for _ in range(2):
        data = {"event_from_user": SimpleNamespace(id=USER_ID), "session_factory": session_factory}
        assert await middleware(handler, _message_update(), data) == "handled"

    assert handler.await_count == 2
    async with session_factory() as session:
        assert (await session.scalars(select(User.id))).all() == [USER_ID]
        # And now the rows that reference it can be saved.
        session.add(Deck(user_id=USER_ID, name="Общая"))
        await session.commit()


async def test_users_row_already_there_is_left_alone(session_factory):
    # Created by import_cards.py, or by another process the cache knows nothing of.
    await add_users(session_factory, USER_ID)
    middleware = WhitelistMiddleware({USER_ID})
    handler = AsyncMock(return_value="handled")
    data = {"event_from_user": SimpleNamespace(id=USER_ID), "session_factory": session_factory}

    assert await middleware(handler, _message_update(), data) == "handled"
    assert await _count(session_factory, User) == 1


async def test_update_from_a_stranger_creates_no_users_row(session_factory):
    middleware = WhitelistMiddleware({USER_ID})
    handler = AsyncMock()
    update = _message_update()
    data = {"event_from_user": SimpleNamespace(id=999), "session_factory": session_factory}

    assert await middleware(handler, update, data) is None

    handler.assert_not_awaited()
    update.message.answer.assert_awaited_once_with(DECLINE_TEXT)
    assert await _count(session_factory, User) == 0


async def test_delete_of_a_word_with_cards_and_reviews_then_again(session_factory):
    await add_users(session_factory, USER_ID)
    async with session_factory() as session:
        deck = Deck(user_id=USER_ID, name="Общая")
        session.add(deck)
        await session.flush()
        session.add(_note(deck_id=deck.id))
        await session.flush()
        session.add_all(
            [
                Card(id="c1", note_id="n1", user_id=USER_ID, type="recognition"),
                Card(id="c2", note_id="n1", user_id=USER_ID, type="production"),
            ]
        )
        await session.flush()
        session.add_all(
            [
                Review(card_id="c1", user_id=USER_ID, rating=3),
                Review(card_id="c2", user_id=USER_ID, rating=1),
            ]
        )
        await session.commit()

    def callback() -> SimpleNamespace:
        return SimpleNamespace(
            data="delnote:n1",
            from_user=SimpleNamespace(id=USER_ID),
            answer=AsyncMock(),
            message=SimpleNamespace(answer=AsyncMock()),
        )

    first = callback()
    await delete_confirm(first, session_factory)
    report = first.message.answer.call_args.args[0]
    assert "Удалила «naapuri» из колоды «Общая»" in report

    for model in (Note, Card, Review):
        assert await _count(session_factory, model) == 0
    assert await _count(session_factory, Deck) == 1

    second = callback()
    await delete_confirm(second, session_factory)
    second.answer.assert_awaited_once_with("Уже удалено.", show_alert=True)
    second.message.answer.assert_not_called()
