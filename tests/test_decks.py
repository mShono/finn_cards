from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage

from kielikaveri.bot.decks import DeckStates, decks_activate, decks_list, decks_new_save
from kielikaveri.db.decks import (
    active_deck,
    create_deck,
    get_or_create_default_deck,
    list_decks,
    set_active_deck,
)
from kielikaveri.db.engine import create_all, make_engine, make_session_factory
from kielikaveri.db.models import Card, CardType, Note, NoteKind, User

NOW = datetime(2026, 8, 26, 10, 0, tzinfo=UTC)


@pytest.fixture
async def session_factory(tmp_path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    await create_all(engine)
    yield make_session_factory(engine)
    await engine.dispose()


def make_state() -> FSMContext:
    return FSMContext(storage=MemoryStorage(), key=StorageKey(bot_id=0, chat_id=1, user_id=1))


def make_message(text: str = "") -> SimpleNamespace:
    return SimpleNamespace(text=text, from_user=SimpleNamespace(id=1), answer=AsyncMock())


def make_callback(data: str) -> SimpleNamespace:
    return SimpleNamespace(
        data=data,
        from_user=SimpleNamespace(id=1),
        answer=AsyncMock(),
        message=SimpleNamespace(answer=AsyncMock()),
    )


# --- db.decks ------------------------------------------------------------------------


async def test_get_or_create_default_deck_creates_one_named_общая_when_none_exist(session_factory):
    async with session_factory() as session:
        deck = await get_or_create_default_deck(session, 1)
        await session.commit()

    assert deck.name == "Общая"
    async with session_factory() as session:
        decks = await list_decks(session, 1)
    assert [d.id for d in decks] == [deck.id]


async def test_get_or_create_default_deck_returns_the_existing_first_deck(session_factory):
    async with session_factory() as session:
        deck = await create_deck(session, 1, "Первая")
        await session.commit()
        first_id = deck.id

    async with session_factory() as session:
        again = await get_or_create_default_deck(session, 1)
    assert again.id == first_id


async def test_active_deck_falls_back_to_default_when_no_user_row_exists(session_factory):
    # The live bot has no guaranteed `users` row for a Telegram user (see
    # db/decks.py's set_active_deck comment) - active_deck must not crash.
    async with session_factory() as session:
        deck = await active_deck(session, 1)
        await session.commit()
    assert deck.name == "Общая"


async def test_set_active_deck_creates_a_missing_user_row(session_factory):
    async with session_factory() as session:
        deck = await create_deck(session, 1, "Из текста")
        await set_active_deck(session, 1, deck.id)
        await session.commit()

    async with session_factory() as session:
        current = await active_deck(session, 1)
        user = await session.get(User, 1)
    assert current.id == deck.id
    assert user is not None


async def test_active_deck_prefers_the_last_picked_deck_over_the_first(session_factory):
    async with session_factory() as session:
        deck_a = await create_deck(session, 1, "Общая")
        deck_b = await create_deck(session, 1, "Из книги")
        await set_active_deck(session, 1, deck_b.id)
        await session.commit()

    async with session_factory() as session:
        current = await active_deck(session, 1)
    assert current.id == deck_b.id
    assert current.id != deck_a.id


async def test_active_deck_ignores_a_deleted_last_deck_id(session_factory):
    async with session_factory() as session:
        deck_a = await create_deck(session, 1, "Общая")
        session.add(User(id=1, last_deck_id="does-not-exist"))
        await session.commit()

    async with session_factory() as session:
        current = await active_deck(session, 1)
    assert current.id == deck_a.id


# --- bot.decks -------------------------------------------------------------------


async def test_decks_list_marks_the_active_deck_and_reports_counts(session_factory):
    async with session_factory() as session:
        session.add(User(id=1))
        deck = await create_deck(session, 1, "Общая")
        await session.flush()
        session.add(
            Note(
                id="n1",
                user_id=1,
                lemma="hakea",
                translation_ru="искать",
                example_fi="x",
                example_ru="y",
                kind=NoteKind.word,
                deck_id=deck.id,
                meta={},
            )
        )
        await session.flush()
        session.add(
            Card(
                id="c1",
                note_id="n1",
                user_id=1,
                type=CardType.recognition,
                due=NOW - timedelta(days=1),
            )
        )
        await session.commit()

    message = make_message()
    await decks_list(message, session_factory)

    text = message.answer.call_args.args[0]
    assert "Общая" in text
    assert "слов: 1" in text
    assert "к повторению: 1" in text


async def test_decks_list_shows_the_default_deck_it_just_created_for_a_fresh_user(session_factory):
    # Regression: active_deck() can create a "Общая" deck as a side effect -
    # decks_list must show that same deck, not the empty list_decks() result
    # from before the creation.
    message = make_message()

    await decks_list(message, session_factory)

    text = message.answer.call_args.args[0]
    assert "Колод пока нет" not in text
    assert "Общая" in text
    buttons = [
        b for row in message.answer.call_args.kwargs["reply_markup"].inline_keyboard for b in row
    ]
    assert any("Новая колода" in b.text for b in buttons)


async def test_decks_new_save_creates_a_deck_and_makes_it_active(session_factory):
    state = make_state()
    await state.set_state(DeckStates.naming)
    message = make_message("Из книги")

    await decks_new_save(message, state, session_factory)

    async with session_factory() as session:
        decks = await list_decks(session, 1)
        current = await active_deck(session, 1)
    assert [d.name for d in decks] == ["Из книги"]
    assert current.name == "Из книги"
    assert await state.get_state() is None
    assert "Из книги" in message.answer.call_args.args[0]


async def test_decks_new_save_reprompts_on_empty_name(session_factory):
    state = make_state()
    await state.set_state(DeckStates.naming)
    message = make_message("   ")

    await decks_new_save(message, state, session_factory)

    async with session_factory() as session:
        decks = await list_decks(session, 1)
    assert decks == []
    assert await state.get_state() == DeckStates.naming


async def test_decks_activate_switches_the_active_deck(session_factory):
    async with session_factory() as session:
        await create_deck(session, 1, "Общая")
        deck_b = await create_deck(session, 1, "Из книги")
        await session.commit()

    callback = make_callback(f"decks:activate:{deck_b.id}")
    await decks_activate(callback, session_factory)

    async with session_factory() as session:
        current = await active_deck(session, 1)
    assert current.id == deck_b.id
    callback.answer.assert_awaited_once()
