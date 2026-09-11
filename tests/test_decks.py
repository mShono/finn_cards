from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import InlineKeyboardButton

from kielikaveri.bot.decks import (
    NOTES_PER_PAGE,
    DeckStates,
    decks_activate,
    decks_list,
    decks_new_save,
    decks_open,
)
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


async def test_decks_keyboard_offers_open_and_activate_for_every_deck(session_factory):
    async with session_factory() as session:
        await create_deck(session, 1, "Общая")
        await create_deck(session, 1, "Из книги")
        await session.commit()

    message = make_message()
    await decks_list(message, session_factory)

    rows = message.answer.call_args.kwargs["reply_markup"].inline_keyboard
    open_labels = {b.text for row in rows for b in row if b.callback_data.startswith("decks:open:")}
    activate_rows = [row for row in rows if any("decks:activate:" in b.callback_data for b in row)]
    assert open_labels == {"📂 Общая", "📂 Из книги"}
    # The active deck ("Общая", created first) has no activate button - only
    # the non-active one does.
    assert len(activate_rows) == 1


async def test_decks_open_puts_each_note_on_an_edit_button_with_its_translation(session_factory):
    async with session_factory() as session:
        deck = await create_deck(session, 1, "Общая")
        await session.flush()
        session.add(
            Note(
                id="n1",
                user_id=1,
                lemma="hakea",
                pos="verbi",
                translation_ru="искать",
                example_fi="x",
                example_ru="y",
                kind=NoteKind.word,
                deck_id=deck.id,
                meta={},
            )
        )
        await session.commit()

    callback = make_callback(f"decks:open:{deck.id}")
    await decks_open(callback, session_factory)

    text = callback.message.answer.call_args.args[0]
    assert "слов: 1" in text
    assert "hakea" not in text
    keyboard = callback.message.answer.call_args.kwargs["reply_markup"]
    edit_buttons = [
        b for row in keyboard.inline_keyboard for b in row if b.callback_data == "noteedit:n1"
    ]
    assert len(edit_buttons) == 1
    assert edit_buttons[0].text == "✍️ 1. hakea - искать"
    assert "verbi" not in edit_buttons[0].text
    callback.answer.assert_awaited_once()


async def _fill_deck(session_factory, deck_id: str, count: int) -> None:
    async with session_factory() as session:
        for n in range(count):
            session.add(
                Note(
                    id=f"n{n}",
                    user_id=1,
                    lemma=f"sana{n}",
                    translation_ru=f"слово {n}",
                    example_fi="x",
                    example_ru="y",
                    kind=NoteKind.word,
                    deck_id=deck_id,
                    created_at=NOW + timedelta(minutes=n),
                    meta={},
                )
            )
        await session.commit()


def _note_buttons(callback):
    keyboard = callback.message.answer.call_args.kwargs["reply_markup"]
    return [
        b
        for row in keyboard.inline_keyboard
        for b in row
        if (b.callback_data or "").startswith("noteedit:")
    ]


async def test_decks_open_pages_a_deck_larger_than_one_screen_newest_first(session_factory):
    # Regression: a deck over NOTES_PER_PAGE used to cut off the extra notes
    # with no way to reach them - and it cut the newest ones, the very words
    # you come back to fix.
    async with session_factory() as session:
        deck = await create_deck(session, 1, "Большая")
        await session.commit()
    await _fill_deck(session_factory, deck.id, NOTES_PER_PAGE + 3)

    first = make_callback(f"decks:open:{deck.id}")
    await decks_open(first, session_factory)

    buttons = _note_buttons(first)
    assert len(buttons) == NOTES_PER_PAGE
    assert buttons[0].text.startswith(f"✍️ 1. sana{NOTES_PER_PAGE + 2}")
    assert "страница 1 из 2" in first.message.answer.call_args.args[0]
    assert first.message.answer.call_args.kwargs["reply_markup"].inline_keyboard[-2] == [
        InlineKeyboardButton(text="вперёд ➡️", callback_data=f"decks:open:{deck.id}:1")
    ]

    second = make_callback(f"decks:open:{deck.id}:1")
    await decks_open(second, session_factory)

    tail = _note_buttons(second)
    assert [b.callback_data for b in tail] == ["noteedit:n2", "noteedit:n1", "noteedit:n0"]
    assert tail[0].text.startswith(f"✍️ {NOTES_PER_PAGE + 1}. sana2")
    assert "страница 2 из 2" in second.message.answer.call_args.args[0]
    assert second.message.answer.call_args.kwargs["reply_markup"].inline_keyboard[-2] == [
        InlineKeyboardButton(text="⬅️ назад", callback_data=f"decks:open:{deck.id}:0")
    ]


async def test_decks_open_clamps_a_page_past_the_end(session_factory):
    async with session_factory() as session:
        deck = await create_deck(session, 1, "Малая")
        await session.commit()
    await _fill_deck(session_factory, deck.id, 2)

    callback = make_callback(f"decks:open:{deck.id}:7")
    await decks_open(callback, session_factory)

    assert len(_note_buttons(callback)) == 2
    assert "страница" not in callback.message.answer.call_args.args[0]


async def test_decks_open_rejects_a_deck_belonging_to_another_user(session_factory):
    async with session_factory() as session:
        deck = await create_deck(session, 2, "Чужая")
        await session.commit()

    callback = make_callback(f"decks:open:{deck.id}")
    await decks_open(callback, session_factory)

    callback.answer.assert_awaited_once_with("Не нашла колоду.", show_alert=True)
    callback.message.answer.assert_not_awaited()
