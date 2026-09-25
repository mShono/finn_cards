"""Commands and menu buttons sent while a handler waits for free-text input,
routed through bot/main.py's real Dispatcher (see InputEscapeMiddleware)."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from aiogram.fsm.storage.base import StorageKey
from aiogram.methods import SendMessage
from sqlalchemy import select
from test_add import (
    WORD_CANDIDATE,
    RecordingSession,
    make_breaker,
    make_settings,
    routed_dispatcher,
    tg_callback_update,
    tg_text_update,
)

from kielikaveri.bot.add import ADD_PROMPT, LEARN_STOPPED_TEXT, LEARN_TEXT_HINT, AddStates
from kielikaveri.bot.edit import EditStates
from kielikaveri.bot.learn import LearnStates
from kielikaveri.db.decks import get_or_create_default_deck
from kielikaveri.db.engine import create_all, make_engine, make_session_factory
from kielikaveri.db.models import Card, CardType, Deck, Note, NoteKind
from kielikaveri.ingest import TokenUsage


@pytest.fixture
async def routed(tmp_path, monkeypatch):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    await create_all(engine)
    session_factory = make_session_factory(engine)

    # Any text that reaches the chat must be visible, never a real LLM call
    check = AsyncMock(return_value=("Ответ", False, [], TokenUsage(1, 1, 2)))
    monkeypatch.setattr("kielikaveri.bot.add.check_and_suggest", check)

    dp = routed_dispatcher()
    telegram = RecordingSession()
    bot = Bot(token="42:TEST", session=telegram)
    key = StorageKey(bot_id=bot.id, chat_id=1, user_id=1)
    # The Dispatcher (and its storage) is shared across tests
    await dp.storage.set_state(key, None)
    await dp.storage.set_data(key, {})
    context = {
        "session_factory": session_factory,
        "settings": make_settings(),
        "breaker": make_breaker(),
    }

    async def send(text: str) -> list[str]:
        since = len(telegram.sent)
        await dp.feed_update(bot, tg_text_update(len(telegram.sent) + 1, text), **context)
        return [m.text for m in telegram.sent[since:] if isinstance(m, SendMessage)]

    async def tap(data: str) -> list[str]:
        since = len(telegram.sent)
        await dp.feed_update(bot, tg_callback_update(len(telegram.sent) + 1, data), **context)
        return [m.text for m in telegram.sent[since:] if isinstance(m, SendMessage)]

    async def get_state() -> str | None:
        return await dp.storage.get_state(key)

    async def get_data() -> dict:
        return await dp.storage.get_data(key)

    async def set_state(state, data: dict | None = None) -> None:
        await dp.storage.set_state(key, state)
        await dp.storage.set_data(key, data or {})

    yield {
        "session_factory": session_factory,
        "check": check,
        "send": send,
        "tap": tap,
        "get_state": get_state,
        "get_data": get_data,
        "set_state": set_state,
    }
    await dp.storage.set_state(key, None)
    await dp.storage.set_data(key, {})
    await engine.dispose()


async def _add_note(session_factory) -> Note:
    async with session_factory() as session:
        deck = await get_or_create_default_deck(session, 1)
        note = Note(
            id="n1",
            user_id=1,
            lemma="hakea",
            pos="verbi",
            translation_ru="искать",
            example_fi="Haen töitä.",
            example_ru="Я ищу работу.",
            kind=NoteKind.word,
            deck_id=deck.id,
            meta={},
        )
        session.add(note)
        await session.commit()
    return note


async def _note(session_factory) -> Note:
    async with session_factory() as session:
        return await session.get(Note, "n1")


async def _deck_names(session_factory) -> list[str]:
    async with session_factory() as session:
        return list((await session.scalars(select(Deck.name))).all())


# --- EditStates.awaiting_value ------------------------------------------------------


@pytest.mark.parametrize(
    "field, text, expected_reply",
    [
        ("translation_ru", "/delete hakea", "Что удалить?"),
        ("translation_ru", "/add", "Просто напиши мне текст"),
        ("lemma", "/add", "Просто напиши мне текст"),
        ("translation_ru", "💬 Добавить", "Просто напиши мне текст"),
    ],
)
async def test_command_during_card_edit_runs_the_command_and_leaves_the_card(
    routed, field, text, expected_reply
):
    await _add_note(routed["session_factory"])
    await routed["set_state"](EditStates.awaiting_value, {"note_id": "n1", "field": field})

    replies = await routed["send"](text)

    assert replies[0].startswith(expected_reply)
    note = await _note(routed["session_factory"])
    assert (note.lemma, note.translation_ru) == ("hakea", "искать")
    assert await routed["get_state"]() is None


async def test_decks_button_during_card_edit_drops_the_edit(routed):
    await _add_note(routed["session_factory"])
    await routed["set_state"](
        EditStates.awaiting_value, {"note_id": "n1", "field": "translation_ru"}
    )

    replies = await routed["send"]("🗂 Колоды")
    assert replies[0].startswith("Твои колоды:")
    assert await routed["get_state"]() is None

    # The next word is chat again, not the card's new translation
    await routed["send"]("talo")
    routed["check"].assert_awaited_once()
    assert (await _note(routed["session_factory"])).translation_ru == "искать"


@pytest.mark.parametrize("text", ["/cancel", "отмена"])
async def test_cancel_during_card_edit_still_cancels(routed, text):
    await _add_note(routed["session_factory"])
    await routed["set_state"](
        EditStates.awaiting_value, {"note_id": "n1", "field": "translation_ru"}
    )

    assert await routed["send"](text) == ["Отменено."]
    assert (await _note(routed["session_factory"])).translation_ru == "искать"
    assert await routed["get_state"]() is None


# --- AddStates.naming_new_deck ------------------------------------------------------

NAMING_DATA = {"batch_id": "abc123", "candidates": [WORD_CANDIDATE]}


@pytest.mark.parametrize("text", ["💬 Добавить", "/add"])
async def test_add_prompt_during_deck_naming_drops_the_naming(routed, text):
    await routed["set_state"](AddStates.naming_new_deck, NAMING_DATA)

    replies = await routed["send"](text)
    assert replies[0].startswith("Просто напиши мне текст")
    assert await routed["get_state"]() is None

    # The text the prompt asked for goes to the chat, not into a deck name
    await routed["send"]("Minä opiskelen suomea")
    routed["check"].assert_awaited_once()
    assert await _deck_names(routed["session_factory"]) == []


@pytest.mark.parametrize("text", ["/cancel", "отмена", "Отмена"])
async def test_cancel_during_deck_naming_creates_no_deck(routed, text):
    await routed["set_state"](AddStates.naming_new_deck, NAMING_DATA)

    assert await routed["send"](text) == ["Отменено, слова не сохранила."]
    assert await _deck_names(routed["session_factory"]) == []
    assert await routed["get_state"]() is None
    async with routed["session_factory"]() as session:
        assert (await session.scalars(select(Note))).all() == []


async def test_plain_name_during_deck_naming_still_creates_the_deck(routed):
    await routed["set_state"](AddStates.naming_new_deck, NAMING_DATA)

    await routed["send"]("Из книги")

    assert await _deck_names(routed["session_factory"]) == ["Из книги"]


# --- states outside InputEscapeMiddleware -------------------------------------------


async def _start_review(routed) -> None:
    note = await _add_note(routed["session_factory"])
    async with routed["session_factory"]() as session:
        session.add(
            Card(
                id="c1",
                note_id=note.id,
                user_id=1,
                type=CardType.recognition,
                due=datetime.now(UTC) - timedelta(minutes=1),
            )
        )
        await session.commit()
    await routed["send"]("/learn")
    assert await routed["get_state"]() == LearnStates.reviewing.state


async def test_old_edit_menu_cancel_leaves_a_review_session_alone(routed):
    await _start_review(routed)

    await routed["tap"]("noteeditcancel")

    assert await routed["get_state"]() == LearnStates.reviewing.state


async def test_edit_menu_cancel_still_drops_an_edit_in_progress(routed):
    await routed["set_state"](
        EditStates.awaiting_value, {"note_id": "n1", "field": "translation_ru"}
    )

    await routed["tap"]("noteeditcancel")

    assert await routed["get_state"]() is None


@pytest.mark.parametrize(
    "text, expected_reply",
    [
        ("🗂 Колоды", "Твои колоды"),
        ("📚 Учить", "🇫🇮 hakea"),
    ],
)
async def test_menu_button_during_review_keeps_the_session(routed, text, expected_reply):
    await _start_review(routed)

    replies = await routed["send"](text)

    assert replies[0].startswith(expected_reply)
    assert await routed["get_state"]() == LearnStates.reviewing.state


@pytest.mark.parametrize("text", ["💬 Добавить", "/add"])
@pytest.mark.parametrize(
    "state", [LearnStates.deck_choice, LearnStates.debt_choice, LearnStates.reviewing]
)
async def test_add_during_learn_ends_the_session(routed, state, text):
    await routed["set_state"](state, {"queue": ["c1"], "reviewed_count": 0})

    replies = await routed["send"](text)

    assert replies == [LEARN_STOPPED_TEXT, ADD_PROMPT]
    assert await routed["get_state"]() is None
    assert await routed["get_data"]() == {}

    # The text the prompt asks for now reaches the chat
    assert await routed["send"]("talo") == ["Ответ"]
    routed["check"].assert_awaited_once()


async def test_add_with_text_during_review_ends_the_session_before_the_chat(routed):
    routed["check"].return_value = ("Вот", False, [WORD_CANDIDATE], TokenUsage(1, 1, 2))
    await _start_review(routed)

    replies = await routed["send"]("/add talo")

    assert replies == [LEARN_STOPPED_TEXT, "Вот", "В какую колоду добавить?"]
    routed["check"].assert_awaited_once()
    assert await routed["get_state"]() == AddStates.choosing_deck.state
    # No leftovers of the session next to the picker's own data
    assert set(await routed["get_data"]()) == {"batch_id", "candidates"}


async def test_add_button_outside_learn_only_prompts(routed):
    replies = await routed["send"]("💬 Добавить")

    assert replies == [ADD_PROMPT]


# --- plain text during /learn -------------------------------------------------------


@pytest.mark.parametrize(
    "state, data",
    [
        (LearnStates.deck_choice, {}),
        (LearnStates.debt_choice, {"debt_now": "2026-09-25T10:00:00+00:00", "deck_id": None}),
        (
            LearnStates.reviewing,
            {
                "queue": ["c1"],
                "session_started_at": "2026-09-25T10:00:00+00:00",
                "reviewed_count": 0,
                "session_max_minutes": 10,
            },
        ),
    ],
)
async def test_plain_text_during_learn_gets_a_hint_not_the_llm(routed, state, data):
    # Candidates would open the deck picker and overwrite the learn state
    routed["check"].return_value = ("Вот", False, [WORD_CANDIDATE], TokenUsage(1, 1, 2))
    await routed["set_state"](state, data)

    replies = await routed["send"]("talo")

    assert replies == [LEARN_TEXT_HINT]
    routed["check"].assert_not_awaited()
    assert await routed["get_state"]() == state.state
    assert await routed["get_data"]() == data


async def test_reveal_still_works_after_plain_text_during_review(routed):
    routed["check"].return_value = ("Вот", False, [WORD_CANDIDATE], TokenUsage(1, 1, 2))
    await _start_review(routed)

    await routed["send"]("talo")
    replies = await routed["tap"]("learn:reveal:c1")

    assert replies == ["искать\n\nHaen töitä.\nЯ ищу работу."]
    assert await routed["get_state"]() == LearnStates.reviewing.state


@pytest.mark.parametrize(
    "state, data",
    [
        (LearnStates.reviewing, {"queue": ["c1"]}),
        (AddStates.naming_new_deck, NAMING_DATA),
        (EditStates.awaiting_value, {"note_id": "n1", "field": "lemma"}),
    ],
)
async def test_start_resets_any_state(routed, state, data):
    await routed["set_state"](state, data)

    replies = await routed["send"]("/start")

    assert replies[0].startswith("Привет!")
    assert await routed["get_state"]() is None
