"""Commands and menu buttons sent while a handler waits for free-text input,
routed through bot/main.py's real Dispatcher (see InputEscapeMiddleware)."""

import re
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

from kielikaveri.bot.add import (
    ADD_PROMPT,
    LEARN_STOPPED_TEXT,
    LEARN_TEXT_HINT,
    UNKNOWN_COMMAND_TEXT,
    WORDS_NOT_SAVED_TEXT,
    AddStates,
)
from kielikaveri.bot.decks import DeckStates
from kielikaveri.bot.edit import EditStates
from kielikaveri.bot.learn import SIDE_PROMPT, LearnStates
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
    # After the warning about the unsaved words
    assert replies[-1].startswith("Просто напиши мне текст")
    assert await routed["get_state"]() is None

    # The text the prompt asked for goes to the chat, not into a deck name
    await routed["send"]("Minä opiskelen suomea")
    routed["check"].assert_awaited_once()
    assert await _deck_names(routed["session_factory"]) == []


@pytest.mark.parametrize("text", ["/cancel", "отмена", "Отмена"])
async def test_cancel_during_deck_naming_creates_no_deck(routed, text):
    await routed["set_state"](AddStates.naming_new_deck, NAMING_DATA)

    assert await routed["send"](text) == [
        "Отменено, слова не сохранила. Чтобы сохранить, пришли текст ещё раз."
    ]
    assert await _deck_names(routed["session_factory"]) == []
    assert await routed["get_state"]() is None
    async with routed["session_factory"]() as session:
        assert (await session.scalars(select(Note))).all() == []


async def test_plain_name_during_deck_naming_still_creates_the_deck(routed):
    await routed["set_state"](AddStates.naming_new_deck, NAMING_DATA)

    await routed["send"]("Из книги")

    assert await _deck_names(routed["session_factory"]) == ["Из книги"]


@pytest.mark.parametrize(
    "state, data, tap_data, cancelled_reply",
    [
        (
            AddStates.choosing_deck,
            NAMING_DATA,
            "addnewdeck:abc123",
            "Отменено, слова не сохранила. Чтобы сохранить, пришли текст ещё раз.",
        ),
        (None, None, "decks:new", "Отменено."),
    ],
    ids=["add-picker", "decks-screen"],
)
async def test_new_deck_prompt_names_a_word_that_cancels_it(
    routed, state, data, tap_data, cancelled_reply
):
    await routed["set_state"](state, data)

    (prompt,) = await routed["tap"](tap_data)
    # The cancel was invisible: the prompt only asked for a name
    (offered,) = re.findall(r"«(.+?)»", prompt)

    assert await routed["send"](offered) == [cancelled_reply]
    assert await _deck_names(routed["session_factory"]) == []
    assert await routed["get_state"]() is None


# --- DeckStates.naming (🗂 Колоды → ➕) ---------------------------------------------


@pytest.mark.parametrize(
    "text, expected_reply",
    [
        ("/add", "Просто напиши мне текст"),
        ("💬 Добавить", "Просто напиши мне текст"),
        ("/delete hakea", "Не нашла такое слово."),
    ],
)
async def test_command_during_decks_naming_runs_the_command_and_creates_no_deck(
    routed, text, expected_reply
):
    await routed["set_state"](DeckStates.naming)

    replies = await routed["send"](text)

    assert replies[0].startswith(expected_reply)
    assert await routed["get_state"]() is None
    assert await _deck_names(routed["session_factory"]) == []


@pytest.mark.parametrize("text", ["/cancel", "отмена", "Отмена"])
async def test_cancel_during_decks_naming_creates_no_deck(routed, text):
    await routed["set_state"](DeckStates.naming)

    assert await routed["send"](text) == ["Отменено."]
    assert await _deck_names(routed["session_factory"]) == []
    assert await routed["get_state"]() is None


async def test_plain_name_during_decks_naming_still_creates_the_deck(routed):
    await routed["set_state"](DeckStates.naming)

    assert await routed["send"]("Из книги") == ["Колода «Из книги» создана."]
    assert await _deck_names(routed["session_factory"]) == ["Из книги"]
    assert await routed["get_state"]() is None


# --- AddStates.awaiting_instruction (clarifying question) ---------------------------

QUESTION_DATA = {"pending_text": "kuusi"}


@pytest.mark.parametrize("text", ["💬 Добавить", "🗂 Колоды", "/help"])
async def test_navigation_during_question_drops_the_question(routed, text):
    await routed["set_state"](AddStates.awaiting_instruction, QUESTION_DATA)

    await routed["send"](text)
    assert await routed["get_state"]() is None

    # The next text is a fresh one, not an answer about "kuusi"
    await routed["send"]("Asun Helsingissä")
    routed["check"].assert_awaited_once()
    assert routed["check"].await_args.kwargs["context_text"] is None


async def test_plain_answer_to_question_still_carries_the_context(routed):
    await routed["set_state"](AddStates.awaiting_instruction, QUESTION_DATA)

    await routed["send"]("ель")

    routed["check"].assert_awaited_once()
    assert routed["check"].await_args.kwargs["context_text"] == "kuusi"


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
    assert await routed["get_state"]() == LearnStates.side_choice.state
    await routed["tap"]("learn:side:mix")
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


async def test_decks_button_during_review_keeps_the_session(routed):
    await _start_review(routed)

    replies = await routed["send"]("🗂 Колоды")

    assert replies[0].startswith("Твои колоды")
    assert await routed["get_state"]() == LearnStates.reviewing.state


async def test_learn_button_during_review_starts_over_with_the_side_question(routed):
    await _start_review(routed)

    replies = await routed["send"]("📚 Учить")

    assert replies == [SIDE_PROMPT]
    assert await routed["get_state"]() == LearnStates.side_choice.state
    assert await routed["tap"]("learn:side:mix") == ["🇫🇮 hakea"]


@pytest.mark.parametrize("text", ["💬 Добавить", "/add"])
@pytest.mark.parametrize(
    "state",
    [
        LearnStates.deck_choice,
        LearnStates.side_choice,
        LearnStates.debt_choice,
        LearnStates.reviewing,
    ],
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
        (LearnStates.side_choice, {"deck_id": None}),
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

    # Deck naming warns about its unsaved words first
    assert replies[-1].startswith("Привет!")
    assert await routed["get_state"]() is None


# --- unknown commands ---------------------------------------------------------------


@pytest.mark.parametrize("text", ["/foo", "/foo bar", "/cancel", "/Learn"])
async def test_unknown_command_gets_an_answer(routed, text):
    assert await routed["send"](text) == [UNKNOWN_COMMAND_TEXT]
    routed["check"].assert_not_awaited()


async def test_unknown_command_during_review_keeps_the_session(routed):
    await _start_review(routed)

    assert await routed["send"]("/foo") == [UNKNOWN_COMMAND_TEXT]
    assert await routed["get_state"]() == LearnStates.reviewing.state


async def test_unknown_command_during_input_drops_the_input(routed):
    await routed["set_state"](DeckStates.naming)

    assert await routed["send"]("/foo") == [UNKNOWN_COMMAND_TEXT]
    assert await routed["get_state"]() is None
    assert await _deck_names(routed["session_factory"]) == []


@pytest.mark.parametrize("text", ["/help", "/decks"])
async def test_known_commands_are_not_unknown(routed, text):
    replies = await routed["send"](text)

    assert replies
    assert UNKNOWN_COMMAND_TEXT not in replies


# --- words not saved yet (UnsavedWordsMiddleware) -----------------------------------


@pytest.mark.parametrize("text", ["📚 Учить", "🗂 Колоды", "💬 Добавить", "/help", "/foo"])
async def test_leaving_deck_naming_warns_the_words_are_lost(routed, text):
    await routed["set_state"](AddStates.naming_new_deck, NAMING_DATA)

    replies = await routed["send"](text)

    # Warning first, then the command itself still runs
    assert replies[0] == WORDS_NOT_SAVED_TEXT
    assert len(replies) > 1
    assert await routed["get_state"]() != AddStates.naming_new_deck.state


async def test_naming_the_deck_gives_no_warning(routed, monkeypatch):
    save = AsyncMock()
    monkeypatch.setattr("kielikaveri.bot.add._save_candidates_and_report", save)
    await routed["set_state"](AddStates.naming_new_deck, NAMING_DATA)

    replies = await routed["send"]("Из книги")

    assert WORDS_NOT_SAVED_TEXT not in replies
    save.assert_awaited_once()


@pytest.mark.parametrize("text", ["📚 Учить", "/learn", "/start"])
async def test_learn_or_start_during_deck_pick_warns_the_words_are_lost(routed, text):
    await routed["set_state"](AddStates.choosing_deck, NAMING_DATA)

    replies = await routed["send"](text)

    assert replies[0] == WORDS_NOT_SAVED_TEXT
    assert len(replies) > 1
    assert await routed["get_state"]() != AddStates.choosing_deck.state


@pytest.mark.parametrize("text", ["🗂 Колоды", "/decks", "/help", "💬 Добавить", "/add"])
async def test_browsing_during_deck_pick_keeps_the_words_quietly(routed, text):
    await routed["set_state"](AddStates.choosing_deck, NAMING_DATA)

    replies = await routed["send"](text)

    assert WORDS_NOT_SAVED_TEXT not in replies
    # The picker's buttons still work afterwards
    assert await routed["get_state"]() == AddStates.choosing_deck.state
    assert await routed["get_data"]() == NAMING_DATA


async def test_learn_without_pending_words_gives_no_warning(routed):
    replies = await routed["send"]("📚 Учить")

    assert WORDS_NOT_SAVED_TEXT not in replies
