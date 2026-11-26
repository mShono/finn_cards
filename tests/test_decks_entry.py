"""The bot's way into the reader's decks (plan 2026-11-20-kirjakaveri-reader,
П6/К3): 🗂 Колоды and /decks hand out a web_app button, a typed /delete points
at the reader's search, old deck/edit/delete buttons are answered - and /add,
which stays in the bot, still picks and creates decks as before. All routed
through bot/main.py's real Dispatcher."""

import subprocess
import sys
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from aiogram.fsm.storage.base import StorageKey
from aiogram.methods import AnswerCallbackQuery, SendMessage
from aiogram.types import InlineKeyboardMarkup
from bot_helpers import (
    WORD_CANDIDATE,
    RecordingSession,
    make_breaker,
    make_settings,
    routed_dispatcher,
    tg_callback_update,
    tg_text_update,
)
from conftest import add_users
from sqlalchemy import select

from kielikaveri.bot.add import WORDS_NOT_SAVED_TEXT, AddStates
from kielikaveri.bot.decks import NEW_DECK_PROMPT
from kielikaveri.bot.handlers import (
    DECKS_ENTRY_TEXT,
    DECKS_MOVED_TEXT,
    DELETE_ENTRY_TEXT,
    READER_NOT_CONFIGURED_TEXT,
    help_,
)
from kielikaveri.bot.main import BOT_COMMANDS
from kielikaveri.db.decks import create_deck
from kielikaveri.db.engine import create_all, make_engine, make_session_factory
from kielikaveri.db.models import Deck, Note, NoteKind
from kielikaveri.ingest import ResolvedForms, TokenUsage

READER_URL = "https://reader.example/"
PICK_DATA = {"batch_id": "abc123", "candidates": [WORD_CANDIDATE]}


@pytest.fixture
async def routed(tmp_path, monkeypatch):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    await create_all(engine)
    session_factory = make_session_factory(engine)
    await add_users(session_factory, 1)

    # /add's LLM turn and its forms lookup - never a real call
    check = AsyncMock(return_value=("Нашла слово.", False, [WORD_CANDIDATE], TokenUsage(1, 1, 2)))
    monkeypatch.setattr("kielikaveri.bot.add.check_and_suggest", check)
    forms = AsyncMock(return_value=(ResolvedForms({}, "fst", True), None))
    monkeypatch.setattr("kielikaveri.bot.add.resolve_note_forms", forms)

    dp = routed_dispatcher()
    telegram = RecordingSession()
    bot = Bot(token="42:TEST", session=telegram)
    key = StorageKey(bot_id=bot.id, chat_id=1, user_id=1)
    # The Dispatcher (and its storage) is shared across tests
    await dp.storage.set_state(key, None)
    await dp.storage.set_data(key, {})

    async def feed(update, reader_url: str) -> list:
        since = len(telegram.sent)
        await dp.feed_update(
            bot,
            update,
            session_factory=session_factory,
            settings=make_settings(reader_url=reader_url),
            breaker=make_breaker(),
        )
        return telegram.sent[since:]

    async def send(text: str, reader_url: str = READER_URL) -> list[SendMessage]:
        sent = await feed(tg_text_update(len(telegram.sent) + 1, text), reader_url)
        return [m for m in sent if isinstance(m, SendMessage)]

    async def tap(data: str, reader_url: str = READER_URL) -> list:
        return await feed(tg_callback_update(len(telegram.sent) + 1, data), reader_url)

    async def set_state(state, data: dict | None = None) -> None:
        await dp.storage.set_state(key, state)
        await dp.storage.set_data(key, data or {})

    yield {
        "session_factory": session_factory,
        "check": check,
        "send": send,
        "tap": tap,
        "set_state": set_state,
        "get_state": lambda: dp.storage.get_state(key),
        "get_data": lambda: dp.storage.get_data(key),
    }
    await dp.storage.set_state(key, None)
    await dp.storage.set_data(key, {})
    await engine.dispose()


def web_app_urls(message: SendMessage) -> list[str]:
    markup = message.reply_markup
    if not isinstance(markup, InlineKeyboardMarkup):
        return []
    return [b.web_app.url for row in markup.inline_keyboard for b in row if b.web_app]


def callback_data(message: SendMessage) -> list[str]:
    markup = message.reply_markup
    assert isinstance(markup, InlineKeyboardMarkup)
    return [b.callback_data for row in markup.inline_keyboard for b in row]


async def _notes(session_factory) -> list[tuple[str, str]]:
    """(lemma, deck name) of every note."""
    async with session_factory() as session:
        rows = await session.execute(select(Note.lemma, Deck.name).join(Deck))
        return sorted(tuple(row) for row in rows)


# --- 🗂 Колоды, /decks --------------------------------------------------------------


@pytest.mark.parametrize("text", ["🗂 Колоды", "/decks"])
async def test_decks_entry_replies_with_inline_web_app_button_to_the_decks_screen(routed, text):
    sent = await routed["send"](text)

    assert [m.text for m in sent] == [DECKS_ENTRY_TEXT]
    assert web_app_urls(sent[0]) == [READER_URL + "?screen=decks"]


@pytest.mark.parametrize("text", ["🗂 Колоды", "/decks", "/delete", "/delete hakea"])
async def test_decks_and_delete_entries_without_reader_url_say_not_configured(routed, text):
    sent = await routed["send"](text, reader_url="")

    assert [m.text for m in sent] == [READER_NOT_CONFIGURED_TEXT]
    assert web_app_urls(sent[0]) == []


async def test_decks_entry_creates_no_deck(routed):
    # The bot's old list created «Общая» - now the reader's GET /api/decks does
    await routed["send"]("🗂 Колоды")

    async with routed["session_factory"]() as session:
        assert (await session.scalars(select(Deck))).all() == []


def test_decks_stays_in_bot_commands_and_delete_is_gone():
    commands = [c.command for c in BOT_COMMANDS]

    assert "decks" in commands
    assert "delete" not in commands


async def test_help_drops_delete_and_points_decks_at_the_reader():
    message = AsyncMock()

    await help_(message)

    text = message.answer.await_args.args[0]
    assert "/delete" not in text
    (decks_line,) = [line for line in text.splitlines() if line.startswith("/decks")]
    assert "читалк" in decks_line


def test_no_decks_or_edit_router_is_in_the_dispatcher():
    # A fresh process: the shared test Dispatcher can't be rebuilt (aiogram
    # attaches a router once).
    script = (
        "from kielikaveri.bot.main import build_dispatcher\n"
        "dp = build_dispatcher({1})\n"
        "names = [r.name for r in dp.sub_routers]\n"
        "assert 'decks' not in names and 'edit' not in names, names\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


# --- /delete ------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["/delete", "/delete hakea"])
async def test_typed_delete_replies_with_inline_web_app_button_to_the_search(routed, text):
    sent = await routed["send"](text)

    assert [m.text for m in sent] == [DELETE_ENTRY_TEXT]
    assert web_app_urls(sent[0]) == [READER_URL + "?screen=search"]


# --- old buttons in the chat history ------------------------------------------------


@pytest.mark.parametrize(
    "data",
    [
        "decks:list",
        "decks:new",
        "decks:open:d1:0",
        "noteedit:n1",
        "noteeditfield:n1:lemma",
        "noteeditfield:n1:translation_ru",
        "noteeditcancel",
        "delnote:n1",
        "delnote:cancel",
    ],
)
async def test_old_decks_button_is_answered_without_a_message(routed, data):
    sent = await routed["tap"](data)

    assert [type(m) for m in sent] == [AnswerCallbackQuery]
    assert sent[0].text == DECKS_MOVED_TEXT


async def test_old_delete_button_deletes_nothing(routed):
    async with routed["session_factory"]() as session:
        deck = await create_deck(session, 1, "Общая")
        session.add(
            Note(
                id="n1",
                user_id=1,
                lemma="hakea",
                translation_ru="искать",
                example_fi="Haen töitä.",
                example_ru="Я ищу работу.",
                kind=NoteKind.word,
                deck_id=deck.id,
                meta={},
            )
        )
        await session.commit()

    await routed["tap"]("delnote:n1")

    assert await _notes(routed["session_factory"]) == [("hakea", "Общая")]


@pytest.mark.parametrize("data", ["noteeditcancel", "decks:list", "delnote:cancel"])
async def test_old_decks_button_leaves_a_deck_pick_alone(routed, data):
    await routed["set_state"](AddStates.choosing_deck, PICK_DATA)

    await routed["tap"](data)

    assert await routed["get_state"]() == AddStates.choosing_deck.state
    assert await routed["get_data"]() == PICK_DATA


@pytest.mark.parametrize("data", ["adddeck:abc123:d1", "addnewdeck:abc123"])
async def test_add_picker_buttons_are_not_taken_for_old_deck_buttons(routed, data):
    # No pick in progress: bot/add.py's stray handler answers, not old_decks_button
    sent = await routed["tap"](data)

    assert [type(m) for m in sent] == [AnswerCallbackQuery]
    assert sent[0].text != DECKS_MOVED_TEXT
    assert "неактуальна" in sent[0].text


# --- /add still picks and creates decks ---------------------------------------------


async def _two_decks(session_factory) -> str:
    """«Общая» and «Книга»; returns «Книга»'s id."""
    async with session_factory() as session:
        await create_deck(session, 1, "Общая")
        book = await create_deck(session, 1, "Книга")
        await session.commit()
        return book.id


async def test_add_saves_into_the_picked_deck(routed):
    book_id = await _two_decks(routed["session_factory"])

    replies = await routed["send"]("/add hakea")
    picker = replies[-1]
    assert picker.text == "В какую колоду добавить?"
    (pick,) = [d for d in callback_data(picker) if d.endswith(f":{book_id}")]
    assert pick.startswith("adddeck:")

    await routed["tap"](pick)

    routed["check"].assert_awaited_once()
    assert await _notes(routed["session_factory"]) == [("hakea", "Книга")]
    assert await routed["get_state"]() is None


async def test_add_new_deck_creates_it_and_saves_the_words(routed):
    await _two_decks(routed["session_factory"])
    picker = (await routed["send"]("hakea"))[-1]
    (new_deck,) = [d for d in callback_data(picker) if d.startswith("addnewdeck:")]

    sent = await routed["tap"](new_deck)
    assert [m.text for m in sent if isinstance(m, SendMessage)] == [NEW_DECK_PROMPT]
    assert await routed["get_state"]() == AddStates.naming_new_deck.state

    replies = await routed["send"]("Из книги")

    assert replies[0].text == "Колода «Из книги» создана."
    # The name went to the deck, not to the LLM
    routed["check"].assert_awaited_once()
    assert await _notes(routed["session_factory"]) == [("hakea", "Из книги")]
    assert await routed["get_state"]() is None


@pytest.mark.parametrize("text", ["🗂 Колоды", "/decks"])
async def test_decks_entry_during_deck_pick_keeps_the_pick_working(routed, text):
    book_id = await _two_decks(routed["session_factory"])
    picker = (await routed["send"]("hakea"))[-1]
    data_before = await routed["get_data"]()

    sent = await routed["send"](text)

    # Just the button - no warning, the words are still there
    assert [m.text for m in sent] == [DECKS_ENTRY_TEXT]
    assert web_app_urls(sent[0]) == [READER_URL + "?screen=decks"]
    assert await routed["get_state"]() == AddStates.choosing_deck.state
    assert await routed["get_data"]() == data_before

    (pick,) = [d for d in callback_data(picker) if d.endswith(f":{book_id}")]
    await routed["tap"](pick)

    assert await _notes(routed["session_factory"]) == [("hakea", "Книга")]


async def test_decks_entry_during_deck_pick_then_new_deck_still_works(routed):
    await _two_decks(routed["session_factory"])
    picker = (await routed["send"]("hakea"))[-1]
    await routed["send"]("🗂 Колоды")
    (new_deck,) = [d for d in callback_data(picker) if d.startswith("addnewdeck:")]

    await routed["tap"](new_deck)
    await routed["send"]("Из книги")

    assert await _notes(routed["session_factory"]) == [("hakea", "Из книги")]


async def test_decks_entry_during_new_deck_naming_warns_and_drops_it(routed):
    # As before К3: the name prompt is an input state, so the menu button is
    # navigation - InputEscapeMiddleware drops it, after the warning.
    await routed["set_state"](AddStates.naming_new_deck, PICK_DATA)

    sent = await routed["send"]("🗂 Колоды")

    assert [m.text for m in sent] == [WORDS_NOT_SAVED_TEXT, DECKS_ENTRY_TEXT]
    assert web_app_urls(sent[1]) == [READER_URL + "?screen=decks"]
    assert await routed["get_state"]() is None
    async with routed["session_factory"]() as session:
        assert (await session.scalars(select(Deck))).all() == []
