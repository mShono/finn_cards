"""/read - the reader's entry point in the commands menu - the menu button
bot/main.py sets on start, and the 📚 button / typed /learn that now open
learning in the Mini App (plan 2026-11-20-kirjakaveri-reader, П5/У3)."""

import subprocess
import sys
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlsplit

import pytest
from aiogram import Bot
from aiogram.fsm.storage.base import StorageKey
from aiogram.methods import AnswerCallbackQuery, SendMessage
from aiogram.types import InlineKeyboardMarkup, MenuButtonCommands
from bot_helpers import (
    RecordingSession,
    make_breaker,
    make_settings,
    routed_dispatcher,
    tg_callback_update,
    tg_text_update,
)

from kielikaveri.bot.add import WORDS_NOT_SAVED_TEXT, AddStates
from kielikaveri.bot.decks import DeckStates
from kielikaveri.bot.handlers import (
    LEARN_ENTRY_TEXT,
    LEARN_MOVED_TEXT,
    READER_NOT_CONFIGURED_TEXT,
    help_,
    with_query_param,
)
from kielikaveri.bot.main import BOT_COMMANDS, set_up_menu
from kielikaveri.db.engine import create_all, make_engine, make_session_factory

READER_URL = "https://reader.example/"


@pytest.fixture
async def routed():
    # /read itself reads nothing, but WhitelistMiddleware creates the user's
    # `users` row on the first update.
    engine = make_engine("sqlite+aiosqlite://")
    await create_all(engine)
    session_factory = make_session_factory(engine)
    dp = routed_dispatcher()
    telegram = RecordingSession()
    bot = Bot(token="42:TEST", session=telegram)
    key = StorageKey(bot_id=bot.id, chat_id=1, user_id=1)
    # The Dispatcher (and its storage) is shared across tests
    await dp.storage.set_state(key, None)
    await dp.storage.set_data(key, {})

    async def send(text: str, reader_url: str = READER_URL) -> list[SendMessage]:
        since = len(telegram.sent)
        await dp.feed_update(
            bot,
            tg_text_update(since + 1, text),
            session_factory=session_factory,
            settings=make_settings(reader_url=reader_url),
            breaker=make_breaker(),
        )
        return [m for m in telegram.sent[since:] if isinstance(m, SendMessage)]

    async def tap(data: str) -> list:
        since = len(telegram.sent)
        await dp.feed_update(
            bot,
            tg_callback_update(since + 1, data),
            session_factory=session_factory,
            settings=make_settings(reader_url=READER_URL),
            breaker=make_breaker(),
        )
        return telegram.sent[since:]

    yield {"send": send, "tap": tap, "storage": dp.storage, "key": key}
    await dp.storage.set_state(key, None)
    await dp.storage.set_data(key, {})
    await engine.dispose()


def web_app_urls(message: SendMessage) -> list[str]:
    markup = message.reply_markup
    if not isinstance(markup, InlineKeyboardMarkup):
        return []
    return [b.web_app.url for row in markup.inline_keyboard for b in row if b.web_app]


def test_read_is_in_bot_commands():
    assert "read" in [c.command for c in BOT_COMMANDS]


async def test_read_replies_with_inline_web_app_button(routed):
    sent = await routed["send"]("/read")

    assert len(sent) == 1
    assert web_app_urls(sent[0]) == [READER_URL]


async def test_read_without_reader_url_says_not_configured(routed):
    sent = await routed["send"]("/read", reader_url="")

    assert [m.text for m in sent] == [READER_NOT_CONFIGURED_TEXT]
    assert web_app_urls(sent[0]) == []


async def test_read_escapes_deck_naming(routed):
    await routed["storage"].set_state(routed["key"], DeckStates.naming)

    sent = await routed["send"]("/read")

    assert await routed["storage"].get_state(routed["key"]) is None
    # Reached its handler, not taken as the new deck's name
    assert web_app_urls(sent[-1]) == [READER_URL]


async def test_startup_sets_commands_menu_button():
    bot = AsyncMock()

    await set_up_menu(bot)

    bot.set_my_commands.assert_awaited_once_with(BOT_COMMANDS)
    bot.set_chat_menu_button.assert_awaited_once()
    menu_button = bot.set_chat_menu_button.await_args.kwargs["menu_button"]
    assert isinstance(menu_button, MenuButtonCommands)


# --- learning entry (📚 Учить, /learn) ------------------------------------------------


@pytest.mark.parametrize("text", ["📚 Учить", "/learn"])
async def test_learn_entry_replies_with_inline_web_app_button_to_the_learn_screen(routed, text):
    sent = await routed["send"](text)

    assert [m.text for m in sent] == [LEARN_ENTRY_TEXT]
    assert web_app_urls(sent[0]) == [READER_URL + "?screen=learn"]


@pytest.mark.parametrize("text", ["📚 Учить", "/learn"])
async def test_learn_entry_without_reader_url_says_not_configured(routed, text):
    sent = await routed["send"](text, reader_url="")

    assert [m.text for m in sent] == [READER_NOT_CONFIGURED_TEXT]
    assert web_app_urls(sent[0]) == []


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://reader.example/", "https://reader.example/?screen=learn"),
        ("https://reader.example/app", "https://reader.example/app?screen=learn"),
        ("https://reader.example/?v=2", "https://reader.example/?v=2&screen=learn"),
        ("https://reader.example/?screen=books", "https://reader.example/?screen=learn"),
        ("https://reader.example/#top", "https://reader.example/?screen=learn#top"),
    ],
)
def test_with_query_param_keeps_the_rest_of_the_url(url, expected):
    assert with_query_param(url, "screen", "learn") == expected


async def test_learn_entry_url_keeps_a_query_already_in_reader_url(routed):
    sent = await routed["send"]("📚 Учить", reader_url="https://reader.example/?v=2")

    (url,) = web_app_urls(sent[0])
    assert parse_qs(urlsplit(url).query) == {"v": ["2"], "screen": ["learn"]}


async def test_learn_entry_drops_whatever_the_bot_waited_for(routed):
    # A state no router knows (say, of a flow since removed): no input state
    # for InputEscapeMiddleware to drop - learn_entry clears it itself.
    await routed["storage"].set_state(routed["key"], "OldStates:waiting")
    await routed["storage"].set_data(routed["key"], {"queue": ["c1"]})

    sent = await routed["send"]("📚 Учить")

    assert web_app_urls(sent[-1]) == [READER_URL + "?screen=learn"]
    assert await routed["storage"].get_state(routed["key"]) is None
    assert await routed["storage"].get_data(routed["key"]) == {}


async def test_learn_entry_during_deck_pick_warns_and_drops_the_words(routed):
    # PICKER_BREAKERS still lists 📚/learn: the warning must stay true
    await routed["storage"].set_state(routed["key"], AddStates.choosing_deck)
    await routed["storage"].set_data(routed["key"], {"batch_id": "b", "candidates": [{}]})

    sent = await routed["send"]("/learn")

    assert [m.text for m in sent] == [WORDS_NOT_SAVED_TEXT, LEARN_ENTRY_TEXT]
    assert await routed["storage"].get_state(routed["key"]) is None
    assert await routed["storage"].get_data(routed["key"]) == {}


def test_learn_is_not_in_bot_commands():
    assert "learn" not in [c.command for c in BOT_COMMANDS]


async def test_help_does_not_mention_learn():
    message = AsyncMock()

    await help_(message)

    assert "/learn" not in message.answer.await_args.args[0]


def test_no_learn_router_is_in_the_dispatcher():
    # A fresh process: the shared test Dispatcher can't be rebuilt (aiogram
    # attaches a router once).
    script = (
        "from kielikaveri.bot.main import build_dispatcher\n"
        "dp = build_dispatcher({1})\n"
        "assert 'learn' not in [r.name for r in dp.sub_routers], 'learn router is routed'\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize("data", ["learn:reveal:c1", "learn:side:mix", "learn:rate:c1:3"])
async def test_old_learn_button_is_answered_without_a_message(routed, data):
    sent = await routed["tap"](data)

    assert [type(m) for m in sent] == [AnswerCallbackQuery]
    assert sent[0].text == LEARN_MOVED_TEXT
