"""/read - the reader's entry point in the commands menu - and the menu
button bot/main.py sets on start."""

from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from aiogram.fsm.storage.base import StorageKey
from aiogram.methods import SendMessage
from aiogram.types import InlineKeyboardMarkup, MenuButtonCommands
from bot_helpers import (
    RecordingSession,
    make_breaker,
    make_settings,
    routed_dispatcher,
    tg_text_update,
)

from kielikaveri.bot.decks import DeckStates
from kielikaveri.bot.handlers import READER_NOT_CONFIGURED_TEXT
from kielikaveri.bot.main import BOT_COMMANDS, set_up_menu

READER_URL = "https://reader.example/"


@pytest.fixture
async def routed():
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
            session_factory=None,
            settings=make_settings(reader_url=reader_url),
            breaker=make_breaker(),
        )
        return [m for m in telegram.sent[since:] if isinstance(m, SendMessage)]

    yield {"send": send, "storage": dp.storage, "key": key}
    await dp.storage.set_state(key, None)
    await dp.storage.set_data(key, {})


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
