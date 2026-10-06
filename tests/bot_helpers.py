"""Helpers shared by the test modules that drive bot handlers (test_add,
test_routing): stock settings/breaker, a canned LLM candidate, and a harness
that feeds Telegram updates through bot/main.py's real Dispatcher.

A plain module rather than conftest.py: pytest imports conftest for every test
module, and this pulls in aiogram and the whole bot package, which the pure
logic tests (scheduler, morphology, queue, ...) have no need to load.
"""

import functools
from datetime import UTC, datetime, timedelta

from aiogram import Dispatcher
from aiogram.client.session.base import BaseSession
from aiogram.methods import SendMessage
from aiogram.types import CallbackQuery as TgCallbackQuery
from aiogram.types import Chat, InlineKeyboardMarkup, Update
from aiogram.types import Message as TgMessage
from aiogram.types import User as TgUser

from kielikaveri.bot.main import build_dispatcher
from kielikaveri.config import Settings
from kielikaveri.llm.breaker import CallBreaker

WORD_CANDIDATE = {
    "lemma": "hakea",
    "pos": "verbi",
    "translation_ru": "искать",
    "example_fi": "Haen töitä kaupungista.",
    "example_ru": "Я ищу работу в городе.",
    "kind": "word",
    "meta": {"topics": ["työnhaku"]},
}


def make_settings(**overrides) -> Settings:
    defaults = {
        "openai_api_key": "sk-test",
        "openai_text_model": "gpt-5.6-terra",
        "openai_timeout_seconds": 1.0,
        "breaker_max_calls": 60,
        "breaker_window_minutes": 10,
    }
    return Settings(**{**defaults, **overrides})


def make_breaker(**overrides) -> CallBreaker:
    defaults = {"max_calls": 60, "window": timedelta(minutes=10)}
    return CallBreaker(**{**defaults, **overrides})


# --- routing through the real Dispatcher -------------------------------------------


class RecordingSession(BaseSession):
    """Stands in for Telegram: records outgoing API calls instead of sending them."""

    def __init__(self) -> None:
        super().__init__()
        self.sent: list = []

    async def make_request(self, bot, method, timeout=None):
        self.sent.append(method)
        if isinstance(method, SendMessage):
            return TgMessage(
                message_id=len(self.sent),
                date=datetime.now(UTC),
                chat=Chat(id=1, type="private"),
                text=method.text,
                # A received Message only ever carries an inline keyboard -
                # /start's reply keyboard isn't echoed back by Telegram either.
                reply_markup=method.reply_markup
                if isinstance(method.reply_markup, InlineKeyboardMarkup)
                else None,
            )
        return True

    async def stream_content(self, *args, **kwargs):
        raise NotImplementedError

    async def close(self) -> None:
        pass


@functools.cache
def routed_dispatcher() -> Dispatcher:
    # bot/main.py's own build_dispatcher - same middlewares, same router
    # order. Built once per process: aiogram refuses to attach a router twice.
    return build_dispatcher({1})


def tg_user() -> TgUser:
    return TgUser(id=1, is_bot=False, first_name="Test")


def tg_text_update(update_id: int, text: str) -> Update:
    return Update(
        update_id=update_id,
        message=TgMessage(
            message_id=update_id,
            date=datetime.now(UTC),
            chat=Chat(id=1, type="private"),
            from_user=tg_user(),
            text=text,
        ),
    )


def tg_callback_update(update_id: int, data: str) -> Update:
    return Update(
        update_id=update_id,
        callback_query=TgCallbackQuery(
            id=str(update_id),
            from_user=tg_user(),
            chat_instance="test",
            data=data,
            message=TgMessage(
                message_id=update_id,
                date=datetime.now(UTC),
                chat=Chat(id=1, type="private"),
                text="В какую колоду добавить?",
            ),
        ),
    )
