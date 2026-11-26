from __future__ import annotations

import asyncio
import contextlib
import functools
from datetime import timedelta

from aiogram import Bot, Dispatcher
from aiogram.types import BotCommand, MenuButtonCommands
from aiohttp import web

from kielikaveri.bot.add import WORDS_NOT_SAVED_TEXT, AddStates
from kielikaveri.bot.add import router as add_router
from kielikaveri.bot.books import router as books_router
from kielikaveri.bot.books import send_chapter_event
from kielikaveri.bot.decks import DeckStates
from kielikaveri.bot.decks import router as decks_router
from kielikaveri.bot.edit import CANCEL_WORDS, EditStates
from kielikaveri.bot.edit import router as edit_router
from kielikaveri.bot.handlers import MAIN_KEYBOARD, router
from kielikaveri.bot.middleware import (
    InputEscapeMiddleware,
    RequestLoggingMiddleware,
    UnsavedWordsMiddleware,
    WhitelistMiddleware,
    is_navigation,
)
from kielikaveri.config import Settings, load_settings
from kielikaveri.db.engine import make_engine, make_session_factory
from kielikaveri.llm.breaker import CallBreaker
from kielikaveri.llm.client import make_client
from kielikaveri.logging_config import setup_logging
from kielikaveri.reader.app import make_app
from kielikaveri.translate.worker import Translator

# Shown behind Telegram's "/" menu button. Kept short - the persistent
# keyboard (bot/handlers.py's MAIN_KEYBOARD) is the primary way in; /read is
# the reader's entry point (see bot/handlers.py's read). No /learn: learning
# lives in the Mini App, reached by the 📚 button (bot/handlers.py's learn_entry).
BOT_COMMANDS = [
    BotCommand(command="read", description="📖 Читать"),
    BotCommand(command="add", description="Добавить слова из текста"),
    BotCommand(command="delete", description="Удалить слово"),
    BotCommand(command="decks", description="Колоды"),
    BotCommand(command="help", description="Справка"),
]

# States whose handler takes the next message as a value - see
# InputEscapeMiddleware for why commands and menu buttons must skip them.
INPUT_STATES = {
    EditStates.awaiting_value.state,
    AddStates.naming_new_deck.state,
    AddStates.awaiting_instruction.state,
    DeckStates.naming.state,
}
MENU_TEXTS = {button.text for row in MAIN_KEYBOARD.keyboard for button in row}
# The deck picker itself survives browsing (🗂 Колоды, /help) - its buttons
# still work after. Only these overwrite its state and kill it.
PICKER_BREAKERS = {"/learn", "/start", "📚 Учить"}


def drops_unsaved_words(raw_state: str | None, text: str) -> bool:
    """Whether this message ends a deck pick without saving its words."""
    if raw_state == AddStates.naming_new_deck.state:
        # InputEscapeMiddleware drops the naming on any navigation
        return is_navigation(text, MENU_TEXTS, CANCEL_WORDS)
    if raw_state == AddStates.choosing_deck.state:
        if text in PICKER_BREAKERS:
            return True
        # "/learn@bot args" -> "/learn"
        command = text.split(maxsplit=1)[0].split("@")[0] if text.startswith("/") else ""
        return command in PICKER_BREAKERS
    return False


async def set_up_menu(bot: Bot) -> None:
    """The commands list and the menu button that shows it."""
    await bot.set_my_commands(BOT_COMMANDS)
    # Always the commands button: a Mini App button here would replace the
    # command list. The reader opens from /read's inline button instead. Set
    # explicitly, so a WebApp button left by an older deploy is undone.
    await bot.set_chat_menu_button(menu_button=MenuButtonCommands())


def build_dispatcher(whitelist: set[int]) -> Dispatcher:
    """Middlewares and routers, in order. Separate from run() so routing
    tests go through exactly this, not a hand-copied router list.

    Callable once per process - aiogram refuses to attach a router twice.
    """
    dp = Dispatcher()
    # bot/books.py ignores a .txt sent while one of these waits for input.
    dp["input_states"] = INPUT_STATES
    # Registered after construction, so both run after the Dispatcher's own
    # UserContextMiddleware and can rely on event_from_user being set.
    # RequestLoggingMiddleware first - it must wrap WhitelistMiddleware to
    # log request_completed for updates the whitelist check blocks too.
    dp.update.outer_middleware(RequestLoggingMiddleware())
    dp.update.outer_middleware(WhitelistMiddleware(whitelist))
    # Before InputEscapeMiddleware - it must see the words before they're dropped
    dp.message.outer_middleware(UnsavedWordsMiddleware(drops_unsaved_words, WORDS_NOT_SAVED_TEXT))
    dp.message.outer_middleware(InputEscapeMiddleware(INPUT_STATES, MENU_TEXTS, CANCEL_WORDS))
    # No learning router: learning lives in the Mini App (reader/learn.py),
    # the bot only hands out its button (bot/handlers.py's learn_entry).
    dp.include_router(router)
    dp.include_router(decks_router)
    dp.include_router(edit_router)
    dp.include_router(books_router)
    # add_router last - chat_message's plain-text catch-all would otherwise
    # shadow every other router's text/state-based message handlers (e.g.
    # decks.py's DeckStates.naming prompt, edit.py's EditStates.awaiting_value
    # prompt) registered after it.
    dp.include_router(add_router)
    return dp


async def run(settings: Settings) -> None:
    if not settings.bot_token:
        raise RuntimeError("BOT_TOKEN is not set - see .env.example")
    if not settings.whitelist:
        # An empty whitelist doesn't fail open - WhitelistMiddleware blocks
        # everyone, owner included, with no error anywhere. Better to refuse
        # to start than to run a bot that silently ignores every message.
        raise RuntimeError("WHITELIST_USER_IDS is not set - see .env.example")

    bot = Bot(token=settings.bot_token)
    await set_up_menu(bot)
    dp = build_dispatcher(settings.whitelist)

    # Schema is managed by alembic (`alembic upgrade head`), not created here.
    engine = make_engine(settings.database_url)
    session_factory = make_session_factory(engine)

    # One breaker for the bot's whole lifetime (plan 3.8, point 3) - it must
    # not reset per handler call, or it could never trip.
    breaker = CallBreaker(
        settings.breaker_max_calls, timedelta(minutes=settings.breaker_window_minutes)
    )

    # Same process, own breaker (plan 2026-11-20-kirjakaveri-reader, "Старт П1").
    translator = Translator(
        session_factory,
        functools.partial(
            make_client,
            settings.openai_api_key,
            settings.translate_timeout_seconds,
            max_retries=settings.translate_max_retries,
        ),
        settings.openai_text_model,
        CallBreaker(
            settings.translate_breaker_max_calls,
            timedelta(minutes=settings.translate_breaker_window_minutes),
            daily_max_calls=settings.translate_daily_max_calls,
        ),
        functools.partial(send_chapter_event, bot),
    )
    # The reader (plan 2026-11-20-kirjakaveri-reader, "П2"). aiohttp's own
    # access log is off: reader/app.py logs each request without headers.
    reader = web.AppRunner(make_app(session_factory, settings, translator), access_log=None)
    await reader.setup()
    await web.TCPSite(reader, settings.reader_host, settings.reader_port).start()
    # Logs at once if the loop ever dies (plan 2026-11-20-kirjakaveri-reader, И1).
    translator_task = translator.start()

    try:
        await dp.start_polling(
            bot,
            session_factory=session_factory,
            settings=settings,
            breaker=breaker,
            translator=translator,
        )
    finally:
        # A chapter cut off here stays `translating` and is requeued on start.
        translator_task.cancel()
        # Exception too, so the cleanup below always runs; a death is already
        # logged with its traceback by the task's done-callback.
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await translator_task
        await reader.cleanup()
        await engine.dispose()
        await bot.session.close()


def main() -> None:
    settings = load_settings()
    # Without a handler, every logger.info() call across the app (token-usage
    # analytics, cache hits, ingest results) is silently dropped everywhere,
    # not just hidden from journalctl's caller - found live 27.08.2026 while
    # trying to read a log line that was never written. setup_logging wires
    # both a console handler (systemd/journald captures stdout) and a
    # rotated file, so history survives past journald's own retention.
    setup_logging(settings.log_level, settings.log_file)
    asyncio.run(run(settings))


if __name__ == "__main__":
    main()
