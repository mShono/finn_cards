from __future__ import annotations

import asyncio
from datetime import timedelta

from aiogram import Bot, Dispatcher
from aiogram.types import BotCommand

from kielikaveri.bot.add import router as add_router
from kielikaveri.bot.decks import router as decks_router
from kielikaveri.bot.handlers import router
from kielikaveri.bot.learn import router as learn_router
from kielikaveri.bot.middleware import WhitelistMiddleware
from kielikaveri.config import load_settings
from kielikaveri.db.engine import make_engine, make_session_factory
from kielikaveri.llm.breaker import CallBreaker

# Shown behind Telegram's "/" menu button. Kept short - the persistent
# keyboard (bot/handlers.py's MAIN_KEYBOARD) is the primary way in.
BOT_COMMANDS = [
    BotCommand(command="learn", description="Повторить карточки"),
    BotCommand(command="add", description="Добавить слова из текста"),
    BotCommand(command="decks", description="Колоды"),
    BotCommand(command="stats", description="Статистика"),
    BotCommand(command="help", description="Справка"),
]


async def run() -> None:
    settings = load_settings()
    if not settings.bot_token:
        raise RuntimeError("BOT_TOKEN is not set - see .env.example")
    if not settings.whitelist:
        # An empty whitelist doesn't fail open - WhitelistMiddleware blocks
        # everyone, owner included, with no error anywhere. Better to refuse
        # to start than to run a bot that silently ignores every message.
        raise RuntimeError("WHITELIST_USER_IDS is not set - see .env.example")

    bot = Bot(token=settings.bot_token)
    await bot.set_my_commands(BOT_COMMANDS)
    dp = Dispatcher()
    # Registered after construction, so it runs after the Dispatcher's own
    # UserContextMiddleware and can rely on event_from_user being set.
    dp.update.outer_middleware(WhitelistMiddleware(settings.whitelist))
    dp.include_router(router)
    dp.include_router(learn_router)
    dp.include_router(decks_router)
    # add_router last - chat_message's plain-text catch-all would otherwise
    # shadow every other router's text/state-based message handlers (e.g.
    # decks.py's DeckStates.naming prompt) registered after it.
    dp.include_router(add_router)

    # Schema is managed by alembic (`alembic upgrade head`), not created here.
    engine = make_engine(settings.database_url)
    session_factory = make_session_factory(engine)

    # One breaker for the bot's whole lifetime (plan 3.8, point 3) - it must
    # not reset per handler call, or it could never trip.
    breaker = CallBreaker(
        settings.breaker_max_calls, timedelta(minutes=settings.breaker_window_minutes)
    )

    try:
        await dp.start_polling(
            bot, session_factory=session_factory, settings=settings, breaker=breaker
        )
    finally:
        await engine.dispose()
        await bot.session.close()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
