from __future__ import annotations

import asyncio

from aiogram import Bot, Dispatcher

from kielikaveri.bot.handlers import router
from kielikaveri.bot.learn import router as learn_router
from kielikaveri.bot.middleware import WhitelistMiddleware
from kielikaveri.config import load_settings
from kielikaveri.db.engine import make_engine, make_session_factory


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
    dp = Dispatcher()
    # Registered after construction, so it runs after the Dispatcher's own
    # UserContextMiddleware and can rely on event_from_user being set.
    dp.update.outer_middleware(WhitelistMiddleware(settings.whitelist))
    dp.include_router(router)
    dp.include_router(learn_router)

    # Schema is managed by alembic (`alembic upgrade head`), not created here.
    engine = make_engine(settings.database_url)
    session_factory = make_session_factory(engine)

    try:
        await dp.start_polling(bot, session_factory=session_factory, settings=settings)
    finally:
        await engine.dispose()
        await bot.session.close()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
