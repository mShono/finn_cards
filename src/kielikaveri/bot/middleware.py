from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware
from aiogram.types import TelegramObject, Update

DECLINE_TEXT = "Этот бот приватный и отвечает только своему владельцу."


class WhitelistMiddleware(BaseMiddleware):
    """Registered on dp.update.outer_middleware - runs before routing.

    Must be registered after the Dispatcher's own UserContextMiddleware
    (i.e. via `dp.update.outer_middleware(...)`, not passed to the
    Dispatcher constructor) so `event_from_user` is already resolved for
    every update type by the time this runs.
    """

    def __init__(self, whitelist: set[int]) -> None:
        self._whitelist = whitelist

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        user = data.get("event_from_user")
        if user is not None and user.id not in self._whitelist:
            if isinstance(event, Update) and event.message is not None:
                await event.message.answer(DECLINE_TEXT)
            return None
        return await handler(event, data)
