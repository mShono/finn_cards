from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware
from aiogram.types import TelegramObject, Update

DECLINE_TEXT = "Этот бот приватный и отвечает только своему владельцу."

# The two senderless update kinds this bot currently acts on - a channel
# post is authored by the channel, not by an individual Telegram user, so
# there is no `event_from_user` to check and nothing to block.
#
# NOT exhaustive: aiogram's UserContextMiddleware also resolves no user for
# message_reaction_count, non-premium chat_boost, removed_chat_boost and
# deleted_business_messages - they just never reach here today because no
# router handles them, so aiogram doesn't request them via allowed_updates.
# Wiring up a handler for any of those must add it to this tuple too, or
# every such update is silently dropped (return None below, no error).
#
# Everything else that resolves to no user (e.g. a channel post
# auto-forwarded into its linked discussion group, where `message.from_user`
# is None and `message.sender_chat` is the channel) must NOT pass through:
# it still carries a handleable payload, just from a sender we can't verify
# against the whitelist, so the safe default is to block it like any other
# stranger. This is different from an anonymous group admin, whose messages
# carry a real `from_user` (the `GroupAnonymousBot` pseudo-user) and are
# blocked by the ordinary not-in-whitelist branch above instead.
_NO_SENDER_UPDATE_FIELDS = ("channel_post", "edited_channel_post")


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
        if user is not None:
            if user.id in self._whitelist:
                return await handler(event, data)
            if isinstance(event, Update) and event.message is not None:
                await event.message.answer(DECLINE_TEXT)
            return None

        if isinstance(event, Update) and any(
            getattr(event, field, None) is not None for field in _NO_SENDER_UPDATE_FIELDS
        ):
            return await handler(event, data)
        return None
