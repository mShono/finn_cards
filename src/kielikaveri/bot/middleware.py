from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware
from aiogram.types import TelegramObject, Update

from kielikaveri.logging_context import new_trace_id, trace_id_var

logger = logging.getLogger(__name__)

DECLINE_TEXT = "Этот бот приватный и отвечает только своему владельцу."

# Just a readable log preview, not a content limit.
_PREVIEW_CHARS = 200


def _update_kind(event: TelegramObject) -> str:
    if not isinstance(event, Update):
        return type(event).__name__
    if event.message is not None:
        return "message"
    if event.callback_query is not None:
        return "callback_query"
    if event.channel_post is not None:
        return "channel_post"
    if event.edited_channel_post is not None:
        return "edited_channel_post"
    return "other"


def _update_preview(event: TelegramObject) -> str:
    """DEBUG-only detail for update_received - the actual text/callback data
    a user sent. Kept out of the INFO-level event so a normal run never
    writes raw message content; see logging policy in logging_config.py's
    module docstring and CLAUDE-adjacent plan notes for the reasoning.
    """
    if not isinstance(event, Update):
        return ""
    if event.message is not None:
        text = (
            getattr(event.message, "text", None)
            or getattr(event.message, "caption", None)
            or "<no text/caption>"
        )[:_PREVIEW_CHARS]
        return f"text={text!r}"
    if event.callback_query is not None:
        # callback_data is a bot-defined button payload, not free user text -
        # Telegram itself caps it at 64 bytes (see add.py's _deck_choice_keyboard
        # comment). Sliced anyway so this log line's safety doesn't depend on
        # trusting that platform limit rather than our own code.
        data = getattr(event.callback_query, "data", None)
        if data is not None:
            data = data[:_PREVIEW_CHARS]
        return f"callback_data={data!r}"
    return ""


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


class RequestLoggingMiddleware(BaseMiddleware):
    """Registered first among outer middlewares (before WhitelistMiddleware) -
    assigns one trace_id per Update and logs the update_received /
    request_completed boundary around everything below it: whitelist check,
    routing, and the handler itself.

    Must run outermost so a blocked or errored update still gets a
    request_completed line - WhitelistMiddleware marks data["_route_blocked"]
    instead of raising, since aiogram gives no other way for an inner
    middleware to report its outcome back up the chain.
    """

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        token = trace_id_var.set(new_trace_id())
        start = time.monotonic()
        user = data.get("event_from_user")
        kind = _update_kind(event)
        logger.info("event=update_received user_id=%s kind=%s", user.id if user else None, kind)
        preview = _update_preview(event)
        if preview:
            logger.debug("event=update_received.detail %s", preview)

        status = "success"
        try:
            result = await handler(event, data)
            if data.get("_route_blocked"):
                status = "blocked"
            return result
        except asyncio.CancelledError:
            # CancelledError is a BaseException (not Exception) since Python
            # 3.8 specifically so broad `except Exception` doesn't swallow
            # it - which also means it falls through to `finally` below
            # untouched unless caught here first. Without this branch, a
            # task cancelled mid-handler (e.g. polling shutdown) still logs
            # request_completed via `finally`, but with the wrong status:
            # status=success for a request that never actually finished.
            status = "cancelled"
            raise
        except Exception:
            status = "error"
            logger.exception("event=request_completed.error")
            raise
        finally:
            duration_ms = int((time.monotonic() - start) * 1000)
            logger.info("event=request_completed status=%s duration_ms=%d", status, duration_ms)
            trace_id_var.reset(token)


class WhitelistMiddleware(BaseMiddleware):
    """Registered on dp.update.outer_middleware, after RequestLoggingMiddleware -
    runs before routing.

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
            logger.warning("event=route.blocked reason=whitelist user_id=%s", user.id)
            data["_route_blocked"] = True
            if isinstance(event, Update):
                if event.message is not None:
                    await event.message.answer(DECLINE_TEXT)
                elif event.callback_query is not None:
                    # Telegram shows a spinning loader on the tapped button
                    # until answerCallbackQuery is called - without this, a
                    # blocked callback_query leaves it stuck spinning until
                    # Telegram's own client-side timeout.
                    await event.callback_query.answer(DECLINE_TEXT)
            return None

        if isinstance(event, Update) and any(
            getattr(event, field, None) is not None for field in _NO_SENDER_UPDATE_FIELDS
        ):
            return await handler(event, data)
        logger.warning("event=route.blocked reason=no_sender")
        data["_route_blocked"] = True
        return None


class InputEscapeMiddleware(BaseMiddleware):
    """Registered on dp.message.outer_middleware - runs after the Dispatcher's
    FSMContextMiddleware has resolved `state`/`raw_state`, before any filter.

    In a state that takes the next message as free-text input (a card's new
    value, a new deck's name, the answer to a clarifying question), a command
    or a main-menu button is navigation,
    not input: without this, "/add talo" became a card's translation and
    "💬 Добавить" left the deck-name prompt armed for the next message.
    Such a message drops the pending input and is routed as if no state were
    set. Cancel words are left alone - the input handlers answer those.

    Clearing inside a handler instead would not work: StateFilter reads the
    `raw_state` resolved once per update, not the storage, so every later
    handler would still see the old state. Rewriting data["raw_state"] here
    is what the rest of the routing actually sees.
    """

    def __init__(
        self, input_states: set[str], menu_texts: set[str], cancel_words: set[str]
    ) -> None:
        self._input_states = input_states
        self._menu_texts = menu_texts
        self._cancel_words = cancel_words

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        raw_state = data.get("raw_state")
        text = getattr(event, "text", None) or ""
        if raw_state in self._input_states and self._is_navigation(text):
            logger.debug("event=route.input_abandoned state=%s", raw_state)
            await data["state"].clear()
            data["raw_state"] = None
        return await handler(event, data)

    def _is_navigation(self, text: str) -> bool:
        if text.strip().lower() in self._cancel_words:
            return False
        return text.startswith("/") or text in self._menu_texts
