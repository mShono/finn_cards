from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiogram.types import Update

from kielikaveri.bot.middleware import DECLINE_TEXT, WhitelistMiddleware


def make_update():
    message = SimpleNamespace(answer=AsyncMock())
    update = Update.model_construct(update_id=1, message=message)
    return update, message


async def test_whitelisted_user_reaches_handler():
    middleware = WhitelistMiddleware({1, 2})
    update, message = make_update()
    handler = AsyncMock(return_value="handled")
    data = {"event_from_user": SimpleNamespace(id=1)}

    result = await middleware(handler, update, data)

    assert result == "handled"
    handler.assert_awaited_once_with(update, data)
    message.answer.assert_not_called()


async def test_non_whitelisted_user_is_declined_without_calling_handler():
    middleware = WhitelistMiddleware({1})
    update, message = make_update()
    handler = AsyncMock()
    data = {"event_from_user": SimpleNamespace(id=999)}

    result = await middleware(handler, update, data)

    assert result is None
    handler.assert_not_awaited()
    message.answer.assert_awaited_once_with(DECLINE_TEXT)


async def test_update_with_no_resolvable_user_passes_through():
    # e.g. channel_post carries no from_user - nothing to check, don't block it.
    middleware = WhitelistMiddleware({1})
    update = Update.model_construct(update_id=1, message=None)
    handler = AsyncMock(return_value="handled")

    result = await middleware(handler, update, {})

    assert result == "handled"
    handler.assert_awaited_once_with(update, {})
