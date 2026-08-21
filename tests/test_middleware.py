from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiogram.types import Chat, Message, Update

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


async def test_channel_post_passes_through():
    # A channel post is authored by the channel, not a Telegram user - there
    # is no from_user, and thus nothing to check against the whitelist.
    middleware = WhitelistMiddleware({1})
    chat = Chat(id=-100123, type="channel")
    post = Message.model_construct(message_id=1, date=0, chat=chat, from_user=None)
    update = Update.model_construct(update_id=1, channel_post=post)
    handler = AsyncMock(return_value="handled")

    result = await middleware(handler, update, {})

    assert result == "handled"
    handler.assert_awaited_once_with(update, {})


async def test_message_with_no_resolvable_user_is_blocked():
    # A channel post auto-forwarded into its linked discussion group: Telegram
    # sets message.from_user to None and message.sender_chat to the channel,
    # so aiogram's UserContextMiddleware never sets event_from_user - but this
    # is still a real, handleable message, not a channel_post update. Must be
    # blocked like any other unverifiable sender, not waved through.
    # (Not the same as an anonymous group admin: those carry a real from_user
    # - the GroupAnonymousBot pseudo-user - and are blocked by the ordinary
    # not-in-whitelist branch instead, exercised by the test above.)
    middleware = WhitelistMiddleware({1})
    group = Chat(id=-100456, type="supergroup")
    message = SimpleNamespace(
        chat=group, sender_chat=Chat(id=-100789, type="channel"), answer=AsyncMock()
    )
    update = Update.model_construct(update_id=1, message=message)
    handler = AsyncMock(return_value="handled")

    result = await middleware(handler, update, {})

    assert result is None
    handler.assert_not_awaited()
    message.answer.assert_not_called()
