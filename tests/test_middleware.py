import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.types import Chat, Message, Update
from conftest import log_fields

from kielikaveri.bot.middleware import (
    DECLINE_TEXT,
    RequestLoggingMiddleware,
    WhitelistMiddleware,
    _update_preview,
)
from kielikaveri.logging_context import trace_id_var


def make_update():
    message = SimpleNamespace(answer=AsyncMock())
    update = Update.model_construct(update_id=1, message=message)
    return update, message


def make_callback_update(data: str = "cb:1", *, inline: bool = False) -> Update:
    # Exactly one of message/inline_message_id is set on a real CallbackQuery
    # (aiogram.types.CallbackQuery docstring) - message for a button under a
    # message the bot sent normally, inline_message_id for one sent via
    # Telegram's inline mode (message=None in that case).
    if inline:
        callback_query = SimpleNamespace(
            data=data,
            message=None,
            inline_message_id="inline123",
            from_user=SimpleNamespace(id=1),
            answer=AsyncMock(),
        )
    else:
        callback_query = SimpleNamespace(
            data=data,
            message=SimpleNamespace(answer=AsyncMock()),
            inline_message_id=None,
            from_user=SimpleNamespace(id=1),
            answer=AsyncMock(),
        )
    return Update.model_construct(update_id=1, callback_query=callback_query)


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


async def test_non_whitelisted_callback_query_is_declined_without_calling_handler():
    # A blocked callback_query must still get answered - otherwise Telegram
    # leaves the tapped button's loading spinner stuck until its own timeout.
    middleware = WhitelistMiddleware({1})
    update = make_callback_update()
    handler = AsyncMock()
    data = {"event_from_user": SimpleNamespace(id=999)}

    result = await middleware(handler, update, data)

    assert result is None
    handler.assert_not_awaited()
    update.callback_query.answer.assert_awaited_once_with(DECLINE_TEXT)
    update.callback_query.message.answer.assert_not_called()


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


async def test_request_logging_assigns_and_clears_trace_id(caplog):
    middleware = RequestLoggingMiddleware()
    update, _message = make_update()
    handler = AsyncMock(return_value="handled")

    assert trace_id_var.get() == "-"
    with caplog.at_level(logging.DEBUG, logger="kielikaveri.bot.middleware"):
        await middleware(handler, update, {})
    assert trace_id_var.get() == "-"  # reset after the request, not leaked to the next task

    records = [r for r in caplog.records if r.name == "kielikaveri.bot.middleware"]
    trace_ids = {r.trace_id for r in records}
    assert len(trace_ids) == 1
    assert next(iter(trace_ids)) != "-"


async def test_request_logging_logs_received_and_completed_with_duration(caplog):
    middleware = RequestLoggingMiddleware()
    update, _message = make_update()
    handler = AsyncMock(return_value="handled")
    data = {"event_from_user": SimpleNamespace(id=1)}

    with caplog.at_level(logging.INFO, logger="kielikaveri.bot.middleware"):
        await middleware(handler, update, data)

    events = [log_fields(r.message) for r in caplog.records]
    received = next(f for f in events if f.get("event") == "update_received")
    assert received["user_id"] == "1"
    assert received["kind"] == "message"

    completed = next(f for f in events if f.get("event") == "request_completed")
    assert completed["status"] == "success"
    assert int(completed["duration_ms"]) >= 0


async def test_request_logging_marks_blocked_status(caplog):
    middleware = RequestLoggingMiddleware()
    update, _message = make_update()

    async def blocked_handler(event, data):
        data["_route_blocked"] = True

    with caplog.at_level(logging.INFO, logger="kielikaveri.bot.middleware"):
        await middleware(blocked_handler, update, {})

    completed = next(
        log_fields(r.message)
        for r in caplog.records
        if log_fields(r.message).get("event") == "request_completed"
    )
    assert completed["status"] == "blocked"


async def test_request_logging_marks_error_status_and_reraises(caplog):
    middleware = RequestLoggingMiddleware()
    update, _message = make_update()

    async def failing_handler(event, data):
        raise ValueError("boom")

    with (
        caplog.at_level(logging.INFO, logger="kielikaveri.bot.middleware"),
        pytest.raises(ValueError),
    ):
        await middleware(failing_handler, update, {})

    completed = next(
        log_fields(r.message)
        for r in caplog.records
        if log_fields(r.message).get("event") == "request_completed"
    )
    assert completed["status"] == "error"


async def test_request_logging_marks_cancelled_status_and_reraises_on_task_cancel(caplog):
    """asyncio.CancelledError is a BaseException, not an Exception, since
    Python 3.8 - a plain `except Exception` around the handler call would
    miss it entirely, and `finally` would still log request_completed but
    with the misleading status=success for a request that was actually
    cancelled mid-flight (e.g. polling shutdown). Reproduces a genuine task
    cancellation, not a raised exception - `middleware(...)` must run as a
    real asyncio.Task for cancel() to deliver CancelledError the way it
    would for a real in-flight update."""
    middleware = RequestLoggingMiddleware()
    update, _message = make_update()
    started = asyncio.Event()

    async def hanging_handler(event, data):
        started.set()
        await asyncio.sleep(10)

    with caplog.at_level(logging.INFO, logger="kielikaveri.bot.middleware"):
        task = asyncio.create_task(middleware(hanging_handler, update, {}))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    completed = next(
        log_fields(r.message)
        for r in caplog.records
        if log_fields(r.message).get("event") == "request_completed"
    )
    assert completed["status"] == "cancelled"


async def test_request_logging_gives_different_updates_different_trace_ids(caplog):
    middleware = RequestLoggingMiddleware()
    handler = AsyncMock(return_value="handled")
    seen_trace_ids = []

    async def capturing_handler(event, data):
        seen_trace_ids.append(trace_id_var.get())
        return await handler(event, data)

    with caplog.at_level(logging.INFO, logger="kielikaveri.bot.middleware"):
        update1, _ = make_update()
        await middleware(capturing_handler, update1, {})
        update2, _ = make_update()
        await middleware(capturing_handler, update2, {})

    assert len(seen_trace_ids) == 2
    assert seen_trace_ids[0] != seen_trace_ids[1]


async def test_request_logging_wraps_whitelist_and_shares_trace_id(caplog):
    """One flow through RequestLoggingMiddleware -> WhitelistMiddleware -> handler
    must leave every log entry tagged with the same trace_id (the actual
    correlation guarantee the whole scheme depends on)."""
    outer = RequestLoggingMiddleware()
    inner = WhitelistMiddleware({1})
    update, _message = make_update()
    handler = AsyncMock(return_value="handled")
    data = {"event_from_user": SimpleNamespace(id=1)}

    async def dispatch(event, data):
        return await inner(handler, event, data)

    with caplog.at_level(logging.DEBUG):
        await outer(dispatch, update, data)

    trace_ids = {r.trace_id for r in caplog.records if hasattr(r, "trace_id")}
    assert len(trace_ids) == 1
    handler.assert_awaited_once()


# --- _update_preview: logged user content can't break the log format --------


def test_update_preview_escapes_embedded_newlines_and_control_chars():
    # A user message containing what looks like a second log line, or a raw
    # ANSI escape, must stay inside this one repr()'d string - not become a
    # literal newline (or terminal control code) once written to the log.
    payload = "hi\ntrace=fake0000 event=request_completed status=success\nERROR fake injected line"
    update = Update.model_construct(
        update_id=1, message=SimpleNamespace(text=payload, from_user=SimpleNamespace(id=1))
    )

    preview = _update_preview(update)

    assert "\n" not in preview
    assert preview == f"text={payload!r}"


def test_update_preview_truncates_long_message_text():
    update = Update.model_construct(
        update_id=1,
        message=SimpleNamespace(text="a" * 5000, from_user=SimpleNamespace(id=1)),
    )

    preview = _update_preview(update)

    assert preview == f"text={'a' * 200!r}"


def test_update_preview_truncates_long_callback_data():
    update = Update.model_construct(
        update_id=1,
        callback_query=SimpleNamespace(data="d" * 5000, from_user=SimpleNamespace(id=1)),
    )

    preview = _update_preview(update)

    assert preview == f"callback_data={'d' * 200!r}"


# --- trace_id isolation under real asyncio concurrency -----------------------


async def test_concurrent_updates_never_leak_trace_id_across_tasks(caplog):
    """The actual correlation guarantee: two Updates processed concurrently
    (as aiogram genuinely does - one asyncio Task per Update, each carrying
    its own copy of the context) must never let one request's trace_id bleed
    into the other's, no matter how their awaits interleave.

    Interleaving is forced deterministically via a strict ping-pong
    (A0, B0, A1, B1, A2, B2) using asyncio.Event, not left to timing - this
    is the adversarial worst case (every single step interleaved), not a
    probable-but-unproven one relying on sleep() durations.
    """
    middleware = RequestLoggingMiddleware()
    checkpoint_logger = logging.getLogger("kielikaveri.bot.middleware")
    turn_a, turn_b = asyncio.Event(), asyncio.Event()
    turn_a.set()
    observed: list[tuple[str, str]] = []  # (label, trace_id_var.get() at that instant)

    async def run(label: str, mine: asyncio.Event, other: asyncio.Event) -> None:
        for step in range(3):
            await mine.wait()
            mine.clear()
            observed.append((label, trace_id_var.get()))
            checkpoint_logger.debug("checkpoint request=%s step=%d", label, step)
            other.set()

    update_a, _ = make_update()
    update_b, _ = make_update()

    with caplog.at_level(logging.DEBUG, logger="kielikaveri.bot.middleware"):
        await asyncio.gather(
            middleware(lambda event, data: run("A", turn_a, turn_b), update_a, {}),
            middleware(lambda event, data: run("B", turn_b, turn_a), update_b, {}),
        )

    # Proves this genuinely interleaved step by step - not "A ran to
    # completion, then B ran", which would make the isolation check below
    # trivially true even with a broken (shared/global) trace_id.
    assert [label for label, _trace in observed] == ["A", "B", "A", "B", "A", "B"]

    trace_of_a = {trace for label, trace in observed if label == "A"}
    trace_of_b = {trace for label, trace in observed if label == "B"}
    assert len(trace_of_a) == 1  # every checkpoint inside A's task saw the same trace_id
    assert len(trace_of_b) == 1  # ... and every checkpoint inside B's task saw the same trace_id
    assert (
        trace_of_a != trace_of_b
    )  # ... and never each other's - the actual leak this guards against
    assert "-" not in (trace_of_a | trace_of_b)  # both were assigned, not left at the unset default

    # Same guarantee end-to-end through the real logging pipeline, not just
    # the raw contextvar: every emitted LogRecord carries its own request's
    # trace_id, never the concurrently-running one's.
    checkpoints = [r for r in caplog.records if r.getMessage().startswith("checkpoint request=")]
    assert len(checkpoints) == 6
    expected_trace = {"A": next(iter(trace_of_a)), "B": next(iter(trace_of_b))}
    for record in checkpoints:
        label = record.getMessage().split("request=")[1].split(" ")[0]
        assert record.trace_id == expected_trace[label]


# --- every update kind gets its own trace, not just plain messages ----------


@pytest.mark.parametrize(
    "build_update,expected_kind",
    [
        (lambda: make_update()[0], "message"),
        (lambda: make_callback_update(), "callback_query"),
        (lambda: make_callback_update(inline=True), "callback_query"),
        (
            lambda: Update.model_construct(
                update_id=1,
                channel_post=Message.model_construct(
                    message_id=1, date=0, chat=Chat(id=-100123, type="channel"), from_user=None
                ),
            ),
            "channel_post",
        ),
        (
            # poll_answer is a real Update field aiogram supports - just one
            # this bot wires up no handler for, so it must fall into the
            # generic bucket rather than crash _update_kind's if-chain.
            lambda: Update.model_construct(update_id=1, poll_answer=SimpleNamespace()),
            "other",
        ),
    ],
    ids=[
        "message",
        "callback_query",
        "inline_mode_callback_query",
        "channel_post",
        "unknown_update_type",
    ],
)
async def test_request_logging_starts_a_fresh_trace_for_every_update_kind(
    build_update, expected_kind, caplog
):
    middleware = RequestLoggingMiddleware()
    update = build_update()
    handler = AsyncMock(return_value="handled")

    with caplog.at_level(logging.INFO, logger="kielikaveri.bot.middleware"):
        await middleware(handler, update, {})

    handler.assert_awaited_once()
    received = next(
        log_fields(r.message)
        for r in caplog.records
        if log_fields(r.message).get("event") == "update_received"
    )
    assert received["kind"] == expected_kind

    trace_ids = {r.trace_id for r in caplog.records}
    assert trace_ids == {next(iter(trace_ids))}  # exactly one trace_id for this update
    assert next(iter(trace_ids)) != "-"  # and it was actually assigned


# --- callback_data survives the full middleware chain unchanged -------------


async def test_callback_data_reaches_the_handler_and_the_log_unchanged(caplog):
    """The value the real handler acts on (event.callback_query.data) and
    the value logged at update_received.detail must be the exact same
    string - through RequestLoggingMiddleware AND WhitelistMiddleware,
    neither of which may read, mutate, or re-encode it along the way."""
    outer = RequestLoggingMiddleware()
    inner = WhitelistMiddleware({1})
    # Realistic shape: add.py's own callback_data packs batch_id and a uuid4
    # deck id separated by colons (see bot/add.py's _deck_choice_keyboard).
    payload = "adddeck:258c89c3f781:eee5797f-9875-4c52-9e01-78da1482b612"
    update = make_callback_update(data=payload)
    received_by_handler: dict[str, str] = {}

    async def real_handler(event, data):
        received_by_handler["data"] = event.callback_query.data
        return "handled"

    async def dispatch(event, data):
        return await inner(real_handler, event, data)

    request_data = {"event_from_user": SimpleNamespace(id=1)}
    with caplog.at_level(logging.DEBUG, logger="kielikaveri.bot.middleware"):
        result = await outer(dispatch, update, request_data)

    assert result == "handled"
    assert received_by_handler["data"] == payload

    detail = next(r.message for r in caplog.records if "event=update_received.detail" in r.message)
    assert f"callback_data={payload!r}" in detail
