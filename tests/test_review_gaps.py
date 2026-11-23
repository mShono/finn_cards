"""Gaps found by the review of 23.11.2026: tests for the current behaviour of
the upload size limit, initData edge cases, /api routing behind the auth
middleware and the traceback of a failed chapter. All texts are synthetic."""

import hashlib
import hmac
import io
import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import urlencode

import pytest
from aiogram import Bot
from aiogram.fsm.storage.base import StorageKey
from aiogram.methods import GetFile, SendMessage
from aiogram.types import Chat, Document, Update
from aiogram.types import Message as TgMessage
from bot_helpers import RecordingSession, make_breaker, make_settings, routed_dispatcher, tg_user
from sqlalchemy import func, select
from test_books import _chapter, _claimed, _stored, make_translator
from test_reader import (  # noqa: F401 - pytest fixtures
    OWNER,
    TOKEN,
    client,
    session_factory,
    translator,
)

from kielikaveri.db.models import Book, ChapterStatus
from kielikaveri.reader.auth import AuthError, user_id_from_header

# A sentence no real book has, so finding it in the log can only mean a leak.
SECRET = "Фиолетовый жираф Аристарх пересчитывал облака над пустым вокзалом."


# --- bot: a book over max_book_bytes ----------------------------------------------


async def test_too_big_txt_is_refused_without_downloading(session_factory, monkeypatch):  # noqa: F811
    download = AsyncMock(side_effect=lambda *a, **k: io.BytesIO(SECRET.encode()))
    monkeypatch.setattr(Bot, "download", download)
    settings = make_settings(max_book_bytes=1_000_000)
    dp = routed_dispatcher()
    telegram = RecordingSession()
    bot = Bot(token="42:TEST", session=telegram)
    key = StorageKey(bot_id=bot.id, chat_id=1, user_id=1)
    await dp.storage.set_state(key, None)
    await dp.storage.set_data(key, {})
    update = Update(
        update_id=1,
        message=TgMessage(
            message_id=1,
            date=datetime.now(UTC),
            chat=Chat(id=1, type="private"),
            from_user=tg_user(),
            document=Document(
                file_id="f",
                file_unique_id="u",
                file_name="Iso.txt",
                mime_type="text/plain",
                file_size=settings.max_book_bytes + 1,
            ),
        ),
    )
    context = {
        "session_factory": session_factory,
        "settings": settings,
        "breaker": make_breaker(),
        "translator": MagicMock(),
    }

    await dp.feed_update(bot, update, **context)

    [reply] = [m for m in telegram.sent if isinstance(m, SendMessage)]
    assert "слишком большой" in reply.text
    assert "1 МБ" in reply.text
    download.assert_not_called()
    assert not any(isinstance(m, GetFile) for m in telegram.sent)
    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(Book)) == 0


# --- reader/auth.py: initData edge cases ------------------------------------------


def _signed(fields: dict[str, str], token: str = TOKEN) -> str:
    """initData signed over exactly `fields` (no field added by default)."""
    check_string = "\n".join(f"{k}={v}" for k, v in sorted(fields.items()))
    secret = hmac.new(b"WebAppData", token.encode(), hashlib.sha256).digest()
    signed = dict(fields, hash=hmac.new(secret, check_string.encode(), hashlib.sha256).hexdigest())
    return urlencode(signed)


def _fields(auth_date: datetime | None = None, user: bool = True) -> dict[str, str]:
    fields = {
        "auth_date": str(int((auth_date or datetime.now(UTC)).timestamp())),
        "query_id": "AAH",
    }
    if user:
        fields["user"] = json.dumps({"id": OWNER, "first_name": "Test"})
    return fields


def _check(init_data: str) -> int:
    return user_id_from_header("tma " + init_data, TOKEN, {OWNER}, timedelta(hours=24))


def test_valid_signature_without_user_is_refused_as_no_user():
    with pytest.raises(AuthError, match="no_user"):
        _check(_signed(_fields(user=False)))


def test_auth_date_more_than_five_minutes_in_the_future_is_refused():
    """A correctly signed auth_date an hour ahead is beyond any clock skew."""
    future = datetime.now(UTC) + timedelta(hours=1)
    with pytest.raises(AuthError, match="from_future"):
        _check(_signed(_fields(auth_date=future)))


def test_init_data_without_hash_is_refused():
    with pytest.raises(AuthError, match="bad_signature"):
        _check(urlencode(_fields()))


# --- reader/app.py: auth runs before routing --------------------------------------


@pytest.mark.parametrize("path", ["/api/nope", "/api/books/x/y", "/api/"])
async def test_unknown_api_path_without_init_data_is_401_not_404(client, path):  # noqa: F811
    assert (await client.get(path)).status == 401


# --- translate/worker.py: the traceback of a failed chapter -----------------------


class FakeParseError(Exception):
    """An unexpected error whose message quotes the text it choked on - as
    a JSON or validation error on the model's answer would."""


async def _fail_chapter_with_secret(factory, caplog):
    caplog.set_level("DEBUG")
    # As in production (logging_config._TEXT_LEAKING_LOGGERS): aiosqlite logs
    # SQL parameters at DEBUG and is held at INFO even when the app runs at DEBUG.
    caplog.set_level("INFO", logger="aiosqlite")
    text = f"Название\n\n1\n\n{SECRET} " + "Обычный синтетический абзац. " * 20
    book_id = await _stored(factory, text=text)
    chapter_id = await _claimed(factory, book_id, 1)
    worker, openai_client, _ = make_translator(factory)
    openai_client.responses.create.side_effect = FakeParseError(f"cannot parse: {SECRET}")

    await worker.translate_one(chapter_id)

    assert (await _chapter(factory, book_id, 1)).status is ChapterStatus.failed
    assert "event=translate.chapter_failed" in caplog.text
    assert any(r.exc_info for r in caplog.records)


async def test_failed_chapter_log_messages_carry_no_book_text(session_factory, caplog):  # noqa: F811
    await _fail_chapter_with_secret(session_factory, caplog)

    assert SECRET not in "\n".join(r.getMessage() for r in caplog.records)


@pytest.mark.xfail(
    strict=True,
    reason="worker logs an unexpected chapter error with logger.exception: the "
    "traceback ends in str(exc), which carries book text when the exception quotes it",
)
async def test_failed_chapter_traceback_carries_no_book_text(session_factory, caplog):  # noqa: F811
    await _fail_chapter_with_secret(session_factory, caplog)

    assert SECRET not in caplog.text
