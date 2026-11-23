"""The book translator's daily spend cap (review item И4)."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from test_books import (  # noqa: F401 - fixtures are used by name
    GUIDE,
    _chapter,
    _claimed,
    _stored,
    known_words,
    session_factory,
)

from kielikaveri.bot.books import failure_reason, send_chapter_event
from kielikaveri.config import Settings
from kielikaveri.db.models import ChapterStatus
from kielikaveri.llm.breaker import BudgetExceededError, CallBreaker, CircuitOpenError
from kielikaveri.llm.client import make_client
from kielikaveri.translate.worker import Translator

NOW = datetime(2026, 11, 23, 10, 0, tzinfo=UTC)


def test_daily_cap_trips_below_the_rate_limit():
    breaker = CallBreaker(100, timedelta(minutes=10), daily_max_calls=3)
    # Spread out, so the rate window never sees more than one call.
    for i in range(3):
        breaker.check(NOW + timedelta(hours=i))

    with pytest.raises(BudgetExceededError):
        breaker.check(NOW + timedelta(hours=3))


def test_budget_error_is_still_a_circuit_open_error():
    # The worker catches CircuitOpenError - the subclass fails the chapter the same way.
    assert issubclass(BudgetExceededError, CircuitOpenError)


def test_daily_cap_releases_after_24_hours():
    breaker = CallBreaker(100, timedelta(minutes=10), daily_max_calls=2)
    breaker.check(NOW)
    breaker.check(NOW + timedelta(hours=1))
    with pytest.raises(BudgetExceededError):
        breaker.check(NOW + timedelta(hours=23))

    # The first call has aged out; the second hasn't yet.
    breaker.check(NOW + timedelta(hours=24, seconds=1))
    with pytest.raises(BudgetExceededError):
        breaker.check(NOW + timedelta(hours=24, seconds=2))


def test_refused_calls_do_not_extend_the_day():
    breaker = CallBreaker(100, timedelta(minutes=10), daily_max_calls=1)
    breaker.check(NOW)
    for i in range(1, 5):
        with pytest.raises(BudgetExceededError):
            breaker.check(NOW + timedelta(hours=i))

    breaker.check(NOW + timedelta(hours=24, seconds=1))


def test_rate_limit_still_raises_plain_circuit_open_error():
    breaker = CallBreaker(2, timedelta(minutes=10), daily_max_calls=100)
    breaker.check(NOW)
    breaker.check(NOW + timedelta(seconds=1))

    with pytest.raises(CircuitOpenError) as exc_info:
        breaker.check(NOW + timedelta(seconds=2))
    assert type(exc_info.value) is CircuitOpenError


def test_no_daily_cap_by_default():
    breaker = CallBreaker(1, timedelta(minutes=10))
    for i in range(1000):
        breaker.check(NOW + timedelta(minutes=11 * i))


def test_failure_reason_tells_the_daily_cap_from_the_rate_limit():
    budget = failure_reason("BudgetExceededError: more than 150 OpenAI calls in 24 hours")
    rate = failure_reason("CircuitOpenError: more than 60 OpenAI calls in 0:10:00")

    assert budget == "дневной лимит переводов исчерпан, нажми 🔁 завтра"
    assert rate != budget
    assert "предохранитель" in rate


async def test_daily_cap_fails_the_chapter_with_its_own_reason(session_factory):  # noqa: F811
    book_id = await _stored(session_factory)
    chapter_id = await _claimed(session_factory, book_id, 1)
    client = MagicMock()
    client.responses.create = AsyncMock(side_effect=[GUIDE])
    notify = AsyncMock()
    translator = Translator(
        session_factory,
        lambda: client,
        "m",
        CallBreaker(100, timedelta(minutes=10), daily_max_calls=0),
        notify,
    )

    await translator.translate_one(chapter_id)

    chapter = await _chapter(session_factory, book_id, 1)
    assert chapter.status is ChapterStatus.failed
    assert chapter.error.startswith("BudgetExceededError:")
    client.responses.create.assert_not_awaited()

    bot = MagicMock(send_message=AsyncMock())
    await send_chapter_event(bot, notify.await_args.args[0])
    assert "дневной лимит переводов исчерпан" in bot.send_message.await_args.args[1]


def test_bot_client_keeps_three_retries():
    assert make_client("sk-test", 60.0).max_retries == 3


def test_translator_client_takes_fewer_retries():
    assert make_client("sk-test", 300.0, max_retries=1).max_retries == 1


def test_translator_settings_defaults():
    settings = Settings(_env_file=None)
    assert settings.translate_daily_max_calls == 150
    assert settings.translate_max_retries == 1
