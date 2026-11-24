"""The reader's own circuit breaker (plan 2026-11-20-kirjakaveri-reader, И6)."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

from bot_helpers import make_settings

from kielikaveri.llm.breaker import CallBreaker
from kielikaveri.reader.app import READER_BREAKER, make_app


def settings_with(**overrides):
    # make_app refuses an empty BOT_TOKEN.
    return make_settings(bot_token="123456:TEST", **overrides)


def app_with(settings, translator=None):
    return make_app(MagicMock(), settings, translator or MagicMock())


def test_app_has_its_own_breaker_with_default_limits():
    breaker = app_with(settings_with())[READER_BREAKER]

    assert isinstance(breaker, CallBreaker)
    assert breaker._max_calls == 150
    assert breaker._window == timedelta(minutes=10)
    assert breaker._daily_max_calls == 400


def test_limits_come_from_settings():
    settings = settings_with(reader_breaker_max_calls=7, reader_breaker_window_minutes=3)

    breaker = app_with(settings)[READER_BREAKER]

    assert breaker._max_calls == 7
    assert breaker._window == timedelta(minutes=3)


def test_each_app_gets_a_fresh_breaker_not_one_passed_in():
    settings = settings_with()
    session_factory, translator = MagicMock(), MagicMock()
    bot_breaker = CallBreaker(
        settings.breaker_max_calls, timedelta(minutes=settings.breaker_window_minutes)
    )

    first = make_app(session_factory, settings, translator)[READER_BREAKER]
    second = make_app(session_factory, settings, translator)[READER_BREAKER]

    assert first is not second
    assert first is not bot_breaker
    assert first is not translator and first is not session_factory


def test_reader_breaker_trips_independently_of_the_bot_breaker():
    settings = settings_with(reader_breaker_max_calls=150)
    bot_breaker = CallBreaker(
        settings.breaker_max_calls, timedelta(minutes=settings.breaker_window_minutes)
    )
    reader_breaker = app_with(settings)[READER_BREAKER]
    now = datetime(2026, 11, 23, 12, 0, tzinfo=UTC)
    # A reading session's worth of lookups: past the bot's 60, under 150.
    for _ in range(100):
        reader_breaker.check(now)

    bot_breaker.check(now)  # the bot still goes through
