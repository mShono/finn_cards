from datetime import UTC, datetime, timedelta

import pytest

from kielikaveri.llm.breaker import CallBreaker, CircuitOpenError

NOW = datetime(2026, 8, 25, 10, 0, tzinfo=UTC)


def test_allows_calls_under_the_limit():
    breaker = CallBreaker(max_calls=3, window=timedelta(minutes=10))
    for i in range(3):
        breaker.check(NOW + timedelta(seconds=i))


def test_trips_once_the_limit_is_exceeded():
    breaker = CallBreaker(max_calls=3, window=timedelta(minutes=10))
    for i in range(3):
        breaker.check(NOW + timedelta(seconds=i))

    with pytest.raises(CircuitOpenError):
        breaker.check(NOW + timedelta(seconds=3))


def test_stays_tripped_on_repeated_calls_within_the_window():
    breaker = CallBreaker(max_calls=1, window=timedelta(minutes=10))
    breaker.check(NOW)

    with pytest.raises(CircuitOpenError):
        breaker.check(NOW + timedelta(seconds=1))
    with pytest.raises(CircuitOpenError):
        breaker.check(NOW + timedelta(seconds=2))


def test_old_calls_fall_out_of_the_window_and_free_up_budget():
    breaker = CallBreaker(max_calls=1, window=timedelta(minutes=10))
    breaker.check(NOW)

    with pytest.raises(CircuitOpenError):
        breaker.check(NOW + timedelta(minutes=5))

    # Past the window - the first call has aged out, budget is free again.
    breaker.check(NOW + timedelta(minutes=11))
