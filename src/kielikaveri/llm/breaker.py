"""In-process circuit breaker for OpenAI call volume (plan 3.8, point 3).

Spend itself isn't the risk - a few dollars a month, see plan 3.8 - an
infinite-retry bug is: it could burn a month's budget in an hour, faster
than the OpenAI dashboard's hard limit (point 1) can act on it. One user
physically cannot make 60 calls in 10 minutes by hand, so tripping this
always means a bug, not real usage.

The rate window is rolling: once old calls age out, calls go through again.
So a slow loop that stays under the rate limit could still run all day -
the optional daily cap (a rolling 24 hours) bounds the total, which matters
for the book translator, where every chapter is real money.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

DAY = timedelta(hours=24)


class CircuitOpenError(Exception):
    pass


class BudgetExceededError(CircuitOpenError):
    """The daily cap, not the rate limit - waiting minutes won't help."""


class CallBreaker:
    def __init__(
        self, max_calls: int, window: timedelta, daily_max_calls: int | None = None
    ) -> None:
        self._max_calls = max_calls
        self._window = window
        self._daily_max_calls = daily_max_calls
        self._calls: deque[datetime] = deque()
        self._day_calls: deque[datetime] = deque()

    def check(self, now: datetime) -> None:
        """Record a call attempt.

        Raises BudgetExceededError if over the daily cap, CircuitOpenError if
        over budget for the window. A refused attempt isn't recorded.
        """
        _prune(self._calls, now - self._window)
        _prune(self._day_calls, now - DAY)
        # Daily first: when both are over, "try tomorrow" is the true answer.
        if self._daily_max_calls is not None and len(self._day_calls) >= self._daily_max_calls:
            logger.warning("event=breaker.daily_trip daily_max_calls=%d", self._daily_max_calls)
            raise BudgetExceededError(
                f"more than {self._daily_max_calls} OpenAI calls in 24 hours - stopped"
            )
        if len(self._calls) >= self._max_calls:
            logger.warning(
                "event=breaker.trip max_calls=%d window_minutes=%s",
                self._max_calls,
                self._window,
            )
            raise CircuitOpenError(
                f"more than {self._max_calls} OpenAI calls in {self._window} - stopped"
            )
        self._calls.append(now)
        if self._daily_max_calls is not None:
            self._day_calls.append(now)


def _prune(calls: deque[datetime], cutoff: datetime) -> None:
    while calls and calls[0] < cutoff:
        calls.popleft()
