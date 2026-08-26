"""In-process circuit breaker for OpenAI call volume (plan 3.8, point 3).

Spend itself isn't the risk - a few dollars a month, see plan 3.8 - an
infinite-retry bug is: it could burn a month's budget in an hour, faster
than the OpenAI dashboard's hard limit (point 1) can act on it. One user
physically cannot make 60 calls in 10 minutes by hand, so tripping this
always means a bug, not real usage.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta


class CircuitOpenError(Exception):
    pass


class CallBreaker:
    def __init__(self, max_calls: int, window: timedelta) -> None:
        self._max_calls = max_calls
        self._window = window
        self._calls: deque[datetime] = deque()

    def check(self, now: datetime) -> None:
        """Record a call attempt. Raises CircuitOpenError if over budget for the window."""
        cutoff = now - self._window
        while self._calls and self._calls[0] < cutoff:
            self._calls.popleft()
        if len(self._calls) >= self._max_calls:
            raise CircuitOpenError(
                f"more than {self._max_calls} OpenAI calls in {self._window} - stopped"
            )
        self._calls.append(now)
