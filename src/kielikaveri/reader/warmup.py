"""Keeping the reader's OpenAI connection warm (plan
2026-11-20-kirjakaveri-reader, "П3", "Скорость тапа", point 3).

The first tap after a pause was the slow one (4.8-9.8 s against ~2 s in a
row): most likely a cold connection to OpenAI. Opening a chapter and the
instant FST answer of /api/lemma both mean a lookup is coming, so they start
a free request (models.retrieve) on the lookup's own client, in the
background - the answer never waits for it.

At most one in flight, at most one a minute, and none while the connection
is already in use: every OpenAI call of the reader (lookup, excerpt, the
lemma settling) notes its time here (touch), and a warm-up is only sent
after WARMUP_IDLE of silence. Not through the breaker: it costs nothing and
must not use up the calls a reader has.

The same record gives event=reader.lookup its idle_s and warm fields.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# No warm-up within this long of the last one, nor of the last OpenAI call.
WARMUP_IDLE = 60.0


class Warmup:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        # Monotonic times; None - never.
        self._last_contact: float | None = None
        self._last_started: float | None = None
        self._last_warmed: float | None = None
        self._task: asyncio.Task | None = None

    def touch(self) -> None:
        """An OpenAI call of the reader just went out (or came back)."""
        self._last_contact = self._clock()

    def idle_seconds(self) -> float | None:
        """Seconds since the last OpenAI contact (calls and warm-ups); None
        if there was none since start."""
        if self._last_contact is None:
            return None
        return self._clock() - self._last_contact

    def warm(self) -> bool:
        """Whether a warm-up succeeded within the last WARMUP_IDLE."""
        return self._last_warmed is not None and self._clock() - self._last_warmed <= WARMUP_IDLE

    @property
    def in_flight(self) -> bool:
        return self._task is not None and not self._task.done()

    def maybe_start(self, client: AsyncOpenAI, model: str) -> asyncio.Task | None:
        """Start a warm-up in the background if one is due; never waits."""
        now = self._clock()
        if self.in_flight:
            return None
        if self._last_started is not None and now - self._last_started < WARMUP_IDLE:
            return None
        if self._last_contact is not None and now - self._last_contact < WARMUP_IDLE:
            return None
        self._last_started = now
        self._task = asyncio.create_task(self._run(client, model))
        return self._task

    async def _run(self, client: AsyncOpenAI, model: str) -> None:
        started = time.monotonic()
        try:
            await client.models.retrieve(model)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a warm-up must never fail loudly
            # The class only: a message can quote the request. No traceback:
            # a failed warm-up is no one's problem, the lookup will retry.
            logger.info(
                "event=reader.warmup ms=%d ok=0 error=%s",
                (time.monotonic() - started) * 1000,
                type(exc).__name__,
            )
            return
        self._last_warmed = self._clock()
        self.touch()
        logger.info("event=reader.warmup ms=%d ok=1", (time.monotonic() - started) * 1000)

    async def stop(self) -> None:
        """Cancel a warm-up in flight (the app is shutting down)."""
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
