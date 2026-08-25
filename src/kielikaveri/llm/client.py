"""OpenAI client factory for text generation (plan 3.8, points 1-2).

AsyncOpenAI, not the sync client - unlike TTS's short calls (tts.py), ingest
calls can run tens of seconds and must not block the bot's event loop.

max_retries=3 relies on the SDK's own retry predicate (timeouts, 429, 5xx)
rather than reimplementing it - a 400 (bad request, schema violation) is
never retried, since the same request would just fail the same way again,
and blind retries on a non-transient error is the textbook way to build the
budget-burning loop plan 3.8 exists to prevent.
"""

from __future__ import annotations

from openai import AsyncOpenAI


def make_client(api_key: str, timeout_seconds: float) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=api_key, max_retries=3, timeout=timeout_seconds)
