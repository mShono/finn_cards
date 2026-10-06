"""Minimal OpenAI TTS wrapper for /learn's "listen" button (plan 3.7, phase 2).

Deliberately thin - one async call, mp3 straight from OpenAI, no circuit
breaker. Async so a slow OpenAI never stalls the bot's event loop, and with
its own client rather than llm.client.make_client: that one is tuned for
ingest (long timeout, three retries), while a button the user is waiting on
gives up after one short retry.
"""

from __future__ import annotations

import logging
import time

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

TTS_MAX_RETRIES = 1


def make_tts_client(api_key: str, timeout_seconds: float) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=api_key, max_retries=TTS_MAX_RETRIES, timeout=timeout_seconds)


async def synthesize_speech(
    client: AsyncOpenAI, model: str, text: str, speed: float = 1.0
) -> bytes:
    logger.debug("event=tts.request model=%s speed=%s", model, speed)
    start = time.monotonic()
    response = await client.audio.speech.create(model=model, voice="alloy", input=text, speed=speed)
    audio = await response.aread()
    duration_ms = int((time.monotonic() - start) * 1000)
    logger.info(
        "event=tts.response model=%s duration_ms=%d bytes=%d", model, duration_ms, len(audio)
    )
    return audio
