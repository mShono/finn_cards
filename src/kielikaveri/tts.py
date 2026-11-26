"""Minimal OpenAI TTS wrapper for the reader's 🔊 (reader/speech.py): a
tapped word and a learning card's example.

Deliberately thin - one async call, audio straight from OpenAI, no circuit
breaker of its own (the reader puts its breaker around it). Async so a slow
OpenAI never stalls the event loop, and with its own client rather than
llm.client.make_client: that one is tuned for ingest (long timeout, three
retries), while a button the user is waiting on gives up after one short
retry.
"""

from __future__ import annotations

import logging
import time

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

TTS_MAX_RETRIES = 1
# Named so caches of synthesized audio (reader/speech.py) can key on it.
TTS_VOICE = "alloy"


def make_tts_client(api_key: str, timeout_seconds: float) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=api_key, max_retries=TTS_MAX_RETRIES, timeout=timeout_seconds)


async def synthesize_speech(
    client: AsyncOpenAI,
    model: str,
    text: str,
    speed: float = 1.0,
    instructions: str | None = None,
    response_format: str | None = None,
) -> bytes:
    """mp3 unless response_format says otherwise ("pcm" for the reader).
    instructions only work with gpt-4o-mini-tts and later, not tts-1."""
    logger.debug("event=tts.request model=%s speed=%s", model, speed)
    start = time.monotonic()
    extra = {} if instructions is None else {"instructions": instructions}
    if response_format is not None:
        extra["response_format"] = response_format
    response = await client.audio.speech.create(
        model=model, voice=TTS_VOICE, input=text, speed=speed, **extra
    )
    audio = await response.aread()
    duration_ms = int((time.monotonic() - start) * 1000)
    logger.info(
        "event=tts.response model=%s duration_ms=%d bytes=%d", model, duration_ms, len(audio)
    )
    return audio
