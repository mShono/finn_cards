"""Minimal OpenAI TTS wrapper for /learn's "listen" button (plan 3.7, phase 2).

Deliberately thin - one synchronous call, mp3 straight from OpenAI, no
retries or circuit-breaker. A real client with retry/rate-limit handling is
phase 3's job (3.8), once LLM calls need it too.
"""

from __future__ import annotations

import logging
import time

from openai import OpenAI

logger = logging.getLogger(__name__)


def synthesize_speech(client: OpenAI, model: str, text: str) -> bytes:
    logger.debug("event=tts.request model=%s", model)
    start = time.monotonic()
    response = client.audio.speech.create(model=model, voice="alloy", input=text)
    audio = response.read()
    duration_ms = int((time.monotonic() - start) * 1000)
    logger.info(
        "event=tts.response model=%s duration_ms=%d bytes=%d", model, duration_ms, len(audio)
    )
    return audio
