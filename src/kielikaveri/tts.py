"""Minimal OpenAI TTS wrapper for /learn's "listen" button (plan 3.7, phase 2).

Deliberately thin - one synchronous call, mp3 straight from OpenAI, no
retries or circuit-breaker. A real client with retry/rate-limit handling is
phase 3's job (3.8), once LLM calls need it too.
"""

from __future__ import annotations

from openai import OpenAI


def synthesize_speech(client: OpenAI, model: str, text: str) -> bytes:
    response = client.audio.speech.create(model=model, voice="alloy", input=text)
    return response.read()
