"""🔊 for a tapped word (plan 2026-11-20-kirjakaveri-reader, "П3").

The word is spoken as the text spells it (decided 24.11.2026). OpenAI sends
raw PCM; the silence around the word is cut off here and the rest becomes a
WAV file in a file cache - READER_TTS_CACHE_DIR, named by the hash of model,
voice, speed, instructions and text - so the same word costs one OpenAI call
ever. A file is written to a temporary name and renamed into place: a reader
never sees half a file, even if the process dies mid-write.

Why it is done this way (tests of 25.11.2026, all on single words):
- OpenAI sometimes answers with a clip that holds no speech: about a second
  of near-silence (peak -45..-56 dBFS; Android's WebView played the mp3 of
  it as loud white noise). tts-1 gave 4 of 15, tts-1-hd 6 of 45,
  gpt-4o-mini-tts none of 31. Cached, such a clip would silence the word for
  good: a fresh clip without a loud enough moment is asked for again and
  never written.
- tts-1 and tts-1-hd squeeze any word into ~0.53 s: cut off or garbled.
  gpt-4o-mini-tts speaks it whole but with up to 1.6 s of silence before it
  and 2.4 s after - hence the trimming.

Logs carry ids, cache hits, sizes and time - never the word.
"""

from __future__ import annotations

import array
import asyncio
import hashlib
import io
import logging
import os
import sys
import tempfile
import time
import wave
from datetime import UTC, datetime
from pathlib import Path

from openai import AsyncOpenAI

from kielikaveri.llm.breaker import CallBreaker
from kielikaveri.reader.lookup import KeyedLocks, Target
from kielikaveri.tts import TTS_VOICE, synthesize_speech

logger = logging.getLogger(__name__)

# For gpt-4o-mini-tts and later; the API says it doesn't work with tts-1*.
INSTRUCTIONS = "Speak the Finnish text clearly, once, in a neutral tone, at a natural pace."
# Tries for a clip with speech before giving up on the tap.
SPEECH_ATTEMPTS = 3

# OpenAI's "pcm": 24 kHz, 16-bit signed little-endian, mono, no header.
SAMPLE_RATE = 24_000
_WINDOW = SAMPLE_RATE // 100  # 10 ms
# A window counts as sound when its peak passes -35 dBFS: speech peaks at
# -1..-8, the empty clips never pass -45.
_LOUD = int(32768 * 10 ** (-35 / 20))
# Kept around the loud part so the word's soft start and end stay (checked
# by ear on kävelin, ystävä, kaukana, talossa).
_BEFORE = SAMPLE_RATE // 10  # 0.10 s
_AFTER = SAMPLE_RATE * 15 // 100  # 0.15 s
# Bumped whenever trimming changes, so old files aren't served.
_FORMAT = "wav-trim-1"


class EmptySpeechError(Exception):
    """OpenAI gave no speech in SPEECH_ATTEMPTS tries."""


def instructions_for(model: str) -> str | None:
    return None if model.startswith("tts-1") else INSTRUCTIONS


def cache_path(cache_dir: Path, model: str, speed: float, text: str) -> Path:
    composite = f"{_FORMAT}\n{model}\n{TTS_VOICE}\n{speed!r}\n{instructions_for(model)}\n{text}"
    return cache_dir / (hashlib.sha256(composite.encode("utf-8")).hexdigest() + ".wav")


def trim_silence(pcm: bytes) -> bytes | None:
    """The loud part of pcm (with a margin) as a WAV file; None when nothing
    in it is loud enough to be speech."""
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) // 2 * 2])
    if sys.byteorder == "big":
        samples.byteswap()
    loud = [
        start
        for start in range(0, len(samples), _WINDOW)
        if max(map(abs, samples[start : start + _WINDOW])) >= _LOUD
    ]
    if not loud:
        return None
    first = max(0, loud[0] - _BEFORE)
    last = min(len(samples), loud[-1] + _WINDOW + _AFTER)
    kept = samples[first:last]
    if sys.byteorder == "big":
        kept.byteswap()
    out = io.BytesIO()
    with wave.open(out, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(SAMPLE_RATE)
        file.writeframes(kept.tobytes())
    return out.getvalue()


def _write_atomically(path: Path, audio: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(audio)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _read(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


async def speak(
    client: AsyncOpenAI,
    breaker: CallBreaker,
    model: str,
    speed: float,
    cache_dir: Path,
    locks: KeyedLocks,
    target: Target,
) -> bytes:
    """The WAV of target.surface. Raises CircuitOpenError (and
    BudgetExceededError), OpenAI's errors and EmptySpeechError on a cache
    miss."""
    path = cache_path(cache_dir, model, speed, target.surface)
    started = time.monotonic()
    async with locks.hold(path.name):
        audio = await asyncio.to_thread(_read, path)
        cached = audio is not None
        if audio is None:
            audio = await _synthesize(client, breaker, model, speed, target)
            await asyncio.to_thread(_write_atomically, path, audio)
    logger.info(
        "event=reader.tts chapter_id=%s paragraph=%d token=%d cached=%s bytes=%d duration_ms=%d",
        target.chapter_id,
        target.paragraph,
        target.token,
        cached,
        len(audio),
        (time.monotonic() - started) * 1000,
    )
    return audio


async def _synthesize(
    client: AsyncOpenAI, breaker: CallBreaker, model: str, speed: float, target: Target
) -> bytes:
    for attempt in range(1, SPEECH_ATTEMPTS + 1):
        # Every try is an OpenAI call: each one goes through the breaker.
        breaker.check(datetime.now(UTC))
        pcm = await synthesize_speech(
            client,
            model,
            target.surface,
            speed,
            instructions=instructions_for(model),
            response_format="pcm",
        )
        audio = await asyncio.to_thread(trim_silence, pcm)
        if audio is not None:
            return audio
        logger.warning(
            "event=reader.tts_empty chapter_id=%s paragraph=%d token=%d attempt=%d bytes=%d",
            target.chapter_id,
            target.paragraph,
            target.token,
            attempt,
            len(pcm),
        )
    raise EmptySpeechError
