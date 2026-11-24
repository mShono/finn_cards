"""🔊 for a tapped word (plan 2026-11-20-kirjakaveri-reader, "П3").

The word is spoken as the text spells it (decided 24.11.2026). The mp3
lands in a file cache - READER_TTS_CACHE_DIR, named by the hash of model,
voice, speed and text - so the same word costs one OpenAI call ever. A file
is written to a temporary name and renamed into place: a reader never sees
half an mp3, even if the process dies mid-write.

Logs carry ids, cache hits, sizes and time - never the word.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from openai import AsyncOpenAI

from kielikaveri.llm.breaker import CallBreaker
from kielikaveri.reader.lookup import KeyedLocks, Target
from kielikaveri.tts import TTS_VOICE, synthesize_speech

logger = logging.getLogger(__name__)


def cache_path(cache_dir: Path, model: str, speed: float, text: str) -> Path:
    composite = f"{model}\n{TTS_VOICE}\n{speed!r}\n{text}"
    return cache_dir / (hashlib.sha256(composite.encode("utf-8")).hexdigest() + ".mp3")


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
    """The mp3 of target.surface. Raises CircuitOpenError (and
    BudgetExceededError) and OpenAI's errors on a cache miss."""
    path = cache_path(cache_dir, model, speed, target.surface)
    started = time.monotonic()
    async with locks.hold(path.name):
        audio = await asyncio.to_thread(_read, path)
        cached = audio is not None
        if audio is None:
            breaker.check(datetime.now(UTC))
            audio = await synthesize_speech(client, model, target.surface, speed)
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
