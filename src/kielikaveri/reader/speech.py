"""🔊 for a tapped word (plan 2026-11-20-kirjakaveri-reader, "П3").

The word is spoken as the text spells it (decided 24.11.2026). The mp3
lands in a file cache - READER_TTS_CACHE_DIR, named by the hash of model,
voice, speed, instructions and text - so the same word costs one OpenAI call
ever. A file is written to a temporary name and renamed into place: a reader
never sees half an mp3, even if the process dies mid-write.

OpenAI sometimes answers with a clip that holds no speech: valid mp3 frames,
about a second of near-silence (peak -45..-55 dBFS; Android's WebView plays
it as loud white noise). Seen 25.11.2026 on short inputs: tts-1 4 of 15,
tts-1-hd 6 of 45, gpt-4o-mini-tts 0 of 15 - but that one puts up to 1.6 s
of silence before the word, so the reader uses tts-1-hd. Cached, an empty
clip would silence the word for good, so every fresh clip goes through
has_speech first: an empty one is asked for again and never written.

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

# For gpt-4o-mini-tts and later, should READER_TTS_MODEL name one.
INSTRUCTIONS = "Speak the Finnish text clearly, once, in a neutral tone, at a natural pace."
# Tries for a clip with speech. At tts-1-hd's ~13% empty answers, three
# leave about 1 new word in 450 without sound (two: 1 in 60).
SPEECH_ATTEMPTS = 3

# MPEG audio frame header fields, by version (3 = MPEG-1, 2 = MPEG-2,
# 0 = MPEG-2.5; 1 is reserved). Layer III only: that is what OpenAI sends.
_BITRATES_KBPS = {
    3: (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320),
    2: (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),
}
_SAMPLE_RATES = {3: (44100, 48000, 32000), 2: (22050, 24000, 16000), 0: (11025, 12000, 8000)}
# The loudest frame's global_gain (the quantizer step, so the overall level)
# in OpenAI's clips, 25.11.2026: 127-133 in all 13 empty ones, 152-162 in
# all 47 with speech, on tts-1, tts-1-hd and gpt-4o-mini-tts alike; a
# second tts-1-hd run of 30 matched a decoder on every clip.
_SPEECH_MIN_GAIN = 145


def _frame_gains(audio: bytes) -> list[int]:
    """global_gain of granule 0, channel 0, of every Layer III frame that
    carries audio; ID3 tags and stray bytes are skipped."""
    gains = []
    i = 0
    if audio[:3] == b"ID3" and len(audio) >= 10:
        i = 10 + ((audio[6] << 21) | (audio[7] << 14) | (audio[8] << 7) | audio[9])
    while i + 4 <= len(audio):
        header = int.from_bytes(audio[i : i + 4], "big")
        version = (header >> 19) & 3
        layer = (header >> 17) & 3
        bitrate_index = (header >> 12) & 15
        rate_index = (header >> 10) & 3
        if (
            header >> 21 != 0x7FF
            or version == 1
            or layer != 1
            or bitrate_index in (0, 15)
            or rate_index == 3
        ):
            i += 1
            continue
        mpeg1 = version == 3
        bitrate = _BITRATES_KBPS[3 if mpeg1 else 2][bitrate_index] * 1000
        rate = _SAMPLE_RATES[version][rate_index]
        size = (144 if mpeg1 else 72) * bitrate // rate + ((header >> 9) & 1)
        mono = (header >> 6) & 3 == 3
        side = i + 4 + (0 if header & 0x10000 else 2)  # protection bit 0: a CRC follows
        if side + 9 > len(audio):
            break
        bits = int.from_bytes(audio[side : side + 9], "big")
        # Side info up to granule 0, channel 0: main_data_begin, private bits
        # (and scfsi for MPEG-1), then part2_3_length (12), big_values (9),
        # global_gain (8).
        if mpeg1:
            skip = 9 + (5 if mono else 3) + (4 if mono else 8)
        else:
            skip = 8 + (1 if mono else 2)
        part2_3_length = (bits >> (72 - skip - 12)) & 0xFFF
        gain = (bits >> (72 - skip - 12 - 9 - 8)) & 0xFF
        # An all-zero frame (digital silence) has no level to speak of.
        if part2_3_length:
            gains.append(gain)
        i += size
    return gains


def has_speech(audio: bytes) -> bool:
    """False for OpenAI's empty clips (see the module docstring) and for
    anything that isn't an mp3 with audio in it."""
    gains = _frame_gains(audio)
    return bool(gains) and max(gains) >= _SPEECH_MIN_GAIN


class EmptySpeechError(Exception):
    """OpenAI gave no speech in SPEECH_ATTEMPTS tries."""


def instructions_for(model: str) -> str | None:
    # The API: instructions don't work with tts-1 and tts-1-hd.
    return None if model.startswith("tts-1") else INSTRUCTIONS


def cache_path(cache_dir: Path, model: str, speed: float, text: str) -> Path:
    composite = f"{model}\n{TTS_VOICE}\n{speed!r}\n{instructions_for(model)}\n{text}"
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
        audio = await synthesize_speech(
            client, model, target.surface, speed, instructions=instructions_for(model)
        )
        if has_speech(audio):
            return audio
        logger.warning(
            "event=reader.tts_empty chapter_id=%s paragraph=%d token=%d attempt=%d bytes=%d",
            target.chapter_id,
            target.paragraph,
            target.token,
            attempt,
            len(audio),
        )
    raise EmptySpeechError
