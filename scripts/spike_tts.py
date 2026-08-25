"""Spike: is OpenAI TTS's Finnish good enough to use? (plan 3.7)

OpenAI doesn't officially list Finnish among its TTS languages - this is
the "five minutes without Azure" rough check from the plan: synthesize each
phrase, feed the audio straight back into OpenAI STT, and see if the model
can even recognize its own synthesis. A mismatch is a strong "bad enough to
skip Azure entirely and fail this spike" signal. A match is NOT proof the
pronunciation is good - it only proves the words were intelligible, not
that stress/rhythm/vowel length sound native. Listen to the saved .mp3
files yourself (or run them through Azure Pronunciation Assessment,
fi-FI) before actually deciding OpenAI vs Azure.

One-off - delete this file once the TTS provider decision (plan 3.7) is made.

Usage: uv run python scripts/spike_tts.py
Requires OPENAI_API_KEY in .env and a small amount of real spend
(a few cents: 3 short TTS + STT calls).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from openai import OpenAI

from kielikaveri.config import load_settings

# Pulled from cards/examples/*.json - already FST-validated real Finnish,
# not invented for this script.
PHRASES = [
    "Haen töitä kaupungista.",
    "Minulla on kipeä hammas.",
    "Pidän suomen kielestä.",
]

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "tts_spike_output"


def normalize(text: str) -> str:
    return re.sub(r"[^\wäöå]+", " ", text.lower()).strip()


def main() -> None:
    settings = load_settings()
    if not settings.openai_api_key:
        sys.exit("OPENAI_API_KEY is not set - see .env.example")

    client = OpenAI(api_key=settings.openai_api_key)
    OUTPUT_DIR.mkdir(exist_ok=True)
    matched = 0

    for i, phrase in enumerate(PHRASES):
        audio_path = OUTPUT_DIR / f"{i}.mp3"

        speech = client.audio.speech.create(
            model=settings.openai_tts_model, voice="alloy", input=phrase
        )
        speech.write_to_file(audio_path)

        with audio_path.open("rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model=settings.openai_stt_model, file=audio_file, language="fi"
            ).text

        ok = normalize(transcript) == normalize(phrase)
        matched += ok
        print(f"{'OK  ' if ok else 'FAIL'} {phrase!r} -> {transcript!r}  ({audio_path})")

    print(f"\n{matched}/{len(PHRASES)} round-tripped exactly.")
    print(f"Audio saved to {OUTPUT_DIR}/ - listen yourself for the real call.")


if __name__ == "__main__":
    main()
