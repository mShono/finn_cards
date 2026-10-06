import logging
from unittest.mock import AsyncMock, MagicMock

from conftest import log_fields

from kielikaveri.tts import TTS_MAX_RETRIES, make_tts_client, synthesize_speech


def make_client() -> MagicMock:
    client = MagicMock()
    response = MagicMock()
    response.aread = AsyncMock(return_value=b"fake-mp3-bytes")
    client.audio.speech.create = AsyncMock(return_value=response)
    return client


async def test_synthesize_speech_passes_model_and_text_and_returns_bytes():
    client = make_client()

    result = await synthesize_speech(client, "tts-1", "Haen töitä.")

    assert result == b"fake-mp3-bytes"
    client.audio.speech.create.assert_awaited_once_with(
        model="tts-1", voice="alloy", input="Haen töitä.", speed=1.0
    )


async def test_synthesize_speech_passes_custom_speed():
    client = make_client()

    await synthesize_speech(client, "tts-1", "Haen töitä.", speed=0.8)

    client.audio.speech.create.assert_awaited_once_with(
        model="tts-1", voice="alloy", input="Haen töitä.", speed=0.8
    )


async def test_synthesize_speech_logs_request_and_response_with_duration(caplog):
    client = make_client()

    with caplog.at_level(logging.DEBUG, logger="kielikaveri.tts"):
        await synthesize_speech(client, "tts-1", "Haen töitä.")

    events = [log_fields(r.message) for r in caplog.records]
    request = next(f for f in events if f.get("event") == "tts.request")
    assert request["model"] == "tts-1"

    response = next(f for f in events if f.get("event") == "tts.response")
    assert response["bytes"] == str(len(b"fake-mp3-bytes"))
    assert int(response["duration_ms"]) >= 0


async def test_the_tts_client_gives_up_quickly_instead_of_using_the_sdk_defaults():
    # SDK defaults are a 600 s read timeout and two retries - a hung OpenAI
    # would keep the user waiting on the listen button for many minutes.
    async with make_tts_client("sk-test", timeout_seconds=15.0) as client:
        assert client.timeout == 15.0
        assert client.max_retries == TTS_MAX_RETRIES <= 1
