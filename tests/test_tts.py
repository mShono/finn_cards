import logging
from unittest.mock import MagicMock

from conftest import log_fields

from kielikaveri.tts import synthesize_speech


def test_synthesize_speech_passes_model_and_text_and_returns_bytes():
    client = MagicMock()
    client.audio.speech.create.return_value.read.return_value = b"fake-mp3-bytes"

    result = synthesize_speech(client, "tts-1", "Haen töitä.")

    assert result == b"fake-mp3-bytes"
    client.audio.speech.create.assert_called_once_with(
        model="tts-1", voice="alloy", input="Haen töitä.", speed=1.0
    )


def test_synthesize_speech_passes_custom_speed():
    client = MagicMock()
    client.audio.speech.create.return_value.read.return_value = b"fake-mp3-bytes"

    synthesize_speech(client, "tts-1", "Haen töitä.", speed=0.8)

    client.audio.speech.create.assert_called_once_with(
        model="tts-1", voice="alloy", input="Haen töitä.", speed=0.8
    )


def test_synthesize_speech_logs_request_and_response_with_duration(caplog):
    client = MagicMock()
    client.audio.speech.create.return_value.read.return_value = b"fake-mp3-bytes"

    with caplog.at_level(logging.DEBUG, logger="kielikaveri.tts"):
        synthesize_speech(client, "tts-1", "Haen töitä.")

    events = [log_fields(r.message) for r in caplog.records]
    request = next(f for f in events if f.get("event") == "tts.request")
    assert request["model"] == "tts-1"

    response = next(f for f in events if f.get("event") == "tts.response")
    assert response["bytes"] == str(len(b"fake-mp3-bytes"))
    assert int(response["duration_ms"]) >= 0
