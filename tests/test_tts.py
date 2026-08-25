from unittest.mock import MagicMock

from kielikaveri.tts import synthesize_speech


def test_synthesize_speech_passes_model_and_text_and_returns_bytes():
    client = MagicMock()
    client.audio.speech.create.return_value.read.return_value = b"fake-mp3-bytes"

    result = synthesize_speech(client, "tts-1", "Haen töitä.")

    assert result == b"fake-mp3-bytes"
    client.audio.speech.create.assert_called_once_with(
        model="tts-1", voice="alloy", input="Haen töitä."
    )
