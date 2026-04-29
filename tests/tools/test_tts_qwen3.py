"""Tests for the Qwen3 TTS provider in tools.tts_tool.py."""

import io
import json
import urllib.error
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in (
        "QWEN3_TTS_BASE_URL",
        "QWEN3_TTS_HOST",
        "HERMES_SESSION_PLATFORM",
    ):
        monkeypatch.delenv(key, raising=False)


class _FakeResponse:
    def __init__(self, payload: bytes = b"audio-bytes"):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self.payload


class TestQwen3TtsConfig:
    def test_normalize_base_url(self):
        from tools.tts_tool import _normalize_qwen3_tts_base_url

        assert _normalize_qwen3_tts_base_url("127.0.0.1:8000") == "http://127.0.0.1:8000/v1"
        assert _normalize_qwen3_tts_base_url("https://tts.example/v1") == "https://tts.example/v1"
        assert (
            _normalize_qwen3_tts_base_url("https://tts.example/v1/audio/speech")
            == "https://tts.example/v1"
        )

    def test_available_from_config_or_env(self, monkeypatch):
        from tools.tts_tool import _check_qwen3_tts_available

        assert _check_qwen3_tts_available({}) is False
        assert _check_qwen3_tts_available({"qwen3": {"base_url": "https://tts.example/v1"}}) is True
        monkeypatch.setenv("QWEN3_TTS_HOST", "127.0.0.1:8000")
        assert _check_qwen3_tts_available({}) is True


class TestGenerateQwen3Tts:
    def test_successful_request_uses_openai_compatible_endpoint(self, tmp_path):
        from tools.tts_tool import _generate_qwen3_tts

        output_path = str(tmp_path / "speech.ogg")
        config = {
            "qwen3": {
                "base_url": "https://tts.example/v1",
                "model": "qwen3-tts-base",
                "voice": "ono_anna",
                "speed": 1.25,
                "language": "English",
                "instructions": "Speak clearly.",
            }
        }

        with patch("urllib.request.urlopen", return_value=_FakeResponse()) as mock_urlopen:
            result = _generate_qwen3_tts("hello", output_path, config)

        assert result == output_path
        assert (tmp_path / "speech.ogg").read_bytes() == b"audio-bytes"
        request = mock_urlopen.call_args.args[0]
        assert request.full_url == "https://tts.example/v1/audio/speech"
        payload = json.loads(request.data.decode("utf-8"))
        assert payload == {
            "model": "qwen3-tts-base",
            "input": "hello",
            "voice": "ono_anna",
            "response_format": "opus",
            "speed": 1.25,
            "language": "English",
            "instructions": "Speak clearly.",
        }

    def test_voice_error_falls_back_to_default_voice(self, tmp_path):
        from tools.tts_tool import DEFAULT_QWEN3_VOICE, _generate_qwen3_tts

        output_path = str(tmp_path / "speech.mp3")
        http_error = urllib.error.HTTPError(
            url="https://tts.example/v1/audio/speech",
            code=400,
            msg="bad request",
            hdrs=None,
            fp=io.BytesIO(b"Unsupported speakers: custom_voice"),
        )

        with patch("urllib.request.urlopen", side_effect=[http_error, _FakeResponse(b"fallback")]) as mock_urlopen:
            _generate_qwen3_tts(
                "hello",
                output_path,
                {"qwen3": {"base_url": "https://tts.example/v1", "voice": "custom_voice"}},
            )

        assert (tmp_path / "speech.mp3").read_bytes() == b"fallback"
        second_request = mock_urlopen.call_args_list[1].args[0]
        second_payload = json.loads(second_request.data.decode("utf-8"))
        assert second_payload["voice"] == DEFAULT_QWEN3_VOICE
        assert second_payload["response_format"] == "mp3"
