"""Tests for voice_agent_core.fish builders.

Real network calls aren't tested here — those require a Fish API key and live
endpoint. We test construction, defaults, and error paths instead.
"""

from __future__ import annotations

import pytest

from voice_agent_core.config import BaseAgentSettings
from voice_agent_core.fish import build_fish_stt, build_fish_tts
from voice_agent_core.fish.tts import _to_ms


class TestBuildFishTTS:
    def test_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FISH_API_KEY", raising=False)
        s = BaseAgentSettings()
        with pytest.raises(ValueError, match="FISH_API_KEY"):
            build_fish_tts(s)

    def test_constructs_with_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FISH_API_KEY", "test-key")
        monkeypatch.delenv("FISH_TTS_OUTPUT_FORMAT", raising=False)
        monkeypatch.delenv("FISH_TTS_SAMPLE_RATE", raising=False)
        s = BaseAgentSettings()
        tts = build_fish_tts(s)
        assert tts.model == "s2-pro"
        assert tts.latency_mode == "balanced"
        # Default flipped to pcm to mitigate the first-phoneme decoder click; pcm/wav
        # both default to 24 kHz on the Fish plugin.
        assert tts.output_format == "pcm"
        assert tts.sample_rate == 24000

    def test_fish_latency_mode_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FISH_API_KEY", "test-key")
        monkeypatch.setenv("FISH_TTS_LATENCY_MODE", "low")
        s = BaseAgentSettings()
        tts = build_fish_tts(s)
        assert tts.latency_mode == "low"

    def test_fish_output_format_revert_to_wav(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The pcm default must stay overridable back to the upstream wav format.
        monkeypatch.setenv("FISH_API_KEY", "test-key")
        monkeypatch.setenv("FISH_TTS_OUTPUT_FORMAT", "wav")
        s = BaseAgentSettings()
        tts = build_fish_tts(s)
        assert tts.output_format == "wav"

    def test_fish_sample_rate_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FISH_API_KEY", "test-key")
        monkeypatch.setenv("FISH_TTS_SAMPLE_RATE", "44100")
        s = BaseAgentSettings()
        tts = build_fish_tts(s)
        assert tts.output_format == "pcm"
        assert tts.sample_rate == 44100

    def test_voice_id_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FISH_API_KEY", "test-key")
        monkeypatch.setenv("TTS_VOICE_ID", "voice-abc")
        s = BaseAgentSettings()
        tts = build_fish_tts(s)
        assert tts.voice_id == "voice-abc"


class TestBuildFishSTT:
    def test_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FISH_API_KEY", raising=False)
        s = BaseAgentSettings()
        with pytest.raises(ValueError, match="FISH_API_KEY"):
            build_fish_stt(s)

    def test_constructs_with_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FISH_API_KEY", "test-key")
        monkeypatch.setenv("STT_LANGUAGE", "auto")
        s = BaseAgentSettings()
        stt = build_fish_stt(s)
        assert stt.provider == "FishAudio"
        assert stt.model == "fish-audio/asr"
        assert stt._opts.language is None

    def test_auto_language_normalized_to_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FISH_API_KEY", "test-key")
        monkeypatch.setenv("STT_LANGUAGE", "auto")
        s = BaseAgentSettings()
        stt = build_fish_stt(s)
        assert stt._opts.language is None

    def test_multi_language_normalized_to_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FISH_API_KEY", "test-key")
        monkeypatch.setenv("STT_LANGUAGE", "multi")
        s = BaseAgentSettings()
        stt = build_fish_stt(s)
        assert stt._opts.language is None

    def test_explicit_language_kept(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FISH_API_KEY", "test-key")
        monkeypatch.setenv("STT_LANGUAGE", "en")
        s = BaseAgentSettings()
        stt = build_fish_stt(s)
        assert stt._opts.language == "en"


class TestToMs:
    def test_positive(self) -> None:
        assert _to_ms(0.150) == 150.0
        assert _to_ms(1.2345) == 1234.5

    def test_zero(self) -> None:
        assert _to_ms(0.0) == 0.0

    def test_negative_returns_none(self) -> None:
        assert _to_ms(-1.0) is None
