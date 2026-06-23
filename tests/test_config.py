"""Tests for voice_agent_core.config."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import dotenv_values
from pydantic import ValidationError

from voice_agent_core.config import (
    BaseAgentSettings,
    load_env_walking_up,
    load_yaml,
)


class TestLoadYaml:
    def test_loads_valid_yaml(self, tmp_path: Path) -> None:
        p = tmp_path / "cfg.yaml"
        p.write_text("name: test\ncount: 3\nflags:\n  - a\n  - b\n")
        result = load_yaml(p)
        assert result == {"name": "test", "count": 3, "flags": ["a", "b"]}

    def test_empty_file_returns_empty_dict(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.yaml"
        p.write_text("")
        assert load_yaml(p) == {}

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_yaml(tmp_path / "nope.yaml")

    def test_non_mapping_root_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "list.yaml"
        p.write_text("- a\n- b\n")
        with pytest.raises(ValueError, match="Expected YAML mapping"):
            load_yaml(p)


class TestLoadEnvWalkingUp:
    def test_finds_env_in_current_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        env = tmp_path / ".env"
        env.write_text("TEST_VAR_XYZ=hello\n")
        monkeypatch.delenv("TEST_VAR_XYZ", raising=False)

        found = load_env_walking_up(start=tmp_path)
        assert found == env
        assert os.environ["TEST_VAR_XYZ"] == "hello"

    def test_finds_env_in_parent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        env = tmp_path / ".env"
        env.write_text("TEST_VAR_PARENT=world\n")
        sub = tmp_path / "a" / "b" / "c"
        sub.mkdir(parents=True)
        monkeypatch.delenv("TEST_VAR_PARENT", raising=False)

        found = load_env_walking_up(start=sub)
        assert found == env
        assert os.environ["TEST_VAR_PARENT"] == "world"

    def test_returns_none_when_no_env_anywhere(self, tmp_path: Path) -> None:
        sub = tmp_path / "a"
        sub.mkdir()
        # tmp_path has no .env; walking up from sub won't find one in tmp_path either
        # but might find one further up on the dev's machine. We can't fully isolate
        # filesystem walks in a test, so this is a best-effort smoke check.
        # The key contract: it returns None if no .env is in tmp_path or sub.
        result = load_env_walking_up(start=sub, name=".env-test-nonexistent-xyz")
        assert result is None


class TestBaseAgentSettings:
    def test_env_example_does_not_parse_comments_as_values(self) -> None:
        root = Path(__file__).resolve().parents[1]
        values = dotenv_values(root / "examples" / ".env.example")
        offenders = {
            key: value
            for key, value in values.items()
            if isinstance(value, str) and value.strip().startswith("#")
        }
        assert offenders == {}

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Clear any leaked env vars
        for key in list(os.environ):
            if key.upper().startswith(
                ("LIVEKIT_", "FISH_", "STT_", "TTS_", "LLM_", "OPENROUTER_", "OTEL_",
                 "LOG_", "TURN_", "PREEMPTIVE_", "MIN_")
            ):
                monkeypatch.delenv(key, raising=False)

        s = BaseAgentSettings()
        assert s.stt_provider == "deepgram"
        assert s.stt_language == "en"
        assert s.tts_provider == "fish"
        assert s.llm_provider == "openrouter"
        assert s.llm_model == "openai/gpt-5.4-mini"
        assert s.tts_model == "s2-pro"
        assert s.tts_voice == ""
        # Provider-specific knobs (fish_tts_*) + credentials moved off BaseAgentSettings
        # to provider settings classes (FishSettings/DeepgramSettings/OpenRouterSettings;
        # covered in test_providers).
        assert not hasattr(s, "fish_tts_latency_mode")
        assert not hasattr(s, "fish_api_key")
        assert not hasattr(s, "stt_stream_adapt")
        assert s.turn_detection_mode == "multilingual"
        assert s.preemptive_generation is True
        assert s.min_endpointing_delay is None
        assert s.log_level == "INFO"
        assert s.log_format == "json"
        assert s.otel_metrics_exporter == "console"

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LIVEKIT_URL", "wss://test.livekit.cloud")
        monkeypatch.setenv("TTS_VOICE", "voice-123")
        monkeypatch.setenv("LLM_PROVIDER", "openrouter")
        monkeypatch.setenv("PREEMPTIVE_GENERATION", "false")
        monkeypatch.setenv("MIN_ENDPOINTING_DELAY", "0.3")

        s = BaseAgentSettings()
        assert s.livekit_url == "wss://test.livekit.cloud"
        assert s.tts_voice == "voice-123"
        assert s.llm_provider == "openrouter"
        assert s.preemptive_generation is False
        assert s.min_endpointing_delay == 0.3

    def test_min_endpointing_delay_rejects_negative(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ge=0 constraint: a negative delay is nonsensical and must fail config load
        # loudly rather than silently clamping.
        monkeypatch.setenv("MIN_ENDPOINTING_DELAY", "-0.1")
        with pytest.raises(ValidationError):
            BaseAgentSettings()

    def test_min_endpointing_delay_accepts_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 0 is valid (force "no delay"), distinct from None (mode-aware default).
        monkeypatch.setenv("MIN_ENDPOINTING_DELAY", "0")
        s = BaseAgentSettings()
        assert s.min_endpointing_delay == 0.0
