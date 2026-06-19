"""Tests for voice_agent_core.pipeline.build_pipeline.

We pass a fake VAD object in tests to avoid loading the real silero model
(which downloads PyTorch weights on first run — slow + needs network).
"""

from __future__ import annotations

import pytest

from voice_agent_core.config import BaseAgentSettings
from voice_agent_core.pipeline import PipelineComponents, build_pipeline


def _set_required_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Minimum env vars needed for build_pipeline to succeed (livekit backend)."""
    monkeypatch.setenv("FISH_API_KEY", "test-fish-key")
    monkeypatch.setenv("LIVEKIT_API_KEY", "test-lk-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "test-lk-secret")
    monkeypatch.setenv("LLM_PROVIDER", "livekit")


class TestBuildPipeline:
    def test_assembles_all_components(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_required_env(monkeypatch)
        fake_vad = object()
        fake_td = object()

        pipeline = build_pipeline(
            BaseAgentSettings(),
            vad=fake_vad,
            turn_detection=fake_td,
        )

        assert isinstance(pipeline, PipelineComponents)
        assert pipeline.stt is not None
        assert pipeline.tts is not None
        assert pipeline.llm is not None
        assert pipeline.vad is fake_vad
        assert pipeline.turn_detection is fake_td

    def test_propagates_fish_key_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # LIVEKIT_* set but FISH_API_KEY missing → build_fish_stt raises
        monkeypatch.setenv("LIVEKIT_API_KEY", "test-lk-key")
        monkeypatch.setenv("LIVEKIT_API_SECRET", "test-lk-secret")
        monkeypatch.delenv("FISH_API_KEY", raising=False)

        with pytest.raises(ValueError, match="FISH_API_KEY"):
            build_pipeline(BaseAgentSettings(), vad=object(), turn_detection=object())

    def test_propagates_llm_credential_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # FISH ok but LIVEKIT creds missing → build_llm raises
        monkeypatch.setenv("FISH_API_KEY", "test-fish-key")
        monkeypatch.setenv("LLM_PROVIDER", "livekit")
        monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
        monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)

        with pytest.raises(ValueError, match="LIVEKIT_API"):
            build_pipeline(BaseAgentSettings(), vad=object(), turn_detection=object())
