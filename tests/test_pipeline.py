"""Tests for voice_agent_core.pipeline.build_pipeline.

We pass a fake VAD object in tests to avoid loading the real silero model
(which downloads PyTorch weights on first run — slow + needs network).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from livekit.agents.stt import STT, SpeechEvent, SpeechEventType, STTCapabilities
from livekit.agents.types import APIConnectOptions

from voice_agent_core.config import BaseAgentSettings
from voice_agent_core.pipeline import PipelineComponents, build_pipeline
from voice_agent_core.stt import StreamAdapter


def _set_required_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Minimum env vars needed for build_pipeline to succeed (livekit backend)."""
    monkeypatch.setenv("FISH_API_KEY", "test-fish-key")
    monkeypatch.setenv("LIVEKIT_API_KEY", "test-lk-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "test-lk-secret")
    monkeypatch.setenv("LLM_PROVIDER", "livekit")
    # Pin STT to fish: the default is now 'deepgram', but these tests exercise
    # pipeline assembly with the keyless fish builder. Pinned like LLM_PROVIDER above.
    monkeypatch.setenv("STT_PROVIDER", "fish")


class _FakeSTT(STT):
    def __init__(self, *, streaming: bool) -> None:
        super().__init__(
            capabilities=STTCapabilities(
                streaming=streaming,
                interim_results=False,
                offline_recognize=True,
            )
        )

    @property
    def model(self) -> str:
        return "fake-stt"

    @property
    def provider(self) -> str:
        return "fake"

    async def _recognize_impl(
        self,
        buffer,
        *,
        language="en",
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        return SpeechEvent(type=SpeechEventType.FINAL_TRANSCRIPT)


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

    def test_default_turn_detection_is_mode_marker_no_job_context(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No turn_detection injected → carries the mode marker string, NOT a
        # constructed MultilingualModel. Proves build_pipeline needs no job context.
        _set_required_env(monkeypatch)
        pipeline = build_pipeline(BaseAgentSettings(), vad=object())
        assert pipeline.turn_detection == "multilingual"

    def test_turn_detection_mode_vad(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_required_env(monkeypatch)
        monkeypatch.setenv("TURN_DETECTION_MODE", "vad")
        pipeline = build_pipeline(BaseAgentSettings(), vad=object())
        assert pipeline.turn_detection == "vad"

    def test_preemptive_generation_defaults_true_on_pipeline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_required_env(monkeypatch)
        pipeline = build_pipeline(BaseAgentSettings(), vad=object())
        assert pipeline.preemptive_generation is True

    def test_preemptive_generation_env_false_carried_to_pipeline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # PREEMPTIVE_GENERATION=false → settings → carried onto the pipeline bundle,
        # which is where build_session later reads it from.
        _set_required_env(monkeypatch)
        monkeypatch.setenv("PREEMPTIVE_GENERATION", "false")
        pipeline = build_pipeline(BaseAgentSettings(), vad=object())
        assert pipeline.preemptive_generation is False

    def test_min_endpointing_delay_defaults_none_on_pipeline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_required_env(monkeypatch)
        pipeline = build_pipeline(BaseAgentSettings(), vad=object())
        assert pipeline.min_endpointing_delay is None

    def test_min_endpointing_delay_env_carried_to_pipeline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_required_env(monkeypatch)
        monkeypatch.setenv("MIN_ENDPOINTING_DELAY", "0.25")
        pipeline = build_pipeline(BaseAgentSettings(), vad=object())
        assert pipeline.min_endpointing_delay == 0.25

    def test_streaming_stt_not_wrapped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Auto: a natively-streaming STT (e.g. Deepgram) is used as-is.
        fake_stt = _FakeSTT(streaming=True)
        with (
            patch("voice_agent_core.pipeline.build_stt", return_value=fake_stt),
            patch("voice_agent_core.pipeline.build_tts", return_value=object()),
            patch("voice_agent_core.pipeline.build_llm", return_value=object()),
        ):
            pipeline = build_pipeline(BaseAgentSettings(), vad=object())

        assert pipeline.stt is fake_stt

    def test_non_streaming_stt_auto_wrapped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Auto: a non-streaming STT (Fish batch) is wrapped in a StreamAdapter — no flag.
        fake_stt = _FakeSTT(streaming=False)
        fake_vad = object()
        stream_adapter_vad = object()

        with (
            patch("voice_agent_core.pipeline.build_stt", return_value=fake_stt),
            patch("voice_agent_core.pipeline.build_tts", return_value=object()),
            patch("voice_agent_core.pipeline.build_llm", return_value=object()),
        ):
            pipeline = build_pipeline(
                BaseAgentSettings(),
                vad=fake_vad,
                stream_adapter_vad=stream_adapter_vad,
            )

        assert isinstance(pipeline.stt, StreamAdapter)
        assert pipeline.stt.wrapped_stt is fake_stt
        assert pipeline.stt.vad is stream_adapter_vad
        assert pipeline.stt.capabilities.streaming is True
        assert pipeline.vad is fake_vad

    def test_stream_adapt_override_false_skips_wrap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Explicit stream_adapt=False forces no wrap even for a non-streaming STT.
        fake_stt = _FakeSTT(streaming=False)
        with (
            patch("voice_agent_core.pipeline.build_stt", return_value=fake_stt),
            patch("voice_agent_core.pipeline.build_tts", return_value=object()),
            patch("voice_agent_core.pipeline.build_llm", return_value=object()),
        ):
            pipeline = build_pipeline(
                BaseAgentSettings(), vad=object(), stream_adapt=False
            )

        assert pipeline.stt is fake_stt

    def test_propagates_fish_key_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # LIVEKIT_* set but FISH_API_KEY missing → build_fish_stt raises.
        # Pin STT_PROVIDER=fish since the default is now deepgram.
        monkeypatch.setenv("STT_PROVIDER", "fish")
        monkeypatch.setenv("LIVEKIT_API_KEY", "test-lk-key")
        monkeypatch.setenv("LIVEKIT_API_SECRET", "test-lk-secret")
        monkeypatch.delenv("FISH_API_KEY", raising=False)

        with pytest.raises(ValueError, match="FISH_API_KEY"):
            build_pipeline(BaseAgentSettings(), vad=object(), turn_detection=object())

    def test_propagates_llm_credential_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # FISH ok but LIVEKIT creds missing → build_llm raises. Pin STT to fish so
        # STT (built before LLM) doesn't raise the deepgram-key error first.
        monkeypatch.setenv("STT_PROVIDER", "fish")
        monkeypatch.setenv("FISH_API_KEY", "test-fish-key")
        monkeypatch.setenv("LLM_PROVIDER", "livekit")
        monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
        monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)

        with pytest.raises(ValueError, match="LIVEKIT_API"):
            build_pipeline(BaseAgentSettings(), vad=object(), turn_detection=object())
