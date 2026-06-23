"""Tests for the runtime helpers (build_session, default_prewarm, etc.)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from voice_agent_core.runtime import (
    build_session,
    default_prewarm,
    default_room_options,
    is_warmup_session,
    warm_tts,
)


class TestIsWarmupSession:
    """Boundary cases for the room-name prefix check."""

    def _ctx(self, room_name: str) -> SimpleNamespace:
        return SimpleNamespace(room=SimpleNamespace(name=room_name))

    def test_warmup_prefix_returns_true(self) -> None:
        assert is_warmup_session(self._ctx("warmup-abc12345")) is True

    def test_real_room_returns_false(self) -> None:
        assert is_warmup_session(self._ctx("lead-qual-abc12345")) is False

    def test_empty_room_name_returns_false(self) -> None:
        assert is_warmup_session(self._ctx("")) is False

    def test_substring_match_not_prefix_returns_false(self) -> None:
        # "warmup-" appearing in the middle must NOT trigger
        assert is_warmup_session(self._ctx("real-warmup-1234")) is False

    def test_just_the_prefix_with_no_id_still_matches(self) -> None:
        # Defensive: even a bare "warmup-" counts as warmup (defensive default)
        assert is_warmup_session(self._ctx("warmup-")) is True


class TestDefaultPrewarm:
    """default_prewarm loads the main VAD always, the adapter VAD only on opt-in."""

    def test_loads_only_main_vad_by_default(self) -> None:
        # Streaming-STT path (default) must NOT pay for a second ONNX session.
        proc = SimpleNamespace(userdata={})
        main_vad = object()
        with patch(
            "voice_agent_core.runtime.silero.VAD.load", side_effect=[main_vad]
        ) as load:
            default_prewarm(proc)

        assert proc.userdata["vad"] is main_vad
        assert "stream_adapter_vad" not in proc.userdata
        assert load.call_count == 1
        # Main VAD carries its conservative tuning.
        assert load.call_args.kwargs["min_silence_duration"] == 0.5
        assert load.call_args.kwargs["prefix_padding_duration"] == 0.0

    def test_loads_both_when_stream_adapter_vad_opted_in(self) -> None:
        proc = SimpleNamespace(userdata={})
        main_vad = object()
        adapter_vad = object()
        with patch(
            "voice_agent_core.runtime.silero.VAD.load",
            side_effect=[main_vad, adapter_vad],
        ) as load:
            default_prewarm(proc, stream_adapter_vad=True)

        assert proc.userdata["vad"] is main_vad
        assert proc.userdata["stream_adapter_vad"] is adapter_vad
        # Adapter VAD is the aggressive profile (short silence + prefix padding).
        assert load.call_count == 2
        second = load.call_args_list[1]
        assert second.kwargs["min_silence_duration"] == 0.35
        assert second.kwargs["prefix_padding_duration"] == 0.35


class TestDefaultRoomOptions:
    """default_room_options should hand back RoomOptions with AI Coustics on input."""

    def test_returns_room_options_with_noise_cancellation(self) -> None:
        opts = default_room_options()
        # Don't assert on internal attribute names of the AI Coustics object —
        # just that audio_input has SOMETHING set as noise_cancellation, since
        # the upstream type is the contract we depend on.
        assert opts.audio_input is not None
        assert opts.audio_input.noise_cancellation is not None


class TestBuildSession:
    """build_session should pass the pipeline's components into AgentSession
    with sane defaults (preemptive_generation=True, turn_handling wrapper)."""

    def _fake_pipeline(
        self,
        preemptive_generation: bool = True,
        min_endpointing_delay: float | None = None,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            stt=MagicMock(name="stt"),
            tts=MagicMock(name="tts"),
            llm=MagicMock(name="llm"),
            vad=MagicMock(name="vad"),
            turn_detection=MagicMock(name="turn_detection"),
            preemptive_generation=preemptive_generation,
            min_endpointing_delay=min_endpointing_delay,
        )

    def test_passes_pipeline_components_through(self) -> None:
        pipeline = self._fake_pipeline()
        with patch("voice_agent_core.runtime.AgentSession") as agent_session_cls:
            build_session(pipeline)
        kwargs = agent_session_cls.call_args.kwargs
        assert kwargs["stt"] is pipeline.stt
        assert kwargs["tts"] is pipeline.tts
        assert kwargs["llm"] is pipeline.llm
        assert kwargs["vad"] is pipeline.vad

    def test_wraps_turn_detection_in_turn_handling_options(self) -> None:
        pipeline = self._fake_pipeline()
        with patch("voice_agent_core.runtime.AgentSession") as agent_session_cls:
            build_session(pipeline)
        kwargs = agent_session_cls.call_args.kwargs
        # TurnHandlingOptions is a TypedDict (dict subclass) in livekit-agents 1.5.x,
        # so use dict access. The contract is just "turn_detection lands here."
        assert kwargs["turn_handling"]["turn_detection"] is pipeline.turn_detection

    def test_preemptive_generation_defaults_true(self) -> None:
        """preemptive_generation lives inside TurnHandlingOptions in livekit-agents
        v1.5+ (passing it directly to AgentSession was deprecated, removed in v2).
        With no explicit arg, build_session reads it off the pipeline; the default
        pipeline carries True so existing demos don't need to change their calls."""
        pipeline = self._fake_pipeline()
        with patch("voice_agent_core.runtime.AgentSession") as agent_session_cls:
            build_session(pipeline)
        kwargs = agent_session_cls.call_args.kwargs
        assert "preemptive_generation" not in kwargs  # NOT at AgentSession level
        assert kwargs["turn_handling"]["preemptive_generation"]["enabled"] is True

    def test_preemptive_generation_can_be_disabled(self) -> None:
        pipeline = self._fake_pipeline()
        with patch("voice_agent_core.runtime.AgentSession") as agent_session_cls:
            build_session(pipeline, preemptive_generation=False)
        kwargs = agent_session_cls.call_args.kwargs
        assert kwargs["turn_handling"]["preemptive_generation"]["enabled"] is False

    def test_preemptive_generation_flows_from_pipeline(self) -> None:
        """settings → pipeline.preemptive_generation → session, with no explicit arg.
        A pipeline carrying False (e.g. PREEMPTIVE_GENERATION=false) disables it."""
        pipeline = self._fake_pipeline(preemptive_generation=False)
        with patch("voice_agent_core.runtime.AgentSession") as agent_session_cls:
            build_session(pipeline)  # no explicit arg → falls back to the pipeline
        kwargs = agent_session_cls.call_args.kwargs
        assert kwargs["turn_handling"]["preemptive_generation"]["enabled"] is False

    def test_explicit_arg_overrides_pipeline_value(self) -> None:
        """Explicit preemptive_generation=True must win even when the pipeline
        (settings/env) carries False. Guards the `is not None` sentinel: a plain
        truthiness check would also let an explicit False fall through, so test both
        directions of override."""
        pipeline = self._fake_pipeline(preemptive_generation=False)
        with patch("voice_agent_core.runtime.AgentSession") as agent_session_cls:
            build_session(pipeline, preemptive_generation=True)
        kwargs = agent_session_cls.call_args.kwargs
        assert kwargs["turn_handling"]["preemptive_generation"]["enabled"] is True

        # ...and the reverse: explicit False over a pipeline carrying True.
        pipeline = self._fake_pipeline(preemptive_generation=True)
        with patch("voice_agent_core.runtime.AgentSession") as agent_session_cls:
            build_session(pipeline, preemptive_generation=False)
        kwargs = agent_session_cls.call_args.kwargs
        assert kwargs["turn_handling"]["preemptive_generation"]["enabled"] is False

    def test_explicit_turn_handling_override_wins(self) -> None:
        """A full turn_handling=... override replaces the wrapper entirely, beating
        both the explicit preemptive_generation arg and the pipeline value."""
        pipeline = self._fake_pipeline(preemptive_generation=True)
        sentinel = object()
        with patch("voice_agent_core.runtime.AgentSession") as agent_session_cls:
            build_session(
                pipeline,
                preemptive_generation=True,
                turn_handling=sentinel,
            )
        kwargs = agent_session_cls.call_args.kwargs
        assert kwargs["turn_handling"] is sentinel

    def test_min_endpointing_delay_mode_aware_default_when_unset(self) -> None:
        """No explicit arg and pipeline carries None → fall back to the mode-aware
        default (0 for non-vad, 0.5 for vad)."""
        pipeline = self._fake_pipeline()  # min_endpointing_delay=None
        pipeline.turn_detection = "stt"
        with patch("voice_agent_core.runtime.AgentSession") as agent_session_cls:
            build_session(pipeline)
        kwargs = agent_session_cls.call_args.kwargs
        assert "min_endpointing_delay" not in kwargs
        assert kwargs["turn_handling"]["endpointing"]["min_delay"] == 0

        pipeline = self._fake_pipeline()
        pipeline.turn_detection = "vad"
        with patch("voice_agent_core.runtime.AgentSession") as agent_session_cls:
            build_session(pipeline)
        kwargs = agent_session_cls.call_args.kwargs
        assert "min_endpointing_delay" not in kwargs
        assert kwargs["turn_handling"]["endpointing"]["min_delay"] == 0.5

    def test_min_endpointing_delay_flows_from_pipeline_over_mode_default(self) -> None:
        """A value carried on the pipeline (settings/env) overrides the mode-aware
        default — including forcing 0.0 on vad mode, since 0.0 is not None."""
        pipeline = self._fake_pipeline(min_endpointing_delay=0.2)
        pipeline.turn_detection = "vad"  # mode default would be 0.5
        with patch("voice_agent_core.runtime.AgentSession") as agent_session_cls:
            build_session(pipeline)
        kwargs = agent_session_cls.call_args.kwargs
        assert "min_endpointing_delay" not in kwargs
        assert kwargs["turn_handling"]["endpointing"]["min_delay"] == 0.2

        pipeline = self._fake_pipeline(min_endpointing_delay=0.0)
        pipeline.turn_detection = "vad"
        with patch("voice_agent_core.runtime.AgentSession") as agent_session_cls:
            build_session(pipeline)
        kwargs = agent_session_cls.call_args.kwargs
        assert "min_endpointing_delay" not in kwargs
        assert kwargs["turn_handling"]["endpointing"]["min_delay"] == 0.0

    def test_explicit_min_endpointing_delay_overrides_pipeline(self) -> None:
        """Explicit arg beats both the pipeline value and the mode-aware default."""
        pipeline = self._fake_pipeline(min_endpointing_delay=0.2)
        pipeline.turn_detection = "vad"
        with patch("voice_agent_core.runtime.AgentSession") as agent_session_cls:
            build_session(pipeline, min_endpointing_delay=0.9)
        kwargs = agent_session_cls.call_args.kwargs
        assert "min_endpointing_delay" not in kwargs
        assert kwargs["turn_handling"]["endpointing"]["min_delay"] == 0.9

    def test_extra_kwargs_passed_through(self) -> None:
        pipeline = self._fake_pipeline()
        with patch("voice_agent_core.runtime.AgentSession") as agent_session_cls:
            build_session(pipeline, allow_interruptions=False)
        assert agent_session_cls.call_args.kwargs["allow_interruptions"] is False

    def test_multilingual_marker_resolved_in_context(self) -> None:
        # "multilingual" marker → MultilingualModel() constructed here (build_session
        # runs in a job context); transformer mode keeps min_endpointing_delay=0.
        pipeline = self._fake_pipeline()
        pipeline.turn_detection = "multilingual"
        fake_model = object()
        with patch(
            "livekit.plugins.turn_detector.multilingual.MultilingualModel",
            return_value=fake_model,
        ), patch("voice_agent_core.runtime.AgentSession") as agent_session_cls:
            build_session(pipeline)
        kwargs = agent_session_cls.call_args.kwargs
        assert kwargs["turn_handling"]["turn_detection"] is fake_model
        assert "min_endpointing_delay" not in kwargs
        assert kwargs["turn_handling"]["endpointing"]["min_delay"] == 0

    def test_vad_marker_passthrough_with_buffer(self) -> None:
        # "vad" passes straight through; VAD-only needs the 0.5s silence buffer.
        pipeline = self._fake_pipeline()
        pipeline.turn_detection = "vad"
        with patch("voice_agent_core.runtime.AgentSession") as agent_session_cls:
            build_session(pipeline)
        kwargs = agent_session_cls.call_args.kwargs
        assert kwargs["turn_handling"]["turn_detection"] == "vad"
        assert "min_endpointing_delay" not in kwargs
        assert kwargs["turn_handling"]["endpointing"]["min_delay"] == 0.5


class TestWarmTts:
    """warm_tts is kept in the public API for future use with TTS providers
    that pool WebSocket connections (Fish currently does not — see warm_tts
    docstring for context). Tests still exercise the helper since it's
    public; demos no longer call it."""

    def _stream(self) -> MagicMock:
        """Fake the async-iterable stream returned by tts.stream()."""
        stream = MagicMock(name="stream")
        stream.push_text = MagicMock()
        stream.end_input = MagicMock()
        # async iteration support — yields nothing, terminates immediately
        stream.__aiter__ = lambda self: self
        stream.__anext__ = AsyncMock(side_effect=StopAsyncIteration())
        stream.aclose = AsyncMock()
        return stream

    @pytest.mark.asyncio
    async def test_calls_streaming_path(self) -> None:
        """Default text was changed from "." to "hi" because Fish (and likely
        other neural TTS) generates pathological amounts of silence audio
        for a lone period — see warm_tts docstring."""
        stream = self._stream()
        tts = MagicMock()
        tts.stream = MagicMock(return_value=stream)
        await warm_tts(tts)
        tts.stream.assert_called_once_with()
        stream.push_text.assert_called_once_with("hi")
        stream.end_input.assert_called_once_with()
        stream.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_custom_text(self) -> None:
        stream = self._stream()
        tts = MagicMock()
        tts.stream = MagicMock(return_value=stream)
        await warm_tts(tts, text="warming up")
        stream.push_text.assert_called_once_with("warming up")

    @pytest.mark.asyncio
    async def test_swallows_provider_failure(self) -> None:
        """If TTS provider is unreachable, warmup must not propagate the error
        — the real synth path is what surfaces failures to the user."""
        tts = MagicMock()
        tts.stream = MagicMock(side_effect=RuntimeError("fish unreachable"))
        # Should NOT raise
        await warm_tts(tts)

    @pytest.mark.asyncio
    async def test_closes_stream_even_on_iteration_error(self) -> None:
        """aclose runs in finally so a flaky iteration still releases the
        connection back to the pool."""
        stream = self._stream()
        stream.__anext__ = AsyncMock(side_effect=RuntimeError("mid-stream boom"))
        tts = MagicMock()
        tts.stream = MagicMock(return_value=stream)
        await warm_tts(tts)  # caught at the outer try
        stream.aclose.assert_awaited_once()
