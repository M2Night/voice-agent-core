"""Instrumented Fish Audio TTS for LiveKit Agents.

Subclasses ``livekit.plugins.fishaudio.TTS`` and adds:

- **Metrics** — emits OTEL histograms for TTFB, RTF, and counter for errors
- **Structured logging** — every synthesis logs via structlog with stable key names
- **Stream-level instrumentation** — measures stream-open→first-text and
  stream-open→first-audio latencies in addition to the plugin-emitted TTFB
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from livekit.agents import APIConnectOptions, tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.plugins import fishaudio

from voice_agent_core.observability import MetricNames, get_logger, get_meter

if TYPE_CHECKING:
    from voice_agent_core.config import BaseAgentSettings

log = get_logger(__name__)
_meter = get_meter("voice_agent_core.fish.tts")

_h_ttfb = _meter.create_histogram(
    MetricNames.FISH_TTS_TTFB_MS,
    unit="ms",
    description="Fish TTS time-to-first-byte",
)
_h_rtf = _meter.create_histogram(
    MetricNames.FISH_TTS_RTF,
    unit="ratio",
    description="Fish TTS real-time factor (synth_duration / audio_duration). <1.0 means realtime.",
)
_h_stream_to_audio = _meter.create_histogram(
    MetricNames.FISH_TTS_TTFT_MS,
    unit="ms",
    description="Stream-open to first-audio-frame latency (perceived user latency)",
)
_c_errors = _meter.create_counter(
    MetricNames.FISH_TTS_ERRORS,
    description="Fish TTS error count",
)


class FishTTS(fishaudio.TTS):
    """Fish Audio TTS with built-in metrics + structured logging.

    Use :func:`build_fish_tts` to construct from a settings object, or pass kwargs
    directly to this constructor — the constructor signature matches the upstream
    ``fishaudio.TTS``.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.on("metrics_collected", self._on_metrics)
        self.on("error", self._on_error)
        log.info(
            "fish_tts.ready",
            provider=self.provider,
            model=self.model,
            voice_id=self.voice_id,
            latency_mode=self.latency_mode,
        )

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        log.debug("fish_tts.synthesize", chars=len(text))
        return super().synthesize(text, conn_options=conn_options)

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.SynthesizeStream:
        return _InstrumentedStream(
            inner=super().stream(conn_options=conn_options),
            owner=self,
        )

    def _attrs(self) -> dict[str, Any]:
        """OTEL metric attribute dict — keep cardinality low."""
        return {"voice_id": self.voice_id, "model": str(self.model)}

    def _on_metrics(self, m: Any) -> None:
        audio_ms = _to_ms(getattr(m, "audio_duration", 0.0))
        total_ms = _to_ms(getattr(m, "duration", 0.0))
        ttfb_ms = _to_ms(getattr(m, "ttfb", -1.0))
        rtf = (total_ms / audio_ms) if audio_ms else None

        attrs = self._attrs()
        if ttfb_ms is not None:
            _h_ttfb.record(ttfb_ms, attributes=attrs)
        if rtf is not None:
            _h_rtf.record(rtf, attributes=attrs)

        log.info(
            "fish_tts.metrics",
            request_id=getattr(m, "request_id", None),
            segment_id=getattr(m, "segment_id", None),
            chars=getattr(m, "characters_count", None),
            ttfb_ms=ttfb_ms,
            audio_ms=audio_ms,
            total_ms=total_ms,
            rtf=rtf,
            streamed=getattr(m, "streamed", None),
            cancelled=getattr(m, "cancelled", None),
            connection_reused=getattr(m, "connection_reused", None),
        )

    def _on_error(self, err: Any) -> None:
        _c_errors.add(1, attributes=self._attrs())
        log.error(
            "fish_tts.error",
            error=repr(getattr(err, "error", err)),
            recoverable=getattr(err, "recoverable", None),
        )


class _InstrumentedStream:
    """Wraps a SynthesizeStream to measure first-text and first-audio timings.

    Forwards all unknown attributes/methods to the wrapped stream via __getattr__,
    so it stays a drop-in replacement even if the upstream interface grows.
    """

    def __init__(self, *, inner: tts.SynthesizeStream, owner: FishTTS) -> None:
        self._inner = inner
        self._owner = owner
        self._opened_at = time.perf_counter()
        self._first_text_at: float | None = None
        self._first_audio_logged = False
        self._chars_buffered = 0

    def push_text(self, token: str) -> None:
        if token and self._first_text_at is None:
            self._first_text_at = time.perf_counter()
            log.debug(
                "fish_tts.first_text_token",
                stream_open_to_first_token_ms=_to_ms(
                    self._first_text_at - self._opened_at
                ),
            )
        if token and not self._first_audio_logged:
            self._chars_buffered += len(token)
        self._inner.push_text(token)

    def flush(self) -> None:
        self._inner.flush()

    def end_input(self) -> None:
        self._inner.end_input()

    async def aclose(self) -> None:
        await self._inner.aclose()

    async def __aenter__(self) -> _InstrumentedStream:
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._inner.__aexit__(*args)

    def __aiter__(self) -> _InstrumentedStream:
        return self

    async def __anext__(self) -> tts.SynthesizedAudio:
        frame = await self._inner.__anext__()
        if not self._first_audio_logged:
            self._first_audio_logged = True
            now = time.perf_counter()
            stream_to_audio_ms = _to_ms(now - self._opened_at)
            llm_to_audio_ms = (
                _to_ms(now - self._first_text_at)
                if self._first_text_at is not None
                else None
            )

            if stream_to_audio_ms is not None:
                _h_stream_to_audio.record(
                    stream_to_audio_ms, attributes=self._owner._attrs()
                )

            log.info(
                "fish_tts.first_audio_frame",
                request_id=frame.request_id,
                segment_id=frame.segment_id,
                llm_to_audio_ms=llm_to_audio_ms,
                stream_open_to_audio_ms=stream_to_audio_ms,
                chars_before_audio=self._chars_buffered,
            )
        return frame

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def build_fish_tts(settings: BaseAgentSettings) -> FishTTS:
    """Construct an instrumented Fish TTS from a settings object.

    Required env: ``FISH_API_KEY``. Optional: ``FISH_VOICE_ID``, ``FISH_TTS_MODEL``,
    ``FISH_TTS_LATENCY_MODE``.
    """
    if not settings.fish_api_key:
        raise ValueError("FISH_API_KEY is required to build Fish TTS")

    kwargs: dict[str, Any] = {
        "api_key": settings.fish_api_key,
        "model": settings.fish_tts_model,
        "latency_mode": settings.fish_tts_latency_mode,
    }
    if settings.fish_voice_id:
        kwargs["voice_id"] = settings.fish_voice_id

    return FishTTS(**kwargs)


def _to_ms(value: float) -> float | None:
    if value < 0:
        return None
    return round(value * 1000, 2)


__all__ = ["FishTTS", "build_fish_tts"]
