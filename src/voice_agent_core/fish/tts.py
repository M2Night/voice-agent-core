"""Instrumented Fish Audio TTS for LiveKit Agents.

Subclasses ``livekit.plugins.fishaudio.TTS`` and adds:

- **Metrics** — emits OTEL histograms for TTFB, RTF, and counter for errors
- **Structured logging** — every synthesis logs via structlog with stable key names
- **Stream-level instrumentation** — measures stream-open→first-text and
  stream-open→first-audio latencies in addition to the plugin-emitted TTFB
- **Sentence-boundary buffering** — accumulates LLM-streamed tokens until a
  punctuation mark, then pushes the complete clause to the Fish TTS stream.
  Reduces "chunk boundary in the middle of a word" artifacts.
- **Native streaming impl** — drops the upstream plugin's per-sentence flush and lets
  Fish chunk by ``chunk_length``/``min_chunk_length``, removing the per-sentence audio
  bursts that starve LiveKit's audio emitter ("flush audio emitter due to slow audio
  generation") and the boundary clicks they produce. (Text still goes through
  ``_InstrumentedStream`` sentence buffering first, so Fish receives clauses — the change
  is that we don't *flush* between them.)
- **Onset fade-in** — a short linear fade on the first audio of each segment to declick
  abrupt onsets.

Policy vs mechanism: :func:`build_fish_tts` fixes the optimization policy — pcm output,
``impl="native"``, ``min_chunk_length=20``, ``onset_fade_ms=8`` (hardcoded module
constants; not env-configurable). The ``FishTTS`` constructor still accepts these as
kwargs so the mechanism stays available for tests / direct construction; only
``FISH_API_KEY`` and ``FISH_TTS_LATENCY_MODE`` are env knobs (see
``voice_agent_core.fish.settings.FishSettings``).
"""

from __future__ import annotations

import array
import asyncio
import dataclasses
import time
from typing import TYPE_CHECKING, Any

import aiohttp
import msgpack
from livekit import rtc
from livekit.agents import APIConnectOptions, APIStatusError, tts, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.plugins import fishaudio
from livekit.plugins.fishaudio.tts import SynthesizeStream as _UpstreamFishStream
from livekit.plugins.fishaudio.tts import _build_tts_request

from voice_agent_core.fish.settings import FishSettings
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

    def __init__(
        self,
        *,
        impl: str = "native",
        min_chunk_length: int = 20,
        onset_fade_ms: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._impl = impl
        self._min_chunk_length = min_chunk_length
        self._onset_fade_ms = onset_fade_ms
        self.on("metrics_collected", self._on_metrics)
        self.on("error", self._on_error)
        log.info(
            "fish_tts.ready",
            provider=self.provider,
            model=self.model,
            voice_id=self.voice_id,
            latency_mode=self.latency_mode,
            output_format=self.output_format,
            sample_rate=self.sample_rate,
            impl=impl,
            min_chunk_length=min_chunk_length,
            onset_fade_ms=onset_fade_ms,
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
        if self._impl == "native":
            inner: tts.SynthesizeStream = _NativeFishStream(
                tts=self,
                conn_options=conn_options,
                min_chunk_length=self._min_chunk_length,
            )
            # Track for aclose() like the upstream stream() does.
            self._streams.add(inner)
        else:
            inner = super().stream(conn_options=conn_options)
        return _InstrumentedStream(
            inner=inner,
            owner=self,
            onset_fade_ms=self._onset_fade_ms,
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


def _native_start_request(opts: Any, min_chunk_length: int) -> dict[str, Any]:
    """Fish ``start`` request for the native streaming impl.

    Reuses the upstream field set (format / sample_rate / chunk_length / latency /
    voice / normalize / prosody / sampling) so behavior matches the plugin, and adds
    ``min_chunk_length`` — which the upstream ``_build_tts_request`` does not send.
    """
    request = dict(_build_tts_request(opts))
    request["min_chunk_length"] = min_chunk_length
    return request


class _NativeFishStream(_UpstreamFishStream):
    """Fish WebSocket streaming without the upstream's per-sentence flush.

    The upstream ``SynthesizeStream`` tokenizes incoming text into sentences and
    sends a ``flush`` after each one, forcing Fish to synthesize every sentence as a
    separate burst. Between bursts the audio stream has gaps, so LiveKit's
    ``AudioEmitter`` underruns ("flush audio emitter due to slow audio generation")
    and each burst boundary is an abrupt amplitude step — an audible click.

    This override drops the per-sentence flush and lets Fish decide when to synthesize
    via ``chunk_length`` / ``min_chunk_length``; a single ``flush`` is sent at
    end-of-input to synthesize the trailing buffer, then ``stop``. Only ``_run_ws`` is
    overridden; the upstream ``_run`` (WS connect, emitter init, segment + error
    handling) is reused.

    Note on "continuous": text still passes through ``_InstrumentedStream``'s
    sentence-boundary buffering before it reaches this stream, so Fish receives clauses,
    not raw per-token text. The win here is *not flushing* between clauses (no forced
    per-sentence bursts). Latency tradeoff: because we no longer force synthesis per
    clause, first audio for a short opening clause depends on Fish starting early from
    ``min_chunk_length`` rather than an explicit flush — keep ``min_chunk_length`` small
    (default 20) so TTFT doesn't regress. Validated by local smoke (TTS first-byte
    stayed ~0.6 s vs the plugin path).
    """

    def __init__(
        self,
        *,
        tts: FishTTS,
        conn_options: APIConnectOptions,
        min_chunk_length: int,
    ) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._min_chunk_length = min_chunk_length

    async def _run_ws(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        output_emitter: tts.AudioEmitter,
    ) -> None:
        start_request = _native_start_request(self._opts, self._min_chunk_length)

        async def send_task() -> None:
            await ws.send_bytes(
                msgpack.packb(
                    {"event": "start", "request": start_request}, use_bin_type=True
                )
            )
            first_token = True
            async for data in self._input_ch:
                # Native mode: ignore per-clause flush sentinels. Fish buffers text
                # and synthesizes by chunk_length/min_chunk_length, so we get larger
                # continuous chunks instead of one burst per sentence.
                if isinstance(data, self._FlushSentinel):
                    continue
                if not data:
                    continue
                if first_token:
                    self._mark_started()
                    first_token = False
                await ws.send_bytes(
                    msgpack.packb({"event": "text", "text": data}, use_bin_type=True)
                )
            # One flush at end-of-input synthesizes the trailing buffer, then stop.
            await ws.send_bytes(msgpack.packb({"event": "flush"}, use_bin_type=True))
            await ws.send_bytes(msgpack.packb({"event": "stop"}, use_bin_type=True))

        async def recv_task() -> None:
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError(
                        "Fish Audio websocket connection closed unexpectedly",
                        status_code=ws.close_code or -1,
                        request_id=None,
                        body=f"{msg.data=} {msg.extra=}",
                    )
                if msg.type != aiohttp.WSMsgType.BINARY:
                    # Mirror upstream: log non-binary frames so protocol drift is visible.
                    log.debug("fish_tts.native_unexpected_msg_type", msg_type=str(msg.type))
                    continue

                data = msgpack.unpackb(msg.data, raw=False)
                event = data.get("event")
                if event == "audio":
                    audio = data.get("audio")
                    if audio:
                        output_emitter.push(audio)
                elif event == "finish":
                    if data.get("reason") == "error":
                        raise APIStatusError(
                            "Fish Audio TTS reported an error",
                            status_code=-1,
                            request_id=None,
                            body=str(data),
                        )
                    break
                else:
                    log.debug("fish_tts.native_unknown_event", fish_event=event)

        tasks = [
            asyncio.create_task(send_task(), name="fish_native_send"),
            asyncio.create_task(recv_task(), name="fish_native_recv"),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)


_SENTENCE_PUNCT = frozenset({".", "。", ",", "，", "!", "！", "?", "？", ";", "；", ":", "：", "\n"})  # noqa: RUF001  (intentional full-width CJK punctuation)
"""Characters that close a TTS-worthy clause.

Includes both ASCII and CJK punctuation so the buffering works across
mixed-language responses. The newline is here too — LLMs sometimes
emit one between list items, and treating it as a clause boundary
keeps Fish's prosody intact.
"""


class _InstrumentedStream:
    """Wraps a SynthesizeStream to add metrics + sentence-boundary buffering.

    Forwards all unknown attributes/methods to the wrapped stream via
    ``__getattr__``, so it stays a drop-in replacement even if the
    upstream interface grows.

    Buffering behavior (the "buffer_sentences" pattern from hanabi):
    ``push_text`` accumulates incoming LLM tokens in
    ``_sentence_buffer`` and only forwards to the underlying Fish TTS
    stream when the buffer contains a closed clause (text ending in
    punctuation followed by non-punctuation). On ``end_input`` /
    ``flush`` / ``aclose`` we drain any remaining buffer so trailing
    text without a final period still gets spoken. The TTS receives
    semantically complete clauses rather than arbitrary LLM token
    boundaries — Fish can plan prosody better, and chunk boundaries no
    longer fall mid-word.
    """

    def __init__(
        self,
        *,
        inner: tts.SynthesizeStream,
        owner: FishTTS,
        onset_fade_ms: int = 0,
    ) -> None:
        self._inner = inner
        self._owner = owner
        self._opened_at = time.perf_counter()
        self._first_text_at: float | None = None
        self._first_audio_logged = False
        self._chars_buffered = 0
        self._sentence_buffer = ""
        # Onset fade-in state (per-segment). _fade_total is computed lazily from the
        # frame's own sample rate so it's correct regardless of output sample rate.
        self._onset_fade_ms = onset_fade_ms
        self._fade_segment: str | None = None
        self._fade_done = 0
        self._fade_total = 0

    def _flush_sentence_buffer(self) -> None:
        """Push any remaining buffered text and reset the buffer."""
        if self._sentence_buffer:
            self._inner.push_text(self._sentence_buffer)
            self._sentence_buffer = ""

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

        # Sentence-boundary buffering: accumulate tokens until we see a
        # punctuation mark followed by non-punctuation. We send the prefix
        # (including the punctuation run) to Fish and keep the rest buffered.
        # The "followed by non-punctuation" check handles patterns like "..."
        # and "?!" where consecutive punctuation should stay in one chunk.
        self._sentence_buffer += token
        while True:
            idx = -1
            for i, ch in enumerate(self._sentence_buffer):
                if ch in _SENTENCE_PUNCT:
                    idx = i
                    break
            if idx == -1:
                break
            end = idx + 1
            while end < len(self._sentence_buffer) and self._sentence_buffer[end] in _SENTENCE_PUNCT:
                end += 1
            if end == len(self._sentence_buffer):
                # Punctuation runs to the end of the buffer — wait for the
                # next token to see if more punctuation follows before we
                # decide where to split.
                break
            part, self._sentence_buffer = (
                self._sentence_buffer[:end],
                self._sentence_buffer[end:],
            )
            self._inner.push_text(part)

    def flush(self) -> None:
        self._flush_sentence_buffer()
        self._inner.flush()

    def end_input(self) -> None:
        self._flush_sentence_buffer()
        self._inner.end_input()

    async def aclose(self) -> None:
        # Drain any partial clause before closing so trailing text isn't lost.
        self._flush_sentence_buffer()
        await self._inner.aclose()

    async def __aenter__(self) -> _InstrumentedStream:
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._inner.__aexit__(*args)

    def __aiter__(self) -> _InstrumentedStream:
        return self

    def _apply_onset_fade(self, frame: tts.SynthesizedAudio) -> tts.SynthesizedAudio:
        """Ramp the first ``onset_fade_ms`` of each segment's PCM from 0→full gain.

        Turns an abrupt onset (silence → full amplitude in ~1 sample) into a short
        slope, which removes the broadband click without audibly softening the attack.
        The fade may span several frames; ``_fade_done`` tracks progress per segment.

        Note: applied consumer-side (after the AudioEmitter), so it reaches playback and
        the session recording but NOT ``LK_DUMP_TTS`` dumps, which are written upstream
        in the emitter — i.e. dumps are pre-fade and can't be used to validate it.

        ``_fade_pcm_bytes`` assumes 16-bit mono PCM; Fish output is documented mono, so
        we skip (rather than corrupt) anything multi-channel as a safety net.
        """
        if frame.frame.num_channels != 1:
            return frame
        if frame.segment_id != self._fade_segment:
            self._fade_segment = frame.segment_id
            self._fade_done = 0
            self._fade_total = max(
                1, int(frame.frame.sample_rate * self._onset_fade_ms / 1000)
            )
        if self._fade_done >= self._fade_total:
            return frame
        faded, self._fade_done = _fade_pcm_bytes(
            bytes(frame.frame.data), self._fade_total, self._fade_done
        )
        new_audio = rtc.AudioFrame(
            data=faded,
            sample_rate=frame.frame.sample_rate,
            num_channels=frame.frame.num_channels,
            samples_per_channel=frame.frame.samples_per_channel,
        )
        return dataclasses.replace(frame, frame=new_audio)

    async def __anext__(self) -> tts.SynthesizedAudio:
        frame = await self._inner.__anext__()
        if self._onset_fade_ms > 0:
            frame = self._apply_onset_fade(frame)
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
                # A/B label: which wire format produced this first frame, so dumps
                # (LK_DUMP_TTS=1) and logs self-identify across runs. wav → upstream
                # routes through AudioStreamDecoder; pcm → raw passthrough.
                output_format=self._owner.output_format,
                sample_rate=self._owner.sample_rate,
            )
        return frame

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _fade_pcm_bytes(data: bytes, fade_total: int, fade_done: int) -> tuple[bytes, int]:
    """Apply a linear gain ramp to 16-bit mono PCM, resuming from ``fade_done``.

    Scales sample ``k`` (counting from segment start) by ``k / fade_total`` for
    ``fade_done <= k < fade_total``; samples at/after ``fade_total`` are untouched.
    Returns the new bytes and the updated ramp position. Pure function — unit-tested.
    """
    samples = array.array("h")
    samples.frombytes(data)
    i = 0
    done = fade_done
    n = len(samples)
    while i < n and done < fade_total:
        samples[i] = int(samples[i] * done / fade_total)
        i += 1
        done += 1
    return samples.tobytes(), done


# Hardcoded Fish TTS optimizations (validated; no per-scenario tradeoff, so not
# env-configurable — see fish/settings.py). pcm avoids the first-phoneme decoder click;
# native streaming drops the per-sentence flush; 8 ms onset fade declicks abrupt starts;
# min_chunk_length 20 keeps Fish emitting continuous chunks. Sample rate is left at the
# Fish plugin per-format default (pcm = 24 kHz; LiveKit resamples to the pipeline rate).
_OUTPUT_FORMAT = "pcm"
_IMPL = "native"
_MIN_CHUNK_LENGTH = 20
_ONSET_FADE_MS = 8


def build_fish_tts(settings: BaseAgentSettings) -> FishTTS:
    """Construct an instrumented Fish TTS from a settings object.

    Generic model/voice come from ``settings`` (``TTS_MODEL`` / ``TTS_VOICE``); the Fish
    API key and latency mode from :class:`~voice_agent_core.fish.settings.FishSettings`
    (``FISH_API_KEY`` / ``FISH_TTS_LATENCY_MODE``). The remaining Fish TTS behavior is a
    hardcoded optimization (see the module constants above).
    """
    fish = FishSettings()
    if not fish.api_key:
        raise ValueError("FISH_API_KEY is required to build Fish TTS")

    kwargs: dict[str, Any] = {
        "api_key": fish.api_key,
        "model": settings.tts_model,
        "latency_mode": fish.tts_latency_mode,
        "output_format": _OUTPUT_FORMAT,
        "impl": _IMPL,
        "min_chunk_length": _MIN_CHUNK_LENGTH,
        "onset_fade_ms": _ONSET_FADE_MS,
    }
    if settings.tts_voice:
        kwargs["voice_id"] = settings.tts_voice

    return FishTTS(**kwargs)


def _to_ms(value: float) -> float | None:
    if value < 0:
        return None
    return round(value * 1000, 2)


__all__ = ["FishTTS", "build_fish_tts"]
