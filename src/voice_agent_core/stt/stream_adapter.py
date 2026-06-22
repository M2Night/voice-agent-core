"""Adapt batch STT implementations to LiveKit's streaming STT interface.

This is useful for providers such as Fish Audio whose ASR API is batch-only. The
adapter uses a local VAD stream to cut incoming audio into speech segments and calls
the wrapped STT's ``recognize`` method for each completed segment. It improves the
latency profile of batch STT, but it is still not equivalent to a native streaming
ASR such as Deepgram: transcripts are emitted only after each VAD segment ends and
the batch request completes.

LiveKit ships a similar adapter. This local implementation exists because the core
library needs a small public surface (``wrapped_stt`` / ``vad``) for tests and
introspection, preserves selected metadata from wrapped recognition events, and
keeps provider-specific logging/metrics forwarding explicit.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable

from livekit.agents import utils
from livekit.agents.stt import (
    STT,
    RecognizeStream,
    SpeechEvent,
    SpeechEventType,
    STTCapabilities,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.vad import VAD, VADEventType

DEFAULT_STREAM_ADAPTER_API_CONNECT_OPTIONS = APIConnectOptions(
    max_retry=0,
    timeout=DEFAULT_API_CONNECT_OPTIONS.timeout,
)


class StreamAdapter(STT):
    """Expose a non-streaming STT as a streaming STT using VAD segmentation."""

    def __init__(self, *, stt: STT, vad: VAD) -> None:
        super().__init__(
            capabilities=STTCapabilities(
                streaming=True,
                interim_results=False,
                diarization=stt.capabilities.diarization,
                aligned_transcript=stt.capabilities.aligned_transcript,
                offline_recognize=stt.capabilities.offline_recognize,
            )
        )
        self._stt = stt
        self._vad = vad
        self._closed = False

        self._stt.on("metrics_collected", self._forward_metrics)
        self._stt.on("error", self._forward_errors)

    def _forward_metrics(self, *args: object, **kwargs: object) -> None:
        self.emit("metrics_collected", *args, **kwargs)

    def _forward_errors(self, *args: object, **kwargs: object) -> None:
        self.emit("error", *args, **kwargs)

    @property
    def wrapped_stt(self) -> STT:
        """The batch STT instance being adapted."""
        return self._stt

    @property
    def vad(self) -> VAD:
        """The VAD instance used to segment incoming audio."""
        return self._vad

    @property
    def model(self) -> str:
        return self._stt.model

    @property
    def provider(self) -> str:
        return self._stt.provider

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        return await self._stt.recognize(
            buffer=buffer,
            language=language,
            conn_options=conn_options,
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_STREAM_ADAPTER_API_CONNECT_OPTIONS,
    ) -> RecognizeStream:
        return _StreamAdapterStream(
            stt=self,
            vad=self._vad,
            wrapped_stt=self._stt,
            language=language,
            conn_options=conn_options,
        )

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stt.off("metrics_collected", self._forward_metrics)
        self._stt.off("error", self._forward_errors)
        await self._stt.aclose()


class _StreamAdapterStream(RecognizeStream):
    def __init__(
        self,
        *,
        stt: StreamAdapter,
        vad: VAD,
        wrapped_stt: STT,
        language: NotGivenOr[str],
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options)
        self._vad = vad
        self._vad_stream = None
        self._wrapped_stt = wrapped_stt
        self._language = language

    async def _metrics_monitor_task(
        self, event_aiter: AsyncIterable[SpeechEvent]
    ) -> None:
        # The wrapped batch STT emits its own metrics, which StreamAdapter forwards.
        async for event in event_aiter:
            if event.type == SpeechEventType.FINAL_TRANSCRIPT:
                self._num_retries = 0

    async def _run(self) -> None:
        self._vad_stream = self._vad.stream()

        async def _forward_input() -> None:
            assert self._vad_stream is not None
            async for input_item in self._input_ch:
                if isinstance(input_item, self._FlushSentinel):
                    self._vad_stream.flush()
                    continue
                self._vad_stream.push_frame(input_item)
            self._vad_stream.end_input()

        async def _recognize_segments() -> None:
            assert self._vad_stream is not None
            async for event in self._vad_stream:
                if event.type == VADEventType.START_OF_SPEECH:
                    self._event_ch.send_nowait(
                        SpeechEvent(type=SpeechEventType.START_OF_SPEECH)
                    )
                    continue

                if event.type != VADEventType.END_OF_SPEECH:
                    continue

                self._event_ch.send_nowait(
                    SpeechEvent(type=SpeechEventType.END_OF_SPEECH)
                )
                if not event.frames:
                    continue

                recognized = await self._wrapped_stt.recognize(
                    buffer=utils.merge_frames(event.frames),
                    language=self._language,
                    conn_options=self._conn_options,
                )
                alternatives = [
                    alt for alt in recognized.alternatives if alt.text.strip()
                ]
                if not alternatives:
                    continue

                self._event_ch.send_nowait(
                    SpeechEvent(
                        type=SpeechEventType.FINAL_TRANSCRIPT,
                        request_id=recognized.request_id,
                        alternatives=alternatives,
                        recognition_usage=recognized.recognition_usage,
                        speech_start_time=recognized.speech_start_time,
                    )
                )

        tasks = [
            asyncio.create_task(_forward_input(), name="stt_stream_adapter_forward_input"),
            asyncio.create_task(
                _recognize_segments(), name="stt_stream_adapter_recognize"
            ),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.cancel_and_wait(*tasks)
            if self._vad_stream is not None:
                await self._vad_stream.aclose()


__all__ = ["StreamAdapter"]
