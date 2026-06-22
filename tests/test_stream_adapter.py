"""Tests for the VAD-based STT StreamAdapter."""

from __future__ import annotations

from collections.abc import Iterable

from livekit import rtc
from livekit.agents.stt import (
    STT,
    SpeechData,
    SpeechEvent,
    SpeechEventType,
    STTCapabilities,
)
from livekit.agents.types import APIConnectOptions
from livekit.agents.vad import VADEvent, VADEventType

from voice_agent_core.stt import StreamAdapter


class _FakeBatchSTT(STT):
    def __init__(self, *, text: str = "hello") -> None:
        super().__init__(
            capabilities=STTCapabilities(
                streaming=False,
                interim_results=False,
                offline_recognize=True,
            )
        )
        self.text = text
        self.last_language: str | None = None
        self.recognize_calls = 0
        self.closed = False

    @property
    def model(self) -> str:
        return "fake-batch"

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
        self.recognize_calls += 1
        self.last_language = language
        return SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            request_id="req-1",
            alternatives=[SpeechData(language="en", text=self.text)],
        )

    async def aclose(self) -> None:
        self.closed = True


class _FakeVAD:
    def __init__(self, events: Iterable[VADEvent] = ()) -> None:
        self.stream_obj = _FakeVADStream(events)

    def stream(self):
        return self.stream_obj


class _FakeVADStream:
    def __init__(self, events: Iterable[VADEvent]) -> None:
        self._events = iter(events)
        self.pushed_frames = []
        self.flush_count = 0
        self.ended = False
        self.closed = False

    def push_frame(self, frame) -> None:
        self.pushed_frames.append(frame)

    def flush(self) -> None:
        self.flush_count += 1

    def end_input(self) -> None:
        self.ended = True

    async def aclose(self) -> None:
        self.closed = True

    def __aiter__(self):
        return self

    async def __anext__(self) -> VADEvent:
        try:
            return next(self._events)
        except StopIteration:
            raise StopAsyncIteration from None


async def test_stream_adapter_delegates_batch_recognize() -> None:
    wrapped = _FakeBatchSTT()
    adapter = StreamAdapter(stt=wrapped, vad=_FakeVAD())
    frame = rtc.AudioFrame(
        data=b"\x00" * 320,
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=160,
    )

    event = await adapter.recognize(buffer=frame, language="zh")

    assert event.type == SpeechEventType.FINAL_TRANSCRIPT
    assert event.request_id == "req-1"
    assert event.alternatives[0].text == "hello"
    assert wrapped.last_language == "zh"


def test_stream_adapter_capabilities_and_metadata() -> None:
    wrapped = _FakeBatchSTT()
    vad = _FakeVAD()
    adapter = StreamAdapter(stt=wrapped, vad=vad)

    assert adapter.capabilities.streaming is True
    assert adapter.capabilities.interim_results is False
    assert adapter.capabilities.offline_recognize is True
    assert adapter.model == "fake-batch"
    assert adapter.provider == "fake"
    assert adapter.wrapped_stt is wrapped
    assert adapter.vad is vad


async def test_stream_adapter_emits_final_transcript_from_vad_segment() -> None:
    frame = rtc.AudioFrame(
        data=b"\x00" * 320,
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=160,
    )
    vad = _FakeVAD(
        [
            VADEvent(
                type=VADEventType.START_OF_SPEECH,
                samples_index=0,
                timestamp=0,
                speech_duration=0,
                silence_duration=0,
            ),
            VADEvent(
                type=VADEventType.END_OF_SPEECH,
                samples_index=160,
                timestamp=0.01,
                speech_duration=0.01,
                silence_duration=0,
                frames=[frame],
            ),
        ]
    )
    wrapped = _FakeBatchSTT()
    adapter = StreamAdapter(stt=wrapped, vad=vad)

    stream = adapter.stream(language="zh")
    stream.push_frame(frame)
    stream.end_input()

    events = [event async for event in stream]

    assert [event.type for event in events] == [
        SpeechEventType.START_OF_SPEECH,
        SpeechEventType.END_OF_SPEECH,
        SpeechEventType.FINAL_TRANSCRIPT,
    ]
    assert events[-1].alternatives[0].text == "hello"
    assert wrapped.recognize_calls == 1
    assert wrapped.last_language == "zh"
    assert vad.stream_obj.pushed_frames == [frame]
    assert vad.stream_obj.ended is True
    assert vad.stream_obj.closed is True


async def test_stream_adapter_skips_end_of_speech_without_frames() -> None:
    vad = _FakeVAD(
        [
            VADEvent(
                type=VADEventType.END_OF_SPEECH,
                samples_index=160,
                timestamp=0.01,
                speech_duration=0.01,
                silence_duration=0,
                frames=[],
            ),
        ]
    )
    wrapped = _FakeBatchSTT()
    stream = StreamAdapter(stt=wrapped, vad=vad).stream()
    stream.end_input()

    events = [event async for event in stream]

    assert [event.type for event in events] == [SpeechEventType.END_OF_SPEECH]
    assert wrapped.recognize_calls == 0


async def test_stream_adapter_skips_blank_transcript() -> None:
    frame = rtc.AudioFrame(
        data=b"\x00" * 320,
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=160,
    )
    vad = _FakeVAD(
        [
            VADEvent(
                type=VADEventType.END_OF_SPEECH,
                samples_index=160,
                timestamp=0.01,
                speech_duration=0.01,
                silence_duration=0,
                frames=[frame],
            ),
        ]
    )
    wrapped = _FakeBatchSTT(text="   ")
    stream = StreamAdapter(stt=wrapped, vad=vad).stream()
    stream.push_frame(frame)
    stream.end_input()

    events = [event async for event in stream]

    assert [event.type for event in events] == [SpeechEventType.END_OF_SPEECH]
    assert wrapped.recognize_calls == 1


async def test_stream_adapter_forwards_metrics_once_and_removes_listeners() -> None:
    wrapped = _FakeBatchSTT()
    adapter = StreamAdapter(stt=wrapped, vad=_FakeVAD())
    seen = []
    adapter.on("metrics_collected", lambda metric: seen.append(metric))

    wrapped.emit("metrics_collected", "metric-1")
    wrapped.emit("metrics_collected", "metric-2")

    assert seen == ["metric-1", "metric-2"]
    await adapter.aclose()
    assert wrapped.closed is True

    wrapped.emit("metrics_collected", "metric-3")
    assert seen == ["metric-1", "metric-2"]
