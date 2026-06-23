"""Tests for voice_agent_core.fish builders.

Real network calls aren't tested here — those require a Fish API key and live
endpoint. We test construction, defaults, and error paths instead.
"""

from __future__ import annotations

import array
from types import SimpleNamespace

import aiohttp
import msgpack
import pytest
from livekit.agents import APIStatusError

from voice_agent_core.config import BaseAgentSettings
from voice_agent_core.fish import build_fish_stt, build_fish_tts
from voice_agent_core.fish.tts import (
    _fade_pcm_bytes,
    _native_start_request,
    _NativeFishStream,
    _to_ms,
)


class TestBuildFishTTS:
    def test_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FISH_API_KEY", raising=False)
        s = BaseAgentSettings()
        with pytest.raises(ValueError, match="FISH_API_KEY"):
            build_fish_tts(s)

    def test_constructs_with_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FISH_API_KEY", "test-key")
        for k in ("FISH_TTS_OUTPUT_FORMAT", "FISH_TTS_SAMPLE_RATE", "FISH_TTS_IMPL",
                  "FISH_TTS_MIN_CHUNK_LENGTH", "FISH_TTS_ONSET_FADE_MS"):
            monkeypatch.delenv(k, raising=False)
        s = BaseAgentSettings()
        tts = build_fish_tts(s)
        assert tts.model == "s2-pro"
        assert tts.latency_mode == "balanced"
        # Default flipped to pcm to mitigate the first-phoneme decoder click; pcm/wav
        # both default to 24 kHz on the Fish plugin.
        assert tts.output_format == "pcm"
        assert tts.sample_rate == 24000
        # Native streaming impl is the default; onset fade defaults to 8 ms.
        assert tts._impl == "native"
        assert tts._min_chunk_length == 20
        assert tts._onset_fade_ms == 8

    def test_optimizations_are_hardcoded_not_env_overridable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # pcm / native / onset-fade / min-chunk-length are hardcoded optimizations —
        # stray FISH_TTS_* env must NOT change them (only FISH_TTS_LATENCY_MODE is a knob).
        monkeypatch.setenv("FISH_API_KEY", "test-key")
        monkeypatch.setenv("FISH_TTS_OUTPUT_FORMAT", "wav")
        monkeypatch.setenv("FISH_TTS_IMPL", "plugin")
        monkeypatch.setenv("FISH_TTS_MIN_CHUNK_LENGTH", "40")
        monkeypatch.setenv("FISH_TTS_ONSET_FADE_MS", "0")
        monkeypatch.setenv("FISH_TTS_SAMPLE_RATE", "44100")
        tts = build_fish_tts(BaseAgentSettings())
        assert tts.output_format == "pcm"
        assert tts._impl == "native"
        assert tts._min_chunk_length == 20
        assert tts._onset_fade_ms == 8
        assert tts.sample_rate == 24000

    def test_native_start_request_adds_min_chunk_length(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The native start request reuses the upstream field set and adds the
        # min_chunk_length the plugin's request omits.
        monkeypatch.setenv("FISH_API_KEY", "test-key")
        tts = build_fish_tts(BaseAgentSettings())
        req = _native_start_request(tts._opts, 20)
        assert req["min_chunk_length"] == 20
        assert req["format"] == "pcm"
        assert "chunk_length" in req
        assert req["sample_rate"] == 24000

    def test_fish_latency_mode_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FISH_API_KEY", "test-key")
        monkeypatch.setenv("FISH_TTS_LATENCY_MODE", "low")
        s = BaseAgentSettings()
        tts = build_fish_tts(s)
        assert tts.latency_mode == "low"

    def test_voice_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FISH_API_KEY", "test-key")
        monkeypatch.setenv("TTS_VOICE", "voice-abc")
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


class TestFadePcmBytes:
    """Onset fade-in: linear gain ramp on the first ``fade_total`` PCM samples."""

    def _samples(self, data: bytes) -> list[int]:
        a = array.array("h")
        a.frombytes(data)
        return list(a)

    def test_ramps_first_samples_and_leaves_rest(self) -> None:
        pcm = array.array("h", [1000] * 6).tobytes()
        out, done = _fade_pcm_bytes(pcm, fade_total=4, fade_done=0)
        # gain = k/4 → 0, 250, 500, 750, then untouched 1000, 1000
        assert self._samples(out) == [0, 250, 500, 750, 1000, 1000]
        assert done == 4

    def test_resumes_across_frames(self) -> None:
        # Second frame continues the ramp from where the first left off.
        pcm = array.array("h", [1000] * 2).tobytes()
        out, done = _fade_pcm_bytes(pcm, fade_total=4, fade_done=2)
        assert self._samples(out) == [500, 750]  # k=2,3
        assert done == 4

    def test_noop_once_fade_complete(self) -> None:
        pcm = array.array("h", [1000] * 3).tobytes()
        out, done = _fade_pcm_bytes(pcm, fade_total=4, fade_done=4)
        assert self._samples(out) == [1000, 1000, 1000]
        assert done == 4


class _FlushSentinel:
    """Stand-in for the base SynthesizeStream._FlushSentinel."""


class _FakeWS:
    """Minimal aiohttp-WS double: records sent (unpacked) events, replays incoming."""

    def __init__(self, incoming: list) -> None:
        self.sent: list = []
        self._incoming = list(incoming)
        self.close_code = 1000

    async def send_bytes(self, data: bytes) -> None:
        self.sent.append(msgpack.unpackb(data, raw=False))

    async def receive(self):
        if self._incoming:
            return self._incoming.pop(0)
        return SimpleNamespace(type=aiohttp.WSMsgType.CLOSED, data=None, extra=None)


class _FakeEmitter:
    def __init__(self) -> None:
        self.pushed: list = []

    def push(self, audio) -> None:
        self.pushed.append(audio)


def _binary(obj: dict):
    return SimpleNamespace(
        type=aiohttp.WSMsgType.BINARY,
        data=msgpack.packb(obj, use_bin_type=True),
        extra=None,
    )


def _native_self(monkeypatch: pytest.MonkeyPatch, input_items: list) -> SimpleNamespace:
    """A fake `self` exposing exactly what `_NativeFishStream._run_ws` reads, so we can
    exercise the overridden websocket loop without constructing the base stream (which
    needs a running job context)."""
    monkeypatch.setenv("FISH_API_KEY", "test-key")
    tts = build_fish_tts(BaseAgentSettings())

    async def _aiter():
        for item in input_items:
            yield item

    return SimpleNamespace(
        _opts=tts._opts,
        _min_chunk_length=20,
        _FlushSentinel=_FlushSentinel,
        _mark_started=lambda: None,
        _input_ch=_aiter(),
    )


class TestNativeRunWs:
    """Fake-WS coverage for the forked _run_ws (it depends on private upstream internals)."""

    async def test_happy_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _native_self(monkeypatch, ["Hello", _FlushSentinel(), " world"])
        ws = _FakeWS([_binary({"event": "audio", "audio": b"\x01\x02"}), _binary({"event": "finish"})])
        emitter = _FakeEmitter()

        await _NativeFishStream._run_ws(fake, ws, emitter)

        events = [m.get("event") for m in ws.sent]
        # start first (with min_chunk_length), then text, terminal flush + stop.
        assert events[0] == "start"
        assert ws.sent[0]["request"]["min_chunk_length"] == 20
        assert events[-2:] == ["flush", "stop"]
        # FlushSentinel is ignored; only real text is forwarded.
        assert [m["text"] for m in ws.sent if m.get("event") == "text"] == ["Hello", " world"]
        # audio pushed to the emitter; exits cleanly on finish.
        assert emitter.pushed == [b"\x01\x02"]

    async def test_raises_on_finish_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _native_self(monkeypatch, [])
        ws = _FakeWS([_binary({"event": "finish", "reason": "error"})])
        with pytest.raises(APIStatusError):
            await _NativeFishStream._run_ws(fake, ws, _FakeEmitter())

    async def test_raises_on_ws_close(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _native_self(monkeypatch, [])
        ws = _FakeWS([SimpleNamespace(type=aiohttp.WSMsgType.CLOSED, data=None, extra=None)])
        with pytest.raises(APIStatusError):
            await _NativeFishStream._run_ws(fake, ws, _FakeEmitter())
