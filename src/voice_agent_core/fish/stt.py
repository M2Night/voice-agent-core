"""Fish Audio ASR adapter for LiveKit Agents.

Drop-in :class:`livekit.agents.stt.STT` implementation wrapping Fish Audio's batch
ASR endpoint. Adds:

- **Metrics** — OTEL histogram for request latency, counter for errors
- **Structured logging** — every recognition logs via structlog
- **Retry on connection errors** — exponential backoff via tenacity; status errors
  (4xx/5xx) are NOT retried since they typically indicate a request problem
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import aiohttp
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
)
from livekit.agents.language import LanguageCode
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from voice_agent_core.fish.settings import FishSettings
from voice_agent_core.observability import MetricNames, get_logger, get_meter

if TYPE_CHECKING:
    from voice_agent_core.config import BaseAgentSettings

DEFAULT_BASE_URL = "https://api.fish.audio"
_USER_AGENT = "voice-agent-core/0.1"

log = get_logger(__name__)
_meter = get_meter("voice_agent_core.fish.stt")
_h_latency = _meter.create_histogram(
    MetricNames.FISH_STT_LATENCY_MS,
    unit="ms",
    description="Fish ASR end-to-end request latency",
)
_c_errors = _meter.create_counter(
    MetricNames.FISH_STT_ERRORS,
    description="Fish ASR error count",
)

@dataclass(slots=True)
class _STTOptions:
    api_key: str
    base_url: str
    language: str | None
    ignore_timestamps: bool
    max_retries: int


class FishSTT(stt.STT):
    """LiveKit STT adapter backed by Fish Audio's batch ASR endpoint.

    Pass ``language=None`` (or ``"auto"``) for automatic language detection — useful
    for multilingual use cases. Set a specific language code (e.g. ``"en"``, ``"zh"``)
    for slightly better accuracy when the language is known up-front.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        language: str | None = None,
        ignore_timestamps: bool = True,
        max_retries: int = 3,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
                offline_recognize=True,
            )
        )

        if not api_key:
            raise ValueError("Fish Audio API key required")

        self._opts = _STTOptions(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            language=language if language not in ("", "auto", "multi") else None,
            ignore_timestamps=ignore_timestamps,
            max_retries=max_retries,
        )
        self._session = http_session

    @property
    def model(self) -> str:
        return "fish-audio/asr"

    @property
    def provider(self) -> str:
        return "FishAudio"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        req_language = language if is_given(language) else self._opts.language
        if req_language in ("", "auto", "multi"):
            req_language = None

        wav_bytes = _to_wav(buffer)
        started_at = time.perf_counter()
        attrs = {"language": req_language or "auto"}

        try:
            payload, request_id, status = await self._post_with_retry(
                wav_bytes=wav_bytes,
                language=req_language,
                timeout_s=conn_options.timeout,
            )
        except APIConnectionError:
            _c_errors.add(1, attributes={**attrs, "kind": "connection"})
            raise
        except APITimeoutError:
            _c_errors.add(1, attributes={**attrs, "kind": "timeout"})
            raise

        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        _h_latency.record(latency_ms, attributes=attrs)

        if status >= 400:
            _c_errors.add(1, attributes={**attrs, "kind": f"http_{status}"})
            log.error(
                "fish_stt.http_error",
                request_id=request_id,
                status_code=status,
                body=payload,
                language=req_language,
                latency_ms=latency_ms,
            )
            raise APIStatusError(
                "Fish Audio ASR request failed",
                status_code=status,
                request_id=request_id,
                body=payload,
            )

        event = _build_speech_event(payload, language=req_language, request_id=request_id)
        transcript = event.alternatives[0].text if event.alternatives else ""
        log.debug(
            "fish_stt.transcript",
            request_id=request_id,
            chars=len(transcript),
            language=req_language,
            latency_ms=latency_ms,
        )
        return event

    async def _post_with_retry(
        self,
        *,
        wav_bytes: bytes,
        language: str | None,
        timeout_s: float,
    ) -> tuple[dict[str, Any], str, int]:
        """POST to Fish ASR with exponential-backoff retry on connection errors.

        Returns (payload, request_id, status_code). Status errors are NOT retried —
        they're the caller's problem (bad audio, bad key, etc.).
        """
        retryer = AsyncRetrying(
            retry=retry_if_exception_type(aiohttp.ClientError),
            stop=stop_after_attempt(self._opts.max_retries),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=2.0),
            reraise=True,
        )

        async for attempt in retryer:
            with attempt:
                try:
                    return await self._post_once(
                        wav_bytes=wav_bytes,
                        language=language,
                        timeout_s=timeout_s,
                    )
                except TimeoutError as exc:
                    raise APITimeoutError() from exc
        # Unreachable — AsyncRetrying always either returns or raises
        raise APIConnectionError("Fish ASR retry loop exited unexpectedly")

    async def _post_once(
        self,
        *,
        wav_bytes: bytes,
        language: str | None,
        timeout_s: float,
    ) -> tuple[dict[str, Any], str, int]:
        form = aiohttp.FormData()
        form.add_field("audio", wav_bytes, filename="audio.wav", content_type="audio/wav")
        if language:
            form.add_field("language", language)
        form.add_field(
            "ignore_timestamps",
            "true" if self._opts.ignore_timestamps else "false",
        )

        try:
            async with self._ensure_session().post(
                url=f"{self._opts.base_url}/v1/asr",
                headers={
                    "Authorization": f"Bearer {self._opts.api_key}",
                    "User-Agent": _USER_AGENT,
                },
                data=form,
                timeout=aiohttp.ClientTimeout(total=timeout_s),
            ) as resp:
                request_id = resp.headers.get("x-request-id", "")
                payload = await _parse_response(resp)
                return payload, request_id, resp.status
        except aiohttp.ClientError as exc:
            # Caller's retry layer decides what to do
            log.warning("fish_stt.connection_error", error=repr(exc))
            raise APIConnectionError(str(exc)) from exc


def build_fish_stt(settings: BaseAgentSettings) -> FishSTT:
    """Construct an instrumented Fish STT from a settings object.

    ``FISH_API_KEY`` comes from :class:`~voice_agent_core.fish.settings.FishSettings`;
    the language hint stays generic (``STT_LANGUAGE``).
    """
    fish = FishSettings()
    if not fish.api_key:
        raise ValueError("FISH_API_KEY is required to build Fish STT")
    return FishSTT(
        api_key=fish.api_key,
        language=settings.stt_language,
    )


def _to_wav(buffer: utils.AudioBuffer) -> bytes:
    frame = utils.merge_frames(buffer) if isinstance(buffer, list) else buffer
    return frame.to_wav_bytes()


async def _parse_response(resp: aiohttp.ClientResponse) -> dict[str, Any]:
    try:
        payload = await resp.json()
    except aiohttp.ContentTypeError:
        return {"error": await resp.text()}
    return payload if isinstance(payload, dict) else {"response": payload}


def _build_speech_event(
    payload: dict[str, Any],
    *,
    language: str | None,
    request_id: str,
) -> stt.SpeechEvent:
    text = str(payload.get("text") or "")
    duration = float(payload.get("duration") or 0.0)
    segments = payload.get("segments") or []

    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        request_id=request_id,
        alternatives=[
            stt.SpeechData(
                language=LanguageCode(language or ""),
                text=text,
                start_time=0.0,
                end_time=duration,
                metadata={"segments": segments} if segments else None,
            )
        ],
    )


__all__ = ["FishSTT", "build_fish_stt"]
