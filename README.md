# voice-agent-core

Shared Python library powering Fish Audio's voice agents on LiveKit.

## What it provides

- **STT/TTS adapters** — Deepgram or Fish STT, Fish or Inworld TTS, with provider defaults resolved through the registry
- **LLM factory** — OpenRouter (default), LiveKit Inference, or any OpenAI-compatible custom endpoint, switchable via env var
- **Pipeline factory** — assemble STT + TTS + LLM + VAD + turn-detection from one settings object
- **Runtime helpers** — `default_prewarm`, `build_session`, `default_room_options`, `warm_tts`, `is_warmup_session`: hardened defaults for LiveKit voice agents
- **Notifier interface + SlackNotifier** — provider-agnostic payloads, Slack Block Kit delivery, retry, and dev-log fallback when no webhook is set
- **Observability** — `structlog` JSON logging + OpenTelemetry metrics with console/no-op exporters; OTLP shipping can be added behind the reserved exporter branch

## Install

For local development from a sibling clone:

```bash
uv add "voice-agent-core @ file://$(pwd)/../voice-agent-core"
```

For production / published consumers, pin to a tagged version:

```bash
uv add "voice-agent-core @ git+https://github.com/M2Night/voice-agent-core.git@v0.2.1"
```

## Examples

See [examples/README.md](examples/README.md) — a minimum runnable agent you can run locally to verify the library end-to-end, and copy as a starting template.

## Status

Pre-1.0. Breaking changes possible until v1.0.0. Pin to a specific git tag in production.

## Used by

- [fish-voice-demos](https://github.com/M2Night/fish-voice-demos) — 5 production voice agents built on this library
