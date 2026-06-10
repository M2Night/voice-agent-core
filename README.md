# voice-agent-core

Shared Python library powering Fish Audio's voice agent by leveraging with LiveKit
## What it provides

- **Fish Audio STT/TTS adapters** with built-in metrics (TTFB, TTFT, RTF) and retry
- **LLM factory** — LiveKit Inference (default) + OpenRouter backends, switchable via env var
- **Pipeline factory** — assemble STT + TTS + LLM + VAD + turn-detection from one settings object
- **Runtime helpers** (v0.2.0+) — `default_prewarm`, `build_session`, `default_room_options`, `warm_tts`, `is_warmup_session`: the hardened defaults every Fish voice demo arrived at after iteration, so new demos start there
- **SlackNotifier** — Block Kit payloads with retry + dev-log fallback when no webhook is set
- **Observability** — `structlog` JSON logging + OpenTelemetry metrics (console exporter by default; swap to Honeycomb/Datadog/Prometheus via one env var)

## Install

For local development from a sibling clone:

```bash
uv add "voice-agent-core @ file://$(pwd)/../voice-agent-core"
```

For production / published consumers, pin to a tagged version:

```bash
uv add "voice-agent-core @ git+https://github.com/M2Night/voice-agent-core.git@v0.2.0"
```

## Examples

See [examples/README.md](examples/README.md) — a minimum runnable agent you can run locally to verify the library end-to-end, and copy as a starting template.

## Status

Pre-1.0. Breaking changes possible until v1.0.0. Pin to a specific git tag in production.

## Used by

- [fish-voice-demos](https://github.com/M2Night/fish-voice-demos) — 5 production voice agents built on this library
