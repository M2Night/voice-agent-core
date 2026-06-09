# Smoke agent

The smallest runnable voice agent built on `voice-agent-core`. Two purposes:

1. **Verify your environment** — confirm Fish Audio + LiveKit + your local Python install work end-to-end before building a real agent on top.
2. **Reference template** — copy `smoke_agent.py` as the starting skeleton for your own agent and add your `Agent` subclass / `@function_tool` methods.

## Quick run

```bash
cd examples
cp .env.example .env                # then edit .env with real keys
cd ..                               # back to repo root so uv finds the venv
uv run python examples/smoke_agent.py dev
```

You'll see structured logs as the worker starts and waits for a job.

### Connect from the browser

1. Open https://agents-playground.livekit.io/
2. Sign in (or use "Connect with a token")
3. Generate a token using your LiveKit project credentials, **agent name = `smoke`**
4. Click Connect
5. Talk through your microphone

The agent will greet you and respond through Fish Audio's voice. Try interrupting it mid-sentence to test turn detection.

## What success looks like

You should see, in the worker terminal:

- `fish_tts.ready` — TTS initialized
- `llm.build  backend=livekit` — LLM constructed
- `pipeline.build_start` / `pipeline.build_done` — pipeline assembled
- After your first utterance:
  - `fish_stt.transcript` — Fish transcribed your speech
  - `fish_tts.first_audio_frame` — Fish produced the first audio frame; latency in `stream_open_to_audio_ms`
  - `fish_tts.metrics` — full TTS metrics (TTFB, RTF, chars)

A successful run proves:
- Fish STT can transcribe your speech
- LLM (LiveKit Inference) generates replies
- Fish TTS synthesizes audio at < 500ms TTFB
- VAD + turn detection drive natural turn-taking
- The whole pipeline assembles via one `build_pipeline()` call

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `FISH_API_KEY is required` | `.env` not loaded — check it's at `examples/.env` |
| `LIVEKIT_API_KEY and LIVEKIT_API_SECRET are both required` | Same — also need API secret, not just key |
| Agent connects but won't speak | Mic muted in Playground, or VAD not detecting speech (check `vad` logs) |
| `Playback failed` in Playground | Browser audio output muted/restricted |
| High `stream_open_to_audio_ms` | Network latency to Fish — check `fish_tts.metrics.ttfb_ms` to isolate |

## Want notifications?

This example doesn't use `SlackNotifier`. To exercise that path too, add a `@function_tool` that calls `notifier.send(...)` on a Slack-configured notifier — see `voice_agent_core/notify.py` docstring for usage.
