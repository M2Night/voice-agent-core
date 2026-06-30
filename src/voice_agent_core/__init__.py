"""Public convenience API for voice-agent-core.

The package root intentionally lazy-loads submodules. A plain
``import voice_agent_core`` should stay cheap and avoid importing optional
provider stacks; ``from voice_agent_core import build_pipeline`` keeps the
same public API and loads only the module that owns the requested symbol.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from voice_agent_core._version import __version__

_LAZY_EXPORTS = {
    "BaseAgentSettings": "voice_agent_core.config",
    "CustomLLMSettings": "voice_agent_core.llm",
    "DEFAULT_SUMMARY_INSTRUCTION": "voice_agent_core.transcript",
    "DeepgramSettings": "voice_agent_core.deepgram",
    "FishSTT": "voice_agent_core.fish",
    "FishSettings": "voice_agent_core.fish",
    "FishTTS": "voice_agent_core.fish",
    "FishTTSLatencyMode": "voice_agent_core.fish",
    "IdleWatcher": "voice_agent_core.idle",
    "InworldSettings": "voice_agent_core.inworld",
    "InworldTTSDeliveryMode": "voice_agent_core.inworld",
    "LLMProvider": "voice_agent_core.providers",
    "LogFormat": "voice_agent_core.config",
    "LogLevel": "voice_agent_core.config",
    "MetricNames": "voice_agent_core.observability",
    "NotificationField": "voice_agent_core.notify",
    "NotificationPayload": "voice_agent_core.notify",
    "Notifier": "voice_agent_core.notify",
    "OTelExporter": "voice_agent_core.config",
    "OnIdle": "voice_agent_core.idle",
    "OpenRouterSettings": "voice_agent_core.llm",
    "PipelineComponents": "voice_agent_core.pipeline",
    "STTProvider": "voice_agent_core.providers",
    "SlackNotifier": "voice_agent_core.notify",
    "StreamAdapter": "voice_agent_core.stt",
    "TTSProvider": "voice_agent_core.providers",
    "attach_idle_watcher": "voice_agent_core.idle",
    "build_deepgram_stt": "voice_agent_core.deepgram",
    "build_fish_stt": "voice_agent_core.fish",
    "build_fish_tts": "voice_agent_core.fish",
    "build_inworld_tts": "voice_agent_core.inworld",
    "build_llm": "voice_agent_core.providers",
    "build_pipeline": "voice_agent_core.pipeline",
    "build_session": "voice_agent_core.runtime",
    "build_stt": "voice_agent_core.providers",
    "build_tts": "voice_agent_core.providers",
    "configure_logging": "voice_agent_core.observability",
    "configure_metrics": "voice_agent_core.observability",
    "default_prewarm": "voice_agent_core.runtime",
    "default_room_options": "voice_agent_core.runtime",
    "format_transcript": "voice_agent_core.transcript",
    "get_logger": "voice_agent_core.observability",
    "get_meter": "voice_agent_core.observability",
    "is_warmup_session": "voice_agent_core.runtime",
    "list_llm_providers": "voice_agent_core.providers",
    "list_stt_providers": "voice_agent_core.providers",
    "list_tts_providers": "voice_agent_core.providers",
    "llm_models": "voice_agent_core.providers",
    "load_env_walking_up": "voice_agent_core.config",
    "load_yaml": "voice_agent_core.config",
    "provider_config_schema": "voice_agent_core.providers",
    "register_llm": "voice_agent_core.providers",
    "register_stt": "voice_agent_core.providers",
    "register_tts": "voice_agent_core.providers",
    "setup_observability": "voice_agent_core.observability",
    "shutdown_observability": "voice_agent_core.observability",
    "stt_models": "voice_agent_core.providers",
    "summarize_transcript": "voice_agent_core.transcript",
    "tts_models": "voice_agent_core.providers",
    "warm_tts": "voice_agent_core.runtime",
}

_SUBMODULES = {
    "config",
    "deepgram",
    "fish",
    "idle",
    "inworld",
    "llm",
    "notify",
    "observability",
    "pipeline",
    "providers",
    "runtime",
    "stt",
    "transcript",
}

__all__ = [*_LAZY_EXPORTS, "__version__"]


def __getattr__(name: str) -> Any:
    """Resolve public re-exports lazily on first access."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None and name in _SUBMODULES:
        module_name = f"{__name__}.{name}"
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name)
    value = module if name in _SUBMODULES else getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
