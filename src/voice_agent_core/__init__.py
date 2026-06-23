"""voice_agent_core — shared infrastructure for Fish Audio voice agent.

Public API re-exported here for convenience::

    from voice_agent_core import (
        BaseAgentSettings,
        load_yaml,
        load_env_walking_up,
        setup_observability,
        get_logger,
        get_meter,
        MetricNames,
    )
"""

from voice_agent_core.config import (
    BaseAgentSettings,
    LogFormat,
    LogLevel,
    OTelExporter,
    load_env_walking_up,
    load_yaml,
)
from voice_agent_core.deepgram import DeepgramSettings, build_deepgram_stt
from voice_agent_core.fish import (
    FishSettings,
    FishSTT,
    FishTTS,
    FishTTSLatencyMode,
    build_fish_stt,
    build_fish_tts,
)
from voice_agent_core.idle import IdleWatcher, OnIdle, attach_idle_watcher
from voice_agent_core.llm import OpenRouterSettings
from voice_agent_core.notify import (
    NotificationField,
    NotificationPayload,
    SlackNotifier,
)
from voice_agent_core.observability import (
    MetricNames,
    configure_logging,
    configure_metrics,
    get_logger,
    get_meter,
    setup_observability,
    shutdown_observability,
)
from voice_agent_core.pipeline import PipelineComponents, build_pipeline
from voice_agent_core.providers import (
    LLMProvider,
    STTProvider,
    TTSProvider,
    build_llm,
    build_stt,
    build_tts,
    list_llm_providers,
    list_stt_providers,
    list_tts_providers,
    llm_models,
    provider_config_schema,
    register_llm,
    register_stt,
    register_tts,
    stt_models,
    tts_models,
)
from voice_agent_core.runtime import (
    build_session,
    default_prewarm,
    default_room_options,
    is_warmup_session,
    warm_tts,
)
from voice_agent_core.stt import StreamAdapter
from voice_agent_core.transcript import (
    DEFAULT_SUMMARY_INSTRUCTION,
    format_transcript,
    summarize_transcript,
)

__version__ = "0.2.1"

__all__ = [
    "DEFAULT_SUMMARY_INSTRUCTION",
    "BaseAgentSettings",
    "DeepgramSettings",
    "FishSTT",
    "FishSettings",
    "FishTTS",
    "FishTTSLatencyMode",
    "IdleWatcher",
    "LLMProvider",
    "LogFormat",
    "LogLevel",
    "MetricNames",
    "NotificationField",
    "NotificationPayload",
    "OTelExporter",
    "OnIdle",
    "OpenRouterSettings",
    "PipelineComponents",
    "STTProvider",
    "SlackNotifier",
    "StreamAdapter",
    "TTSProvider",
    "__version__",
    "attach_idle_watcher",
    "build_deepgram_stt",
    "build_fish_stt",
    "build_fish_tts",
    "build_llm",
    "build_pipeline",
    "build_session",
    "build_stt",
    "build_tts",
    "configure_logging",
    "configure_metrics",
    "default_prewarm",
    "default_room_options",
    "format_transcript",
    "get_logger",
    "get_meter",
    "is_warmup_session",
    "list_llm_providers",
    "list_stt_providers",
    "list_tts_providers",
    "llm_models",
    "load_env_walking_up",
    "load_yaml",
    "provider_config_schema",
    "register_llm",
    "register_stt",
    "register_tts",
    "setup_observability",
    "shutdown_observability",
    "stt_models",
    "summarize_transcript",
    "tts_models",
    "warm_tts",
]
