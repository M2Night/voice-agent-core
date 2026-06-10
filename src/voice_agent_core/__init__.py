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
    LLMBackend,
    LogFormat,
    LogLevel,
    OTelExporter,
    TTSLatencyMode,
    load_env_walking_up,
    load_yaml,
)
from voice_agent_core.fish import FishSTT, FishTTS, build_fish_stt, build_fish_tts
from voice_agent_core.llm import build_llm
from voice_agent_core.notify import (
    NotificationField,
    NotificationPayload,
    SlackNotifier,
)
from voice_agent_core.pipeline import PipelineComponents, build_pipeline
from voice_agent_core.runtime import (
    build_session,
    default_prewarm,
    default_room_options,
    is_warmup_session,
    warm_tts,
)
from voice_agent_core.transcript import (
    DEFAULT_SUMMARY_INSTRUCTION,
    format_transcript,
    summarize_transcript,
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

__version__ = "0.2.1"

__all__ = [
    "DEFAULT_SUMMARY_INSTRUCTION",
    "BaseAgentSettings",
    "FishSTT",
    "FishTTS",
    "LLMBackend",
    "LogFormat",
    "LogLevel",
    "MetricNames",
    "NotificationField",
    "NotificationPayload",
    "OTelExporter",
    "PipelineComponents",
    "SlackNotifier",
    "TTSLatencyMode",
    "__version__",
    "build_fish_stt",
    "build_fish_tts",
    "build_llm",
    "build_pipeline",
    "build_session",
    "configure_logging",
    "configure_metrics",
    "default_prewarm",
    "default_room_options",
    "format_transcript",
    "get_logger",
    "get_meter",
    "is_warmup_session",
    "load_env_walking_up",
    "load_yaml",
    "setup_observability",
    "shutdown_observability",
    "summarize_transcript",
    "warm_tts",
]
