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
from voice_agent_core.observability import (
    MetricNames,
    configure_logging,
    configure_metrics,
    get_logger,
    get_meter,
    setup_observability,
    shutdown_observability,
)

__version__ = "0.1.0"

__all__ = [
    "BaseAgentSettings",
    "FishSTT",
    "FishTTS",
    "LLMBackend",
    "LogFormat",
    "LogLevel",
    "MetricNames",
    "OTelExporter",
    "TTSLatencyMode",
    "__version__",
    "build_fish_stt",
    "build_fish_tts",
    "configure_logging",
    "configure_metrics",
    "get_logger",
    "get_meter",
    "load_env_walking_up",
    "load_yaml",
    "setup_observability",
    "shutdown_observability",
]
