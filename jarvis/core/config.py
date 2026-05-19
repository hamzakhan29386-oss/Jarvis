"""Typed environment configuration for the JARVIS runtime."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args, **_kwargs):
        return False


load_dotenv()


def _str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _int(name: str, default: int) -> int:
    try:
        return int(_str(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_str(name, str(default)))
    except ValueError:
        return default


def _bool(name: str, default: bool = False) -> bool:
    value = _str(name, str(default)).lower()
    return value in {"1", "true", "yes", "on", "enabled"}


def _list(name: str, default: str = "") -> list[str]:
    return [item.strip() for item in _str(name, default).split(",") if item.strip()]


@dataclass(frozen=True)
class RuntimeConfig:
    environment: str
    debug: bool
    log_level: str
    host: str
    port: int
    public_url: str
    websocket_enabled: bool
    websocket_path: str
    telemetry_enabled: bool
    realtime_events_enabled: bool


@dataclass(frozen=True)
class ProviderConfig:
    priority: list[str]
    local_first: bool
    cloud_fallback_enabled: bool
    request_timeout_seconds: float
    stream_timeout_seconds: float
    max_retries: int
    retry_backoff_seconds: float
    provider_cooldown_seconds: float
    circuit_breaker_enabled: bool
    openrouter_base_url: str
    nvidia_base_url: str
    ollama_base_url: str
    local_inference_base_url: str


@dataclass(frozen=True)
class VoiceConfig:
    enabled: bool
    audio_multiplexer_enabled: bool
    input_device: str
    output_device: str
    input_sample_rate: int
    target_sample_rate: int
    wake_model_path: Path
    wake_threshold: float
    vad_engine: str
    stt_engine: str
    whisper_model: str
    tts_engine: str
    barge_in_enabled: bool
    interrupt_detection_enabled: bool


@dataclass(frozen=True)
class MemoryConfig:
    enabled: bool
    memory_file: Path
    chroma_enabled: bool
    chroma_persist_dir: Path
    sqlite_memory_path: Path
    embedding_model: str
    top_k: int
    max_episodes: int
    consolidation_enabled: bool
    pruning_enabled: bool


@dataclass(frozen=True)
class PerceptionConfig:
    cowork_mode_enabled: bool
    screen_capture_enabled: bool
    screen_capture_backend: str
    screen_capture_fps: float
    ocr_enabled: bool
    ocr_engine: str
    vlm_enabled: bool
    vlm_endpoint: str
    active_window_tracking_enabled: bool
    browser_awareness_enabled: bool
    ide_awareness_enabled: bool
    screenshot_retention_policy: str


@dataclass(frozen=True)
class AutonomyConfig:
    world_model_enabled: bool
    workflow_learning_enabled: bool
    proactivity_level: str
    temporal_reasoning_enabled: bool
    autonomous_runtime_enabled: bool
    long_task_runtime_enabled: bool
    long_task_max_steps: int
    long_task_checkpoint_interval_seconds: int


@dataclass(frozen=True)
class PrivacyConfig:
    local_only_mode: bool
    allow_cloud_providers: bool
    sensitive_window_filter_enabled: bool
    ephemeral_perception_mode: bool
    token_redaction_enabled: bool
    memory_privacy_mode: str


@dataclass(frozen=True)
class HudConfig:
    enabled: bool
    mode: str
    telemetry_refresh_ms: int
    realtime_animations: bool
    cowork_overlay_enabled: bool
    visual_debug: bool


@dataclass(frozen=True)
class JarvisConfig:
    runtime: RuntimeConfig
    providers: ProviderConfig
    voice: VoiceConfig
    memory: MemoryConfig
    perception: PerceptionConfig
    autonomy: AutonomyConfig
    privacy: PrivacyConfig
    hud: HudConfig

    def to_safe_dict(self) -> dict:
        """Return config without secret values."""
        return asdict(self)


@lru_cache(maxsize=1)
def get_config() -> JarvisConfig:
    return JarvisConfig(
        runtime=RuntimeConfig(
            environment=_str("JARVIS_ENV", "development"),
            debug=_bool("JARVIS_DEBUG", False),
            log_level=_str("JARVIS_LOG_LEVEL", "INFO"),
            host=_str("JARVIS_HOST", "127.0.0.1"),
            port=_int("JARVIS_PORT", 5000),
            public_url=_str("JARVIS_PUBLIC_URL", "http://127.0.0.1:5000"),
            websocket_enabled=_bool("WEBSOCKET_ENABLED", True),
            websocket_path=_str("WEBSOCKET_PATH", "/ws/events"),
            telemetry_enabled=_bool("TELEMETRY_ENABLED", True),
            realtime_events_enabled=_bool("REALTIME_EVENTS_ENABLED", True),
        ),
        providers=ProviderConfig(
            priority=_list("AI_PROVIDER_PRIORITY", "ollama,local,openrouter,nvidia"),
            local_first=_bool("AI_LOCAL_FIRST", True),
            cloud_fallback_enabled=_bool("AI_CLOUD_FALLBACK_ENABLED", True),
            request_timeout_seconds=_float("AI_REQUEST_TIMEOUT_SECONDS", 30.0),
            stream_timeout_seconds=_float("AI_STREAM_TIMEOUT_SECONDS", 180.0),
            max_retries=_int("AI_MAX_RETRIES", 2),
            retry_backoff_seconds=_float("AI_RETRY_BACKOFF_SECONDS", 1.5),
            provider_cooldown_seconds=_float("AI_PROVIDER_COOLDOWN_SECONDS", 60.0),
            circuit_breaker_enabled=_bool("AI_CIRCUIT_BREAKER_ENABLED", True),
            openrouter_base_url=_str("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            nvidia_base_url=_str("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            ollama_base_url=_str("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            local_inference_base_url=_str("LOCAL_INFERENCE_BASE_URL", "http://127.0.0.1:8000/v1"),
        ),
        voice=VoiceConfig(
            enabled=_bool("VOICE_ENABLED", True),
            audio_multiplexer_enabled=_bool("AUDIO_MULTIPLEXER_ENABLED", True),
            input_device=_str("AUDIO_INPUT_DEVICE", "auto"),
            output_device=_str("AUDIO_OUTPUT_DEVICE", "auto"),
            input_sample_rate=_int("AUDIO_INPUT_SAMPLE_RATE", 48000),
            target_sample_rate=_int("AUDIO_TARGET_SAMPLE_RATE", 16000),
            wake_model_path=Path(_str("WAKE_MODEL_PATH", "wake/models/hey_jarvis_custom.onnx")),
            wake_threshold=_float("WAKE_THRESHOLD", 0.60),
            vad_engine=_str("VAD_ENGINE", "silero"),
            stt_engine=_str("STT_ENGINE", "faster-whisper"),
            whisper_model=_str("WHISPER_MODEL", "tiny.en"),
            tts_engine=_str("TTS_ENGINE", "piper"),
            barge_in_enabled=_bool("BARGE_IN_ENABLED", True),
            interrupt_detection_enabled=_bool("INTERRUPT_DETECTION_ENABLED", True),
        ),
        memory=MemoryConfig(
            enabled=_bool("MEMORY_ENABLED", True),
            memory_file=Path(_str("MEMORY_FILE", "memory_store.json")),
            chroma_enabled=_bool("CHROMA_ENABLED", True),
            chroma_persist_dir=Path(_str("CHROMA_PERSIST_DIR", "data/chroma")),
            sqlite_memory_path=Path(_str("SQLITE_MEMORY_PATH", "data/jarvis_memory.sqlite3")),
            embedding_model=_str("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            top_k=_int("MEMORY_TOP_K", 5),
            max_episodes=_int("MEMORY_MAX_EPISODES", 500),
            consolidation_enabled=_bool("MEMORY_CONSOLIDATION_ENABLED", True),
            pruning_enabled=_bool("MEMORY_PRUNING_ENABLED", True),
        ),
        perception=PerceptionConfig(
            cowork_mode_enabled=_bool("COWORK_MODE_ENABLED", False),
            screen_capture_enabled=_bool("SCREEN_CAPTURE_ENABLED", False),
            screen_capture_backend=_str("SCREEN_CAPTURE_BACKEND", "dxcam"),
            screen_capture_fps=_float("SCREEN_CAPTURE_FPS", 1.0),
            ocr_enabled=_bool("OCR_ENABLED", False),
            ocr_engine=_str("OCR_ENGINE", "paddleocr"),
            vlm_enabled=_bool("VLM_ENABLED", False),
            vlm_endpoint=_str("VLM_ENDPOINT", "http://127.0.0.1:8001/v1"),
            active_window_tracking_enabled=_bool("ACTIVE_WINDOW_TRACKING_ENABLED", True),
            browser_awareness_enabled=_bool("BROWSER_AWARENESS_ENABLED", True),
            ide_awareness_enabled=_bool("IDE_AWARENESS_ENABLED", True),
            screenshot_retention_policy=_str("SCREENSHOT_RETENTION_POLICY", "summaries_only"),
        ),
        autonomy=AutonomyConfig(
            world_model_enabled=_bool("WORLD_MODEL_ENABLED", True),
            workflow_learning_enabled=_bool("WORKFLOW_LEARNING_ENABLED", True),
            proactivity_level=_str("PROACTIVITY_LEVEL", "contextual_nudges"),
            temporal_reasoning_enabled=_bool("TEMPORAL_REASONING_ENABLED", True),
            autonomous_runtime_enabled=_bool("AUTONOMOUS_RUNTIME_ENABLED", True),
            long_task_runtime_enabled=_bool("LONG_TASK_RUNTIME_ENABLED", True),
            long_task_max_steps=_int("LONG_TASK_MAX_STEPS", 50),
            long_task_checkpoint_interval_seconds=_int("LONG_TASK_CHECKPOINT_INTERVAL_SECONDS", 300),
        ),
        privacy=PrivacyConfig(
            local_only_mode=_bool("LOCAL_ONLY_MODE", False),
            allow_cloud_providers=_bool("ALLOW_CLOUD_PROVIDERS", True),
            sensitive_window_filter_enabled=_bool("SENSITIVE_WINDOW_FILTER_ENABLED", True),
            ephemeral_perception_mode=_bool("EPHEMERAL_PERCEPTION_MODE", True),
            token_redaction_enabled=_bool("TOKEN_REDACTION_ENABLED", True),
            memory_privacy_mode=_str("MEMORY_PRIVACY_MODE", "summaries_only"),
        ),
        hud=HudConfig(
            enabled=_bool("HUD_ENABLED", True),
            mode=_str("HUD_MODE", "ironman"),
            telemetry_refresh_ms=_int("HUD_TELEMETRY_REFRESH_MS", 5000),
            realtime_animations=_bool("HUD_REALTIME_ANIMATIONS", True),
            cowork_overlay_enabled=_bool("HUD_COWORK_OVERLAY_ENABLED", False),
            visual_debug=_bool("HUD_VISUAL_DEBUG", False),
        ),
    )
