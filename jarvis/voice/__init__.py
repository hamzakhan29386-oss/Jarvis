"""Native voice package with compatibility exports for the legacy voice.py API."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

_LEGACY: ModuleType | None = None


def _legacy() -> ModuleType:
    global _LEGACY
    if _LEGACY is None:
        legacy_path = Path(__file__).resolve().parent.parent / "voice.py"
        spec = importlib.util.spec_from_file_location("_jarvis_legacy_voice", legacy_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load legacy voice module from {legacy_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _LEGACY = module
    return _LEGACY


def get_voice_engine(*args, **kwargs):
    return _legacy().get_voice_engine(*args, **kwargs)


def listen(*args, **kwargs):
    return _legacy().listen(*args, **kwargs)


def speak(*args, **kwargs):
    return _legacy().speak(*args, **kwargs)


def get_wake_service(*args, **kwargs):
    from wake.service import get_wake_service as _get_wake_service

    return _get_wake_service(*args, **kwargs)


def __getattr__(name: str):
    if name == "StreamingSpeaker":
        return getattr(_legacy(), name)
    raise AttributeError(name)


__all__ = ["get_voice_engine", "listen", "speak", "get_wake_service", "StreamingSpeaker"]
