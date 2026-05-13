"""Compatibility service for the native Python-owned JARVIS voice runtime."""

from __future__ import annotations

import threading
from typing import Iterator

from voice.voice_manager import VoiceManager, get_voice_manager


class ProductionWakeService:
    """Flask-facing wake service API backed by voice.voice_manager."""

    def __init__(self, manager: VoiceManager | None = None):
        self._manager = manager or get_voice_manager()

    def enable(self):
        self._manager.start()

    def disable(self):
        self._manager.stop()

    def restart(self):
        self._manager.restart()

    def is_enabled(self) -> bool:
        return self._manager.is_enabled()

    def subscribe(self) -> Iterator[dict]:
        yield from self._manager.subscribe()

    def get_status(self) -> dict:
        return self._manager.get_status()

    def set_threshold(self, threshold: float):
        self._manager.set_threshold(threshold)

    def enroll_speaker(self):
        return self._manager.enroll_speaker()


_instance: ProductionWakeService | None = None
_lock = threading.Lock()


def get_wake_service() -> ProductionWakeService:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = ProductionWakeService()
    return _instance
