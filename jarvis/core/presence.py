"""Ambient presence and conversational acknowledgement layer."""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from event_bus import emit


@dataclass
class PresenceSnapshot:
    enabled: bool = True
    mode: str = "subtle_presence"
    voice_profile: str = "assist"
    last_ack_at: float | None = None
    last_ack: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class ConversationalCadenceEngine:
    def choose_pause_ms(self, state: str) -> int:
        return {"thinking": 250, "speaking": 120, "executing": 400}.get(state, 180)


class DynamicVoiceProfiles:
    def profile_for_mode(self, mode: str) -> str:
        return {
            "FOCUS": "minimal",
            "CODER": "technical",
            "RESEARCH": "analytical",
            "ASSIST": "assist",
        }.get(mode.upper(), "assist")


class AmbientAwarenessLayer:
    def describe(self) -> dict[str, Any]:
        try:
            from core.assistant_state import get_assistant_state

            return {"assistant_state": get_assistant_state().get_state()}
        except Exception:
            return {"assistant_state": {"state": "unknown"}}


class AdaptiveAcknowledgementSystem:
    def __init__(self):
        self.messages = {
            "idle": "Standing by.",
            "listening": "I'm listening.",
            "thinking": "Working on it.",
            "executing": "Executing.",
            "recovering": "Recovering the runtime.",
            "interrupted": "Interrupted.",
        }

    def message_for_state(self, state: str) -> str:
        return self.messages.get(state, self.messages["idle"])


class PresenceManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._snapshot = PresenceSnapshot()
        self.cadence = ConversationalCadenceEngine()
        self.voice_profiles = DynamicVoiceProfiles()
        self.ambient = AmbientAwarenessLayer()
        self.acknowledgements = AdaptiveAcknowledgementSystem()

    def status(self) -> dict[str, Any]:
        with self._lock:
            payload = asdict(self._snapshot)
        payload["ambient"] = self.ambient.describe()
        payload["cadence_pause_ms"] = self.cadence.choose_pause_ms(
            payload.get("ambient", {}).get("assistant_state", {}).get("state", "idle")
        )
        return payload

    def set_profile(self, *, mode: str | None = None, voice_profile: str | None = None) -> dict[str, Any]:
        with self._lock:
            if mode:
                self._snapshot.mode = mode
            if voice_profile:
                self._snapshot.voice_profile = voice_profile
            payload = asdict(self._snapshot)
        emit("presence_profile_updated", payload, source="presence")
        return payload

    def acknowledge(self, state: str = "idle", *, force: bool = False) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            if not force and self._snapshot.last_ack_at and now - self._snapshot.last_ack_at < 20:
                return {"spoken": False, "reason": "throttled", "presence": asdict(self._snapshot)}
            text = self.acknowledgements.message_for_state(state)
            self._snapshot.last_ack_at = now
            self._snapshot.last_ack = text
            payload = asdict(self._snapshot)
        emit("presence_acknowledgement", {"state": state, "text": text, "presence": payload}, source="presence")
        return {"spoken": True, "text": text, "presence": payload}


_presence: PresenceManager | None = None
_lock = threading.Lock()


def get_presence_manager() -> PresenceManager:
    global _presence
    with _lock:
        if _presence is None:
            _presence = PresenceManager()
        return _presence
