"""Central assistant state manager for realtime HUD and voice coordination."""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any


VALID_STATES = {
    "idle",
    "listening",
    "thinking",
    "speaking",
    "executing",
    "interrupted",
    "offline",
    "recovering",
}


@dataclass
class StateSnapshot:
    state: str = "idle"
    previous_state: str = "idle"
    reason: str = "startup"
    source: str = "state_manager"
    updated_at: float = field(default_factory=time.time)
    correlation_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class AssistantStateManager:
    """Single backend-owned state source for the cognitive runtime."""

    def __init__(self):
        self._lock = threading.RLock()
        self._snapshot = StateSnapshot()

    def set_state(
        self,
        state: str,
        *,
        reason: str = "",
        source: str = "state_manager",
        correlation_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized = state.strip().lower()
        if normalized not in VALID_STATES:
            raise ValueError(f"Invalid assistant state: {state}")
        with self._lock:
            previous = self._snapshot.state
            self._snapshot = StateSnapshot(
                state=normalized,
                previous_state=previous,
                reason=reason or normalized,
                source=source,
                correlation_id=correlation_id,
                metadata=metadata or {},
            )
            payload = asdict(self._snapshot)
        try:
            from event_bus import emit

            emit("assistant_state_changed", payload, priority=2, source=source)
        except Exception:
            pass
        return payload

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            return asdict(self._snapshot)


_manager: AssistantStateManager | None = None
_lock = threading.Lock()


def get_assistant_state() -> AssistantStateManager:
    global _manager
    with _lock:
        if _manager is None:
            _manager = AssistantStateManager()
        return _manager
