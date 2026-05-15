"""
event_bus.py - JARVIS central nervous system.

Thread-safe event dispatch with priority delivery, wildcard subscriptions,
history, replay, and one-shot listeners. Modules can use the singleton
`get_event_bus()` or the convenience functions at the bottom of this file.
"""

from __future__ import annotations

import fnmatch
import logging
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional

log = logging.getLogger("jarvis.event_bus")

EventCallback = Callable[["Event"], None]


@dataclass(order=True)
class _QueuedEvent:
    priority_key: int
    sequence: int
    event: "Event" = field(compare=False)


@dataclass
class Event:
    name: str
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5
    source: str = "system"
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.event_id,
            "name": self.name,
            "payload": self.payload,
            "priority": self.priority,
            "source": self.source,
            "timestamp": self.timestamp,
        }


@dataclass
class Subscription:
    token: str
    pattern: str
    callback: EventCallback
    once: bool = False


class EventBus:
    def __init__(self, history_limit: int = 1000):
        self._history: Deque[Event] = deque(maxlen=history_limit)
        self._subscriptions: Dict[str, Subscription] = {}
        self._lock = threading.RLock()
        self._queue: "queue.PriorityQueue[_QueuedEvent]" = queue.PriorityQueue()
        self._sequence = 0
        self._running = True
        self._worker = threading.Thread(target=self._dispatch_loop, name="jarvis-event-bus", daemon=True)
        self._worker.start()

    def emit(
        self,
        name: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        priority: int = 5,
        source: str = "system",
        sync: bool = False,
    ) -> Event:
        event = Event(name=name, payload=payload or {}, priority=priority, source=source)
        with self._lock:
            self._history.append(event)
            self._sequence += 1
            item = _QueuedEvent(priority_key=priority, sequence=self._sequence, event=event)
        if sync:
            self._dispatch(event)
        else:
            self._queue.put(item)
        return event

    def subscribe(self, pattern: str, callback: EventCallback, *, once: bool = False) -> str:
        token = uuid.uuid4().hex
        with self._lock:
            self._subscriptions[token] = Subscription(token, pattern, callback, once)
        return token

    def once(self, pattern: str, callback: EventCallback) -> str:
        return self.subscribe(pattern, callback, once=True)

    def unsubscribe(self, token: str) -> bool:
        with self._lock:
            return self._subscriptions.pop(token, None) is not None

    def replay_events(
        self,
        pattern: str = "*",
        *,
        limit: Optional[int] = None,
        since: Optional[float] = None,
        callback: Optional[EventCallback] = None,
    ) -> List[Event]:
        with self._lock:
            events = [
                event for event in self._history
                if fnmatch.fnmatch(event.name, pattern) and (since is None or event.timestamp >= since)
            ]
        if limit:
            events = events[-limit:]
        if callback:
            for event in events:
                callback(event)
        return events

    def history(self, limit: int = 100) -> List[Dict[str, Any]]:
        return [event.to_dict() for event in self.replay_events(limit=limit)]

    def shutdown(self) -> None:
        self._running = False
        self._queue.put(_QueuedEvent(9999, 0, Event("event_bus_shutdown")))

    def _dispatch_loop(self) -> None:
        while self._running:
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._dispatch(item.event)
            finally:
                self._queue.task_done()

    def _matching_subscriptions(self, event: Event) -> List[Subscription]:
        with self._lock:
            return [
                sub for sub in self._subscriptions.values()
                if fnmatch.fnmatch(event.name, sub.pattern)
            ]

    def _dispatch(self, event: Event) -> None:
        expired: List[str] = []
        for sub in self._matching_subscriptions(event):
            try:
                sub.callback(event)
            except Exception as exc:
                log.warning("[EventBus] Listener failed for %s: %s", event.name, exc)
            if sub.once:
                expired.append(sub.token)
        for token in expired:
            self.unsubscribe(token)


_bus: Optional[EventBus] = None
_bus_lock = threading.Lock()


def get_event_bus() -> EventBus:
    global _bus
    with _bus_lock:
        if _bus is None:
            _bus = EventBus()
        return _bus


def emit(name: str, payload: Optional[Dict[str, Any]] = None, **kwargs) -> Event:
    return get_event_bus().emit(name, payload, **kwargs)


def subscribe(pattern: str, callback: EventCallback, **kwargs) -> str:
    return get_event_bus().subscribe(pattern, callback, **kwargs)


def unsubscribe(token: str) -> bool:
    return get_event_bus().unsubscribe(token)


def once(pattern: str, callback: EventCallback) -> str:
    return get_event_bus().once(pattern, callback)


def replay_events(*args, **kwargs) -> List[Event]:
    return get_event_bus().replay_events(*args, **kwargs)
