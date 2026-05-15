"""
Async event bus for JARVIS Phase 1.

The bus is intentionally small: producers publish Event objects, consumers
subscribe with sync or async callbacks, and dispatch happens on the asyncio
loop without blocking perception modules.
"""

from __future__ import annotations

import asyncio
import fnmatch
import inspect
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


EventHandler = Callable[["Event"], Awaitable[None] | None]


@dataclass(order=True)
class _QueueItem:
    priority: int
    sequence: int
    event: "Event" = field(compare=False)


@dataclass(slots=True)
class Event:
    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    source: str = "system"
    priority: int = 5
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.event_id,
            "name": self.name,
            "payload": self.payload,
            "source": self.source,
            "priority": self.priority,
            "timestamp": self.timestamp,
        }


@dataclass(slots=True)
class Subscription:
    token: str
    pattern: str
    handler: EventHandler
    once: bool = False


class AsyncEventBus:
    """Priority pub/sub bus for the asyncio runtime."""

    def __init__(self, history_limit: int = 1000):
        self._history: deque[Event] = deque(maxlen=history_limit)
        self._subscriptions: dict[str, Subscription] = {}
        self._queue: asyncio.PriorityQueue[_QueueItem] = asyncio.PriorityQueue()
        self._sequence = 0
        self._running = False
        self._worker: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if self._worker and not self._worker.done():
            return
        self._running = True
        self._worker = asyncio.create_task(self._dispatch_loop(), name="jarvis-async-event-bus")

    async def stop(self) -> None:
        self._running = False
        await self.publish("event_bus.shutdown", source="async_event_bus", priority=9999)
        if self._worker:
            await self._worker

    async def publish(
        self,
        name: str,
        payload: dict[str, Any] | None = None,
        *,
        source: str = "system",
        priority: int = 5,
    ) -> Event:
        event = Event(name=name, payload=payload or {}, source=source, priority=priority)
        async with self._lock:
            self._history.append(event)
            self._sequence += 1
            sequence = self._sequence
        await self._queue.put(_QueueItem(priority=priority, sequence=sequence, event=event))
        return event

    async def subscribe(self, pattern: str, handler: EventHandler, *, once: bool = False) -> str:
        token = uuid.uuid4().hex
        async with self._lock:
            self._subscriptions[token] = Subscription(token, pattern, handler, once)
        return token

    async def unsubscribe(self, token: str) -> bool:
        async with self._lock:
            return self._subscriptions.pop(token, None) is not None

    async def replay(self, pattern: str = "*", limit: int | None = None) -> list[Event]:
        async with self._lock:
            events = [event for event in self._history if fnmatch.fnmatch(event.name, pattern)]
        return events[-limit:] if limit else events

    async def drain(self) -> None:
        """Wait until all queued events have been dispatched."""
        await self._queue.join()

    async def _dispatch_loop(self) -> None:
        while self._running or not self._queue.empty():
            item = await self._queue.get()
            try:
                await self._dispatch(item.event)
            finally:
                self._queue.task_done()

    async def _dispatch(self, event: Event) -> None:
        async with self._lock:
            matches = [
                sub for sub in self._subscriptions.values()
                if fnmatch.fnmatch(event.name, sub.pattern)
            ]
        expired: list[str] = []
        for sub in matches:
            result = sub.handler(event)
            if inspect.isawaitable(result):
                await result
            if sub.once:
                expired.append(sub.token)
        for token in expired:
            await self.unsubscribe(token)


_bus: AsyncEventBus | None = None


def get_async_event_bus() -> AsyncEventBus:
    global _bus
    if _bus is None:
        _bus = AsyncEventBus()
    return _bus
