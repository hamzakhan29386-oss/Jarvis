"""
Async cognition engine for Phase 1.

Perception modules publish events such as user.command or perception.screen.
The cognition engine consumes those events without blocking the event bus and
publishes result events for UI, voice, memory, and automation subscribers.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass
from typing import Any

from core.async_event_bus import AsyncEventBus, Event
from core.task_router import get_task_router
from core.vector_memory import ChromaVectorMemory

log = logging.getLogger("jarvis.core.async_cognition")


@dataclass(slots=True)
class CognitionConfig:
    memory_top_k: int = 5
    speak: bool = False


class AsyncCognitionEngine:
    """Event-driven coordinator for routing, memory, and background thought."""

    def __init__(
        self,
        bus: AsyncEventBus,
        memory: ChromaVectorMemory,
        config: CognitionConfig | None = None,
    ):
        self.bus = bus
        self.memory = memory
        self.config = config or CognitionConfig()
        self._tasks: set[asyncio.Task[Any]] = set()
        self._subscription_tokens: list[str] = []

    async def start(self) -> None:
        await self.memory.initialize()
        self._subscription_tokens.append(await self.bus.subscribe("user.command", self._schedule))
        self._subscription_tokens.append(await self.bus.subscribe("perception.*", self._schedule))
        self._subscription_tokens.append(await self.bus.subscribe("memory.query", self._schedule))
        await self.bus.publish("cognition.started", source="async_cognition")

    async def stop(self) -> None:
        for token in self._subscription_tokens:
            await self.bus.unsubscribe(token)
        self._subscription_tokens.clear()
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.bus.publish("cognition.stopped", source="async_cognition")

    async def wait_pending(self) -> None:
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    def _schedule(self, event: Event) -> None:
        task = asyncio.create_task(self._handle(event), name=f"jarvis-cognition-{event.name}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _handle(self, event: Event) -> None:
        try:
            if event.name == "user.command":
                await self._handle_user_command(event)
            elif event.name == "memory.query":
                await self._handle_memory_query(event)
            elif event.name.startswith("perception."):
                await self._handle_perception(event)
        except Exception as exc:
            log.exception("Cognition handler failed for %s", event.name)
            await self.bus.publish(
                "cognition.error",
                {"event": event.to_dict(), "error": str(exc)},
                source="async_cognition",
                priority=2,
            )

    async def _handle_user_command(self, event: Event) -> None:
        text = str(event.payload.get("text", "")).strip()
        if not text:
            return
        await self.bus.publish("cognition.started_command", {"text": text}, source="async_cognition", priority=3)
        memories = await self.memory.search(text, top_k=self.config.memory_top_k)
        result = await asyncio.to_thread(get_task_router().route, text, self.config.speak)
        await self.memory.remember(
            f"User: {text}\nJARVIS: {result.response}",
            metadata={
                "source": "async_cognition",
                "event_id": event.event_id,
                "tier": result.tier or "",
                "model": result.model or "",
            },
        )
        await self.bus.publish(
            "cognition.completed",
            {
                "text": text,
                "response": result.response,
                "result": result.to_dict(),
                "memory_matches": [asdict(record) for record in memories],
            },
            source="async_cognition",
            priority=3,
        )

    async def _handle_memory_query(self, event: Event) -> None:
        query = str(event.payload.get("query", "")).strip()
        if not query:
            return
        records = await self.memory.search(query, top_k=int(event.payload.get("top_k", self.config.memory_top_k)))
        await self.bus.publish(
            "memory.results",
            {"query": query, "records": [asdict(record) for record in records]},
            source="async_cognition",
        )

    async def _handle_perception(self, event: Event) -> None:
        summary = event.payload.get("summary") or event.payload.get("text")
        if not summary:
            return
        await self.memory.remember(
            str(summary),
            metadata={"source": event.name, "event_id": event.event_id},
        )
        await self.bus.publish(
            "perception.remembered",
            {"source_event": event.to_dict()},
            source="async_cognition",
        )
