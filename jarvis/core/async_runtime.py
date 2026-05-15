"""
Phase 1 asyncio runtime wiring for JARVIS.

This runtime can run beside the existing Flask/threaded desktop app while
perception modules are migrated one by one to publish events.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from core.async_cognition import AsyncCognitionEngine, CognitionConfig
from core.async_event_bus import AsyncEventBus, get_async_event_bus
from core.vector_memory import ChromaVectorMemory


class AsyncJarvisRuntime:
    def __init__(
        self,
        bus: AsyncEventBus | None = None,
        memory: ChromaVectorMemory | None = None,
        cognition_config: CognitionConfig | None = None,
    ):
        self.bus = bus or get_async_event_bus()
        self.memory = memory or ChromaVectorMemory()
        self.cognition = AsyncCognitionEngine(self.bus, self.memory, cognition_config)

    async def start(self) -> None:
        await self.bus.start()
        await self.cognition.start()

    async def stop(self) -> None:
        await self.cognition.stop()
        await self.bus.stop()

    async def submit_user_command(self, text: str) -> None:
        await self.bus.publish("user.command", {"text": text}, source="runtime", priority=3)

    async def wait_idle(self) -> None:
        await self.bus.drain()
        await self.cognition.wait_pending()


@asynccontextmanager
async def running_runtime(
    cognition_config: CognitionConfig | None = None,
) -> AsyncIterator[AsyncJarvisRuntime]:
    runtime = AsyncJarvisRuntime(cognition_config=cognition_config)
    await runtime.start()
    try:
        yield runtime
    finally:
        await runtime.stop()


async def demo_once(text: str) -> list[dict]:
    async with running_runtime() as runtime:
        captured: list[dict] = []

        async def capture(event):
            captured.append(event.to_dict())

        await runtime.bus.subscribe("cognition.completed", capture)
        await runtime.submit_user_command(text)
        await runtime.wait_idle()
        return captured
