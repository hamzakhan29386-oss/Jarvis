"""
Async JARVIS entry point.

Run with:
    python main.py

The console is intentionally thin: user input becomes a user.command event,
the cognition engine handles routing/memory, and subscribers receive results.
"""

from __future__ import annotations

import asyncio

from core.async_runtime import AsyncJarvisRuntime


EXIT_WORDS = {"exit", "quit", "bye", "goodbye", "stop", "shutdown", "shut down"}


async def _read_line(prompt: str) -> str:
    return await asyncio.to_thread(input, prompt)


async def main() -> None:
    runtime = AsyncJarvisRuntime()
    completed = asyncio.Event()

    async def print_response(event):
        response = event.payload.get("response", "")
        if response:
            print(f"\nJARVIS > {response}\n")
        completed.set()

    async def print_error(event):
        print(f"\nJARVIS error > {event.payload.get('error', 'unknown error')}\n")
        completed.set()

    try:
        await runtime.start()
    except RuntimeError as exc:
        print(str(exc))
        print("Install vector memory dependencies with: pip install chromadb")
        return

    await runtime.bus.subscribe("cognition.completed", print_response)
    await runtime.bus.subscribe("cognition.error", print_error)

    print("JARVIS async runtime online. Type 'exit' to quit.")
    try:
        while True:
            text = (await _read_line("You > ")).strip()
            if not text:
                continue
            if text.lower() in EXIT_WORDS:
                break
            completed.clear()
            await runtime.submit_user_command(text)
            await completed.wait()
    finally:
        await runtime.stop()


if __name__ == "__main__":
    asyncio.run(main())
