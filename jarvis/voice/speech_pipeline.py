"""Voice pipeline helpers used by the desktop runtime."""

from __future__ import annotations


def run_voice_command(text: str, speak: bool = True) -> dict:
    from core.task_router import route_text

    return route_text(text, speak=speak)

