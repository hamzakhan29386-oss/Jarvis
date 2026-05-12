"""
Runtime task router.

All desktop requests flow through this small layer:
user text -> structured intent -> actions.py tool execution or brain.py.
"""

from __future__ import annotations

import logging
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from core.intent_parser import ParsedIntent, parse_user_intent

log = logging.getLogger("jarvis.core.task_router")


@dataclass
class AssistantResult:
    ok: bool
    intent: dict[str, Any]
    response: str
    action_result: str | None = None
    model: str | None = None
    tier: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TaskRouter:
    """Coordinates fast local actions with brain.py fallback reasoning."""

    def parse(self, text: str) -> ParsedIntent:
        return parse_user_intent(text)

    def route(self, text: str, speak: bool = False) -> AssistantResult:
        intent = self.parse(text)
        if intent.needs_ai or intent.action == "chat":
            return self._ask_brain(text, intent, speak=speak)
        return self._execute_intent(intent, speak=speak)

    def _execute_intent(self, intent: ParsedIntent, speak: bool = False) -> AssistantResult:
        from actions import execute_action

        result = execute_action(intent.action, intent.args)
        response = self._friendly_action_response(intent, result)
        if speak:
            self._speak_async(response)
        return AssistantResult(
            ok=not self._looks_failed(result),
            intent=intent.to_dict(),
            response=response,
            action_result=result,
            tier="action",
        )

    def _ask_brain(self, text: str, intent: ParsedIntent, speak: bool = False) -> AssistantResult:
        from brain import think

        result = think(text)
        response = result.get("response", "")
        if speak:
            self._speak_async(response)
        return AssistantResult(
            ok=True,
            intent=intent.to_dict(),
            response=response,
            model=result.get("model"),
            tier=result.get("tier"),
            metadata=result,
        )

    def _friendly_action_response(self, intent: ParsedIntent, result: str) -> str:
        if intent.action == "get_system_status":
            try:
                data = json.loads(result)
                return (
                    f"CPU {data.get('cpu_percent', '?')}%, "
                    f"RAM {data.get('ram_percent', '?')}%, "
                    f"disk {data.get('disk_percent', '?')}%, "
                    f"battery {data.get('battery_percent', 'AC')}%."
                )
            except Exception:
                return str(result)
        if intent.action == "youtube_play":
            return f"Playing {intent.args.get('query', 'that')} on YouTube."
        if intent.action == "youtube_search":
            return f"Searching YouTube for {intent.args.get('query', 'that')}."
        if intent.action == "browser_google_search":
            return f"Searching Google for {intent.args.get('query', 'that')}."
        if intent.action == "run_workflow":
            return f"Workflow started: {intent.args.get('name', 'custom')}."
        return str(result)

    def _speak_async(self, text: str) -> None:
        try:
            from voice import speak

            speak(text)
        except Exception as exc:
            log.debug("TTS skipped: %s", exc)

    @staticmethod
    def _looks_failed(result: str) -> bool:
        value = str(result).lower()
        return any(word in value for word in ("failed", "error", "could not", "unknown action"))


_router: TaskRouter | None = None


def get_task_router() -> TaskRouter:
    global _router
    if _router is None:
        _router = TaskRouter()
    return _router


def route_text(text: str, speak: bool = False) -> dict[str, Any]:
    return get_task_router().route(text, speak=speak).to_dict()
