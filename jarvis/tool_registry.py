"""
tool_registry.py - Intelligent tool metadata and selection.
"""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from event_bus import emit
from safety import get_safety_manager


@dataclass
class ToolMetadata:
    name: str
    description: str = ""
    permissions: List[str] = field(default_factory=list)
    safety_level: str = "medium"
    retry_policy: Dict[str, Any] = field(default_factory=lambda: {"retries": 1, "cooldown": 0.5})
    dependencies: List[str] = field(default_factory=list)
    latency_estimate_ms: int = 500
    capabilities: List[str] = field(default_factory=list)
    required_context: List[str] = field(default_factory=list)


@dataclass
class RegisteredTool:
    metadata: ToolMetadata
    func: Callable[..., Any]
    successes: int = 0
    failures: int = 0
    last_latency_ms: Optional[int] = None

    def score(self, query: str = "", context: Optional[Dict[str, Any]] = None) -> float:
        haystack = " ".join([self.metadata.name, self.metadata.description, *self.metadata.capabilities]).lower()
        terms = [t for t in query.lower().split() if len(t) > 2]
        relevance = sum(1 for term in terms if term in haystack) / max(1, len(terms))
        reliability = (self.successes + 1) / (self.successes + self.failures + 2)
        latency = max(0.1, 1.0 - min(self.metadata.latency_estimate_ms, 5000) / 5000)
        return relevance * 0.5 + reliability * 0.35 + latency * 0.15


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, RegisteredTool] = {}

    def register(self, func: Callable[..., Any], metadata: ToolMetadata) -> RegisteredTool:
        tool = RegisteredTool(metadata=metadata, func=func)
        self._tools[metadata.name] = tool
        emit("tool_registered", {"name": metadata.name, "metadata": metadata.__dict__}, source="tool_registry")
        return tool

    def load_from_actions(self) -> None:
        actions = importlib.import_module("actions")
        for name, entry in getattr(actions, "ACTION_REGISTRY", {}).items():
            if name in self._tools:
                continue
            self.register(
                entry["func"],
                ToolMetadata(
                    name=name,
                    description=entry.get("description", ""),
                    permissions=["desktop"] if name in {"open_app", "type_text"} else [],
                    safety_level="high" if name in {"run_script", "write_clipboard", "type_text"} else "medium",
                    capabilities=[name.replace("_", " "), "automation"],
                ),
            )

    def execute(self, name: str, args: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None) -> Any:
        args = args or {}
        tool = self._tools.get(name)
        if not tool:
            raise KeyError(f"Unknown tool: {name}")
        decision = get_safety_manager().assess(name, args, safety_level=tool.metadata.safety_level)
        if not decision.allowed:
            emit("tool_confirmation_required", {"tool": name, "reason": decision.reason}, priority=2, source="tool_registry")
            return {"ok": False, "requires_confirmation": decision.requires_confirmation, "reason": decision.reason}
        retries = int(tool.metadata.retry_policy.get("retries", 0))
        cooldown = float(tool.metadata.retry_policy.get("cooldown", 0.0))
        last_error = None
        for attempt in range(retries + 1):
            started = time.time()
            try:
                result = tool.func(**args)
                tool.successes += 1
                tool.last_latency_ms = int((time.time() - started) * 1000)
                emit("tool_executed", {"tool": name, "ok": True, "latency_ms": tool.last_latency_ms}, source="tool_registry")
                return result
            except Exception as exc:
                last_error = exc
                tool.failures += 1
                if attempt < retries:
                    time.sleep(cooldown)
        emit("tool_failed", {"tool": name, "error": str(last_error)}, priority=2, source="tool_registry")
        raise last_error

    def select(self, query: str, context: Optional[Dict[str, Any]] = None, limit: int = 5) -> List[Dict[str, Any]]:
        ranked = sorted(self._tools.values(), key=lambda tool: tool.score(query, context), reverse=True)
        return [
            {"name": tool.metadata.name, "score": tool.score(query, context), "metadata": tool.metadata.__dict__}
            for tool in ranked[:limit]
        ]

    def list_tools(self) -> List[Dict[str, Any]]:
        return [{"name": name, "metadata": tool.metadata.__dict__} for name, tool in sorted(self._tools.items())]


_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
        _registry.load_from_actions()
    return _registry
