"""
self_optimizer.py - Learns which workflows and tools perform well over time.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional
import threading

from core.paths import user_data_dir
from event_bus import emit, subscribe

OPT_FILE = user_data_dir() / "self_optimizer.json"


class SelfOptimizer:
    def __init__(self, path: Path = OPT_FILE):
        self.path = path
        self._lock = threading.RLock()
        self._data = {"tools": {}, "workflows": {}, "latency": [], "response_quality": []}
        self._load()
        subscribe("tool_executed", self._on_tool)
        subscribe("tool_failed", self._on_tool)
        subscribe("task_completed", self._on_task)
        subscribe("task_failed", self._on_task)

    def _load(self) -> None:
        try:
            if self.path.exists():
                self._data.update(json.loads(self.path.read_text(encoding="utf-8")))
        except Exception:
            pass

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def record_tool(self, name: str, ok: bool, latency_ms: Optional[int] = None) -> None:
        with self._lock:
            stats = self._data.setdefault("tools", {}).setdefault(name, {"success": 0, "failure": 0, "latency_ms": []})
            stats["success" if ok else "failure"] += 1
            if latency_ms is not None:
                stats["latency_ms"].append(latency_ms)
                stats["latency_ms"] = stats["latency_ms"][-50:]
            self._save()

    def recommend_tools(self) -> Dict[str, Any]:
        with self._lock:
            ranked = []
            for name, stats in self._data.get("tools", {}).items():
                total = stats.get("success", 0) + stats.get("failure", 0)
                reliability = stats.get("success", 0) / total if total else 0.5
                avg_latency = sum(stats.get("latency_ms", []) or [500]) / len(stats.get("latency_ms", []) or [500])
                ranked.append({"name": name, "reliability": reliability, "avg_latency_ms": avg_latency})
            return {"tools": sorted(ranked, key=lambda x: (x["reliability"], -x["avg_latency_ms"]), reverse=True)}

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {"optimizer": self._data, "recommendations": self.recommend_tools()}

    def _on_tool(self, event) -> None:
        self.record_tool(event.payload.get("tool", "unknown"), event.name == "tool_executed", event.payload.get("latency_ms"))

    def _on_task(self, event) -> None:
        workflow_id = event.payload.get("workflow_id") or event.payload.get("goal_id") or "general"
        with self._lock:
            wf = self._data.setdefault("workflows", {}).setdefault(workflow_id, {"success": 0, "failure": 0, "last": None})
            wf["success" if event.name == "task_completed" else "failure"] += 1
            wf["last"] = time.time()
            self._save()


_optimizer: Optional[SelfOptimizer] = None


def get_self_optimizer() -> SelfOptimizer:
    global _optimizer
    if _optimizer is None:
        _optimizer = SelfOptimizer()
    return _optimizer
