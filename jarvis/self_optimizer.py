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


class SkillGraph:
    """Lightweight graph of repeated workflows that may become reusable skills."""

    def __init__(self, data: Dict[str, Any]):
        self.data = data

    def suggestions(self) -> list[Dict[str, Any]]:
        suggestions = []
        for workflow_id, stats in self.data.get("workflows", {}).items():
            success = int(stats.get("success", 0))
            failure = int(stats.get("failure", 0))
            total = success + failure
            confidence = success / total if total else 0.0
            if success >= 3 and confidence >= 0.8:
                suggestions.append({
                    "workflow_id": workflow_id,
                    "confidence": round(confidence, 3),
                    "success_count": success,
                    "policy": "suggest_first",
                })
        return sorted(suggestions, key=lambda item: (item["confidence"], item["success_count"]), reverse=True)


class ExecutionPatternMiner:
    """Mines success/failure patterns from recorded tools and workflows."""

    def __init__(self, data: Dict[str, Any]):
        self.data = data

    def mine(self) -> Dict[str, Any]:
        tools = self.data.get("tools", {})
        workflows = self.data.get("workflows", {})
        failed_tools = [
            name for name, stats in tools.items()
            if int(stats.get("failure", 0)) > int(stats.get("success", 0))
        ]
        repeated_workflows = [
            name for name, stats in workflows.items()
            if int(stats.get("success", 0)) + int(stats.get("failure", 0)) >= 3
        ]
        return {"failed_tools": failed_tools, "repeated_workflows": repeated_workflows}


class WorkflowCompressor:
    """Turns repeated successful workflows into proposed reusable skill records."""

    def __init__(self, data: Dict[str, Any]):
        self.data = data

    def proposals(self) -> list[Dict[str, Any]]:
        return [
            {
                "name": f"{workflow_id}_routine",
                "workflow_id": workflow_id,
                "activation": "suggest_first",
            }
            for workflow_id, stats in self.data.get("workflows", {}).items()
            if int(stats.get("success", 0)) >= 3
        ]


class AdaptiveRetryEngine:
    """Derives bounded retry policy hints from historical tool reliability."""

    def __init__(self, data: Dict[str, Any]):
        self.data = data

    def policy_for_tool(self, name: str) -> Dict[str, Any]:
        stats = self.data.get("tools", {}).get(name, {})
        success = int(stats.get("success", 0))
        failure = int(stats.get("failure", 0))
        total = success + failure
        reliability = success / total if total else 0.5
        return {
            "tool": name,
            "max_attempts": 1 if reliability < 0.4 else 3,
            "cooldown_seconds": 3.0 if reliability < 0.6 else 1.0,
            "reliability": round(reliability, 3),
        }


class LatencyLearningSystem:
    """Tracks average tool latency for router and executor scoring."""

    def __init__(self, data: Dict[str, Any]):
        self.data = data

    def tool_latency(self) -> Dict[str, float]:
        result = {}
        for name, stats in self.data.get("tools", {}).items():
            samples = stats.get("latency_ms", []) or []
            if samples:
                result[name] = round(sum(samples) / len(samples), 2)
        return result


class ProviderOptimizationLayer:
    """Placeholder scoring surface for router metrics without changing provider calls."""

    def __init__(self, data: Dict[str, Any]):
        self.data = data

    def scores(self) -> Dict[str, Any]:
        providers = self.data.get("providers", {})
        return {
            name: {
                "success_rate": stats.get("success_rate", 0.5),
                "avg_latency_ms": stats.get("avg_latency_ms"),
            }
            for name, stats in providers.items()
        }


class SelfOptimizer:
    def __init__(self, path: Path = OPT_FILE):
        self.path = path
        self._lock = threading.RLock()
        self._data = {"tools": {}, "workflows": {}, "latency": [], "response_quality": [], "skills": {}}
        self._load()
        subscribe("tool_executed", self._on_tool)
        subscribe("tool_failed", self._on_tool)
        subscribe("task_completed", self._on_task)
        subscribe("task_failed", self._on_task)
        subscribe("provider_call_succeeded", self._on_provider)
        subscribe("provider_call_failed", self._on_provider)

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
            skill_graph = SkillGraph(self._data)
            miner = ExecutionPatternMiner(self._data)
            compressor = WorkflowCompressor(self._data)
            latency = LatencyLearningSystem(self._data)
            providers = ProviderOptimizationLayer(self._data)
            return {
                "optimizer": self._data,
                "recommendations": self.recommend_tools(),
                "skill_suggestions": skill_graph.suggestions(),
                "workflow_proposals": compressor.proposals(),
                "patterns": miner.mine(),
                "latency": latency.tool_latency(),
                "provider_scores": providers.scores(),
            }

    def _on_tool(self, event) -> None:
        self.record_tool(event.payload.get("tool", "unknown"), event.name == "tool_executed", event.payload.get("latency_ms"))

    def _on_task(self, event) -> None:
        workflow_id = event.payload.get("workflow_id") or event.payload.get("goal_id") or "general"
        with self._lock:
            wf = self._data.setdefault("workflows", {}).setdefault(workflow_id, {"success": 0, "failure": 0, "last": None})
            wf["success" if event.name == "task_completed" else "failure"] += 1
            wf["last"] = time.time()
            self._save()

    def _on_provider(self, event) -> None:
        provider = event.payload.get("provider", "unknown")
        with self._lock:
            stats = self._data.setdefault("providers", {}).setdefault(
                provider,
                {"success": 0, "failure": 0, "latency_ms": []},
            )
            if event.name == "provider_call_succeeded":
                stats["success"] += 1
                if event.payload.get("latency_ms") is not None:
                    stats["latency_ms"].append(event.payload["latency_ms"])
                    stats["latency_ms"] = stats["latency_ms"][-50:]
            else:
                stats["failure"] += 1
                stats["last_error"] = event.payload.get("error", "")
            total = stats["success"] + stats["failure"]
            stats["success_rate"] = stats["success"] / total if total else 0.5
            samples = stats.get("latency_ms", [])
            stats["avg_latency_ms"] = sum(samples) / len(samples) if samples else None
            self._save()


_optimizer: Optional[SelfOptimizer] = None


def get_self_optimizer() -> SelfOptimizer:
    global _optimizer
    if _optimizer is None:
        _optimizer = SelfOptimizer()
    return _optimizer
