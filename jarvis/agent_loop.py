"""
agent_loop.py - Persistent autonomous cognition loop.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from attention_manager import get_attention_manager
from event_bus import emit
from execution_replay import get_execution_replay
from goal_planner import get_goal_planner
from tool_registry import get_tool_registry
from world_state import get_world_state


@dataclass
class AgentJob:
    kind: str
    payload: Dict[str, Any]
    priority: int = 5
    created_at: float = field(default_factory=time.time)


class AutonomousAgentLoop:
    def __init__(self, interval_s: float = 5.0):
        self.interval_s = interval_s
        self._queue: "queue.PriorityQueue[tuple[int, float, AgentJob]]" = queue.PriorityQueue()
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._status = {"running": False, "paused": False, "last_tick": None, "last_result": None}

    def start(self) -> Dict[str, Any]:
        if self._thread and self._thread.is_alive():
            return self.status()
        self._stop.clear()
        self._pause.clear()
        self._thread = threading.Thread(target=self._run, name="jarvis-agent-loop", daemon=True)
        self._thread.start()
        self._status["running"] = True
        get_world_state().update_state({"autonomous_mode": True}, source="agent_loop")
        emit("autonomous_mode_enabled", self.status(), priority=2, source="agent_loop")
        return self.status()

    def pause(self) -> Dict[str, Any]:
        self._pause.set()
        self._status["paused"] = True
        emit("autonomous_loop_paused", self.status(), source="agent_loop")
        return self.status()

    def resume(self) -> Dict[str, Any]:
        self._pause.clear()
        self._status["paused"] = False
        emit("autonomous_loop_resumed", self.status(), source="agent_loop")
        return self.status()

    def stop(self) -> Dict[str, Any]:
        self._stop.set()
        get_world_state().update_state({"autonomous_mode": False}, source="agent_loop")
        emit("autonomous_mode_disabled", self.status(), source="agent_loop")
        return self.status()

    def enqueue(self, job: AgentJob) -> None:
        self._queue.put((job.priority, job.created_at, job))
        emit("task_created", job.__dict__, source="agent_loop")

    def status(self) -> Dict[str, Any]:
        self._status["queue_size"] = self._queue.qsize()
        return dict(self._status)

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._pause.is_set():
                time.sleep(0.5)
                continue
            try:
                result = self.tick()
                self._status.update({"running": True, "last_tick": time.time(), "last_result": result})
            except Exception as exc:
                emit("autonomous_loop_error", {"error": str(exc)}, priority=2, source="agent_loop")
            time.sleep(self.interval_s)
        self._status["running"] = False

    def tick(self) -> Dict[str, Any]:
        world = self.observe()
        attention = self.evaluate_attention()
        goal_task = self.evaluate_goals()
        job = self._dequeue_job()
        plan = self.generate_plan(job, goal_task, attention)
        result = self.execute_action(plan)
        verified = self.verify_result(result)
        reflection = self.reflect(result, verified)
        if not verified.get("ok") and plan:
            self.retry_if_needed(plan, result)
        return {"world": world, "attention": attention, "plan": plan, "result": result, "verified": verified, "reflection": reflection}

    def observe(self) -> Dict[str, Any]:
        return get_world_state().refresh_environment()

    def evaluate_attention(self) -> Dict[str, Any]:
        return get_attention_manager().evaluate_attention()

    def evaluate_goals(self) -> Optional[Dict[str, Any]]:
        return get_goal_planner().next_subtask()

    def generate_plan(self, job: Optional[AgentJob], goal_task: Optional[Dict[str, Any]], attention: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if job:
            return {"kind": job.kind, "payload": job.payload, "source": "queue"}
        if goal_task:
            return {"kind": "goal_subtask", "payload": goal_task, "source": "goal_planner"}
        return None

    def execute_action(self, plan: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not plan:
            return {"ok": True, "idle": True}
        emit("task_started", plan, source="agent_loop")
        payload = plan.get("payload", {})
        if plan["kind"] == "tool":
            result = get_tool_registry().execute(payload["name"], payload.get("args", {}))
        elif plan["kind"] == "goal_subtask":
            subtask = payload["subtask"]
            result = {"ok": True, "message": f"Ready for subtask: {subtask['title']}", "subtask_id": subtask["id"]}
        else:
            result = {"ok": True, "payload": payload}
        get_execution_replay().record("agent_plan", {"plan": plan, "result": result}, outcome="executed")
        emit("task_completed" if self._result_ok(result) else "task_failed", {"plan": plan, "result": result}, source="agent_loop")
        return result if isinstance(result, dict) else {"ok": True, "result": result}

    def verify_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {"ok": self._result_ok(result), "checked_at": time.time()}

    def reflect(self, result: Dict[str, Any], verified: Dict[str, Any]) -> Dict[str, Any]:
        reflection = {"summary": "nominal" if verified.get("ok") else "needs retry", "timestamp": time.time()}
        emit("reflection_completed", reflection, source="agent_loop")
        return reflection

    def retry_if_needed(self, plan: Dict[str, Any], result: Dict[str, Any]) -> None:
        if plan.get("retry_count", 0) >= 2:
            return
        plan["retry_count"] = plan.get("retry_count", 0) + 1
        self.enqueue(AgentJob(kind=plan["kind"], payload=plan["payload"], priority=3))

    def _dequeue_job(self) -> Optional[AgentJob]:
        try:
            _, _, job = self._queue.get_nowait()
            return job
        except queue.Empty:
            return None

    def _result_ok(self, result: Any) -> bool:
        if isinstance(result, dict):
            return result.get("ok", True) is not False and not result.get("requires_confirmation")
        return True


_loop: Optional[AutonomousAgentLoop] = None


def get_agent_loop() -> AutonomousAgentLoop:
    global _loop
    if _loop is None:
        _loop = AutonomousAgentLoop()
    return _loop
