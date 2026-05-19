"""Long-horizon task runtime with checkpoints and confirmation boundaries."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from event_bus import emit


class GoalDecomposer:
    def decompose(self, objective: str) -> list[str]:
        return [
            "Clarify objective and constraints",
            "Inspect current state",
            "Plan next bounded action",
            "Execute with confirmation boundaries",
            "Verify result and checkpoint",
        ] if objective else []


class ConstraintReasoner:
    def normalize(self, constraints: list[str] | None) -> list[str]:
        base = ["confirm before visible desktop, network, filesystem, or terminal changes"]
        return base + [item for item in (constraints or []) if item]


class RecursivePlanner:
    def plan(self, objective: str, constraints: list[str]) -> list[str]:
        steps = GoalDecomposer().decompose(objective)
        if constraints:
            steps.insert(1, "Check safety and confirmation constraints")
        return steps


class ReflectionEngine:
    def reflect(self, label: str, note: str = "") -> dict[str, Any]:
        return {"label": label, "note": note, "timestamp": time.time()}


@dataclass
class LongTask:
    id: str
    objective: str
    status: str = "planned"
    constraints: list[str] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    requires_confirmation: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LongTaskRuntime:
    def __init__(self):
        self._lock = threading.RLock()
        self._tasks: dict[str, LongTask] = {}
        self.constraints = ConstraintReasoner()
        self.planner = RecursivePlanner()
        self.reflection = ReflectionEngine()

    def create_task(
        self,
        objective: str,
        *,
        constraints: list[str] | None = None,
        plan: list[str] | None = None,
        requires_confirmation: bool = True,
    ) -> dict[str, Any]:
        normalized_constraints = self.constraints.normalize(constraints)
        task = LongTask(
            id=f"task-{uuid.uuid4().hex[:10]}",
            objective=objective,
            constraints=normalized_constraints,
            plan=plan or self.planner.plan(objective, normalized_constraints),
            requires_confirmation=requires_confirmation,
        )
        task.checkpoints.append(self._checkpoint("created", "Task created."))
        with self._lock:
            self._tasks[task.id] = task
        emit("long_task_created", task.to_dict(), source="long_task_runtime")
        return task.to_dict()

    def list_tasks(self) -> list[dict[str, Any]]:
        with self._lock:
            return [task.to_dict() for task in self._tasks.values()]

    def update_status(self, task_id: str, status: str) -> dict[str, Any]:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return {"ok": False, "error": "task_not_found"}
            task.status = status
            task.updated_at = time.time()
            task.checkpoints.append(self._checkpoint(status, f"Task marked {status}."))
            payload = task.to_dict()
        emit("long_task_status_changed", payload, source="long_task_runtime")
        return {"ok": True, "task": payload}

    def checkpoint(self, task_id: str, label: str, note: str = "") -> dict[str, Any]:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return {"ok": False, "error": "task_not_found"}
            item = self._checkpoint(label, note)
            task.checkpoints.append(item)
            task.updated_at = time.time()
            payload = task.to_dict()
        emit("long_task_checkpointed", {"task_id": task_id, "checkpoint": item}, source="long_task_runtime")
        return {"ok": True, "task": payload}

    def _checkpoint(self, label: str, note: str) -> dict[str, Any]:
        return self.reflection.reflect(label, note)


_runtime: LongTaskRuntime | None = None
_lock = threading.Lock()


def get_long_task_runtime() -> LongTaskRuntime:
    global _runtime
    with _lock:
        if _runtime is None:
            _runtime = LongTaskRuntime()
        return _runtime
