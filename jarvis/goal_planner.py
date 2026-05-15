"""
goal_planner.py - Persistent hierarchical goals and dynamic task planning.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import threading

from core.paths import user_data_dir
from event_bus import emit

GOALS_FILE = user_data_dir() / "goals.json"


@dataclass
class Subtask:
    title: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    dependencies: List[str] = field(default_factory=list)
    priority: int = 5
    execution_state: str = "pending"
    retry_state: Dict[str, Any] = field(default_factory=lambda: {"attempts": 0, "max": 2})
    verification_state: str = "unverified"
    execution_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Goal:
    title: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    subtasks: List[Subtask] = field(default_factory=list)
    priority: int = 5
    state: str = "active"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    history: List[Dict[str, Any]] = field(default_factory=list)


class GoalPlanner:
    def __init__(self, path: Path = GOALS_FILE):
        self.path = path
        self._lock = threading.RLock()
        self._goals: Dict[str, Goal] = {}
        self._load()

    def _load(self) -> None:
        try:
            if self.path.exists():
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                for item in raw:
                    subtasks = [Subtask(**sub) for sub in item.get("subtasks", [])]
                    item["subtasks"] = subtasks
                    goal = Goal(**item)
                    self._goals[goal.id] = goal
        except Exception:
            self._goals = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = []
        for goal in self._goals.values():
            item = goal.__dict__.copy()
            item["subtasks"] = [sub.__dict__ for sub in goal.subtasks]
            data.append(item)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def create_goal(self, title: str, subtasks: Optional[List[str]] = None, priority: int = 5) -> Goal:
        goal = Goal(title=title, priority=priority, subtasks=[Subtask(title=s) for s in (subtasks or [])])
        with self._lock:
            self._goals[goal.id] = goal
            self._save()
        emit("goal_created", self.goal_to_dict(goal), priority=priority, source="goal_planner")
        return goal

    def next_subtask(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            goals = sorted(
                [goal for goal in self._goals.values() if goal.state == "active"],
                key=lambda g: g.priority,
                reverse=True,
            )
            complete = {
                sub.id for goal in self._goals.values() for sub in goal.subtasks
                if sub.execution_state == "completed"
            }
            for goal in goals:
                candidates = [
                    sub for sub in goal.subtasks
                    if sub.execution_state == "pending" and all(dep in complete for dep in sub.dependencies)
                ]
                if candidates:
                    sub = sorted(candidates, key=lambda s: s.priority, reverse=True)[0]
                    return {"goal": self.goal_to_dict(goal), "subtask": sub.__dict__}
        return None

    def mark_subtask(self, subtask_id: str, state: str, result: Any = None) -> bool:
        with self._lock:
            for goal in self._goals.values():
                for sub in goal.subtasks:
                    if sub.id == subtask_id:
                        sub.execution_state = state
                        sub.execution_history.append({"state": state, "result": result, "timestamp": time.time()})
                        goal.updated_at = time.time()
                        goal.history.append({"subtask": subtask_id, "state": state, "timestamp": time.time()})
                        self._save()
                        emit(f"task_{state}", {"goal_id": goal.id, "subtask_id": subtask_id, "result": result}, source="goal_planner")
                        return True
        return False

    def replan_goal(self, goal_id: str, new_subtasks: List[str]) -> bool:
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return False
            goal.subtasks.extend(Subtask(title=s) for s in new_subtasks)
            goal.updated_at = time.time()
            goal.history.append({"event": "replanned", "subtasks": new_subtasks, "timestamp": time.time()})
            self._save()
        emit("goal_replanned", self.goal_to_dict(goal), source="goal_planner")
        return True

    def list_goals(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [self.goal_to_dict(goal) for goal in self._goals.values()]

    def goal_to_dict(self, goal: Goal) -> Dict[str, Any]:
        item = goal.__dict__.copy()
        item["subtasks"] = [sub.__dict__ for sub in goal.subtasks]
        return item


_planner: Optional[GoalPlanner] = None


def get_goal_planner() -> GoalPlanner:
    global _planner
    if _planner is None:
        _planner = GoalPlanner()
    return _planner
