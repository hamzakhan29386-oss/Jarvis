"""
world_state.py - Persistent JARVIS environment awareness.
"""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional
import threading

from core.paths import user_data_dir
from event_bus import emit, subscribe

STATE_FILE = user_data_dir() / "world_state.json"


def _default_state() -> Dict[str, Any]:
    return {
        "active_goals": [],
        "current_task": None,
        "open_applications": [],
        "focused_application": None,
        "browser_tabs": [],
        "clipboard_state": {"text_preview": "", "updated_at": None},
        "filesystem_context": {},
        "active_coding_project": None,
        "current_workspace": os.getcwd(),
        "system_resources": {},
        "current_media": None,
        "ongoing_workflows": [],
        "active_timers": [],
        "user_attention_state": "unknown",
        "recent_actions": [],
        "terminal_state": {},
        "browser_context": {},
        "desktop_session_context": {},
        "operating_mode": "ASSIST",
        "autonomous_mode": False,
        "updated_at": time.time(),
    }


class WorldStateEngine:
    def __init__(self, path: Path = STATE_FILE):
        self.path = path
        self._lock = threading.RLock()
        self._state = _default_state()
        self._load()
        subscribe("task_*", self._on_task_event)
        subscribe("goal_*", self._on_goal_event)
        subscribe("memory_updated", self._on_memory_event)

    def _load(self) -> None:
        try:
            if self.path.exists():
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self._state.update(loaded)
        except Exception:
            pass

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def update_state(self, key: str | Dict[str, Any], value: Any = None, *, source: str = "world_state") -> Dict[str, Any]:
        with self._lock:
            updates = key if isinstance(key, dict) else {key: value}
            self._state.update(updates)
            self._state["updated_at"] = time.time()
            self._save()
            snapshot = copy.deepcopy(self._state)
        emit("world_state_updated", {"updates": updates, "state": snapshot}, source=source)
        return snapshot

    def get_state(self, key: Optional[str] = None, default: Any = None) -> Any:
        with self._lock:
            if key is None:
                return copy.deepcopy(self._state)
            return copy.deepcopy(self._state.get(key, default))

    def push_action(self, action: Dict[str, Any] | str, *, outcome: str = "unknown", source: str = "action_engine") -> Dict[str, Any]:
        item = action if isinstance(action, dict) else {"action": action}
        item = {**item, "outcome": outcome, "timestamp": time.time()}
        with self._lock:
            self._state.setdefault("recent_actions", []).append(item)
            self._state["recent_actions"] = self._state["recent_actions"][-100:]
            self._state["updated_at"] = time.time()
            self._save()
        emit("action_recorded", item, source=source)
        return item

    def set_goal(self, goal: Dict[str, Any] | str, *, priority: int = 5) -> Dict[str, Any]:
        item = goal if isinstance(goal, dict) else {"title": goal}
        item.setdefault("id", f"goal-{int(time.time() * 1000)}")
        item.setdefault("status", "active")
        item.setdefault("priority", priority)
        item.setdefault("created_at", time.time())
        with self._lock:
            goals = [g for g in self._state.setdefault("active_goals", []) if g.get("id") != item["id"]]
            goals.append(item)
            self._state["active_goals"] = goals
            self._save()
        emit("goal_created", item, priority=priority, source="world_state")
        return item

    def clear_goal(self, goal_id: str) -> bool:
        with self._lock:
            before = len(self._state.get("active_goals", []))
            self._state["active_goals"] = [
                goal for goal in self._state.get("active_goals", [])
                if goal.get("id") != goal_id and goal.get("title") != goal_id
            ]
            changed = len(self._state["active_goals"]) != before
            if changed:
                self._save()
        if changed:
            emit("goal_cleared", {"goal_id": goal_id}, source="world_state")
        return changed

    def export_state(self) -> Dict[str, Any]:
        return self.get_state()

    def snapshot(self, label: str = "snapshot") -> Dict[str, Any]:
        snap = {"label": label, "timestamp": time.time(), "state": self.get_state()}
        emit("world_state_snapshot", snap, source="world_state")
        return snap

    def refresh_environment(self) -> Dict[str, Any]:
        updates: Dict[str, Any] = {"current_workspace": os.getcwd()}
        try:
            import psutil
            updates["system_resources"] = {
                "cpu_percent": psutil.cpu_percent(interval=0.0),
                "ram_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage(str(Path.cwd().anchor or "/")).percent,
            }
            updates["open_applications"] = sorted({
                p.info.get("name") for p in psutil.process_iter(["name"]) if p.info.get("name")
            })[:80]
        except Exception:
            pass
        try:
            import pyperclip
            text = pyperclip.paste() or ""
            updates["clipboard_state"] = {"text_preview": text[:200], "updated_at": time.time()}
        except Exception:
            pass
        try:
            import pygetwindow as gw
            win = gw.getActiveWindow()
            if win:
                updates["focused_application"] = getattr(win, "title", None)
        except Exception:
            pass
        return self.update_state(updates, source="environment_probe")

    def _on_task_event(self, event) -> None:
        if event.name in {"task_started", "task_created"}:
            self.update_state("current_task", event.payload, source="event_bus")
        elif event.name in {"task_completed", "task_failed"}:
            self.push_action(event.payload, outcome=event.name.replace("task_", ""), source="event_bus")

    def _on_goal_event(self, event) -> None:
        if event.name == "goal_completed":
            goal_id = event.payload.get("id") or event.payload.get("goal_id")
            if goal_id:
                self.clear_goal(goal_id)

    def _on_memory_event(self, event) -> None:
        self.update_state("last_memory_update", event.payload, source="memory")


_world: Optional[WorldStateEngine] = None
_lock = threading.Lock()


def get_world_state() -> WorldStateEngine:
    global _world
    with _lock:
        if _world is None:
            _world = WorldStateEngine()
        return _world


def update_state(*args, **kwargs):
    return get_world_state().update_state(*args, **kwargs)


def get_state(*args, **kwargs):
    return get_world_state().get_state(*args, **kwargs)


def push_action(*args, **kwargs):
    return get_world_state().push_action(*args, **kwargs)


def set_goal(*args, **kwargs):
    return get_world_state().set_goal(*args, **kwargs)


def clear_goal(*args, **kwargs):
    return get_world_state().clear_goal(*args, **kwargs)


def export_state():
    return get_world_state().export_state()
