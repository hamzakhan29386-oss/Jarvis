"""
execution_replay.py - Persistent action, goal, and outcome replay log.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import threading

from core.paths import user_data_dir
from event_bus import emit, subscribe

REPLAY_FILE = user_data_dir() / "execution_replay.json"


class ExecutionReplay:
    def __init__(self, path: Path = REPLAY_FILE):
        self.path = path
        self._lock = threading.RLock()
        self._records: List[Dict[str, Any]] = []
        self._load()
        subscribe("task_*", self._record_event)
        subscribe("action_recorded", self._record_event)
        subscribe("goal_*", self._record_event)

    def _load(self) -> None:
        try:
            if self.path.exists():
                self._records = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            self._records = []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._records[-2000:], indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def record(self, kind: str, payload: Dict[str, Any], *, outcome: str = "observed") -> Dict[str, Any]:
        item = {
            "id": uuid.uuid4().hex,
            "kind": kind,
            "payload": payload,
            "outcome": outcome,
            "timestamp": time.time(),
        }
        with self._lock:
            self._records.append(item)
            self._save()
        emit("execution_replay_recorded", item, source="execution_replay")
        return item

    def recent(self, limit: int = 50, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            records = [r for r in self._records if kind is None or r.get("kind") == kind]
            return records[-limit:]

    def replay_workflow(self, workflow_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            return [r for r in self._records if r.get("payload", {}).get("workflow_id") == workflow_id]

    def _record_event(self, event) -> None:
        self.record(event.name, event.payload, outcome="event")


_replay: Optional[ExecutionReplay] = None


def get_execution_replay() -> ExecutionReplay:
    global _replay
    if _replay is None:
        _replay = ExecutionReplay()
    return _replay
