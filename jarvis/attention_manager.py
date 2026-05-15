"""
attention_manager.py - Cognitive priority and interruption control.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from event_bus import emit, subscribe
from world_state import get_world_state


@dataclass
class AttentionItem:
    key: str
    label: str
    urgency: float = 0.0
    importance: float = 0.0
    context_relevance: float = 0.0
    interruptive: bool = False
    source: str = "system"
    created_at: float = field(default_factory=time.time)

    def score(self, now: Optional[float] = None) -> float:
        now = now or time.time()
        age_minutes = max(0.0, (now - self.created_at) / 60.0)
        decay = max(0.2, 1.0 - (age_minutes * 0.03))
        return ((self.urgency * 0.4) + (self.importance * 0.4) + (self.context_relevance * 0.2)) * decay


class AttentionManager:
    def __init__(self):
        self._items: Dict[str, AttentionItem] = {}
        self._focus: Optional[AttentionItem] = None
        subscribe("goal_created", self._on_goal)
        subscribe("task_failed", self._on_task_failed)
        subscribe("world_state_updated", self._on_world_state)
        subscribe("autonomous_mode_enabled", self._on_autonomy)

    def add_item(self, item: AttentionItem) -> AttentionItem:
        self._items[item.key] = item
        emit("attention_item_added", {"item": item.__dict__}, source="attention")
        self.evaluate_attention()
        return item

    def evaluate_attention(self) -> Dict[str, Any]:
        world = get_world_state().get_state()
        goals = world.get("active_goals", [])
        for goal in goals:
            key = f"goal:{goal.get('id', goal.get('title'))}"
            self._items.setdefault(
                key,
                AttentionItem(
                    key=key,
                    label=goal.get("title", "Active goal"),
                    urgency=float(goal.get("urgency", 0.4)),
                    importance=float(goal.get("priority", 5)) / 10.0,
                    context_relevance=0.7,
                    source="goal",
                ),
            )
        ranked = sorted(self._items.values(), key=lambda item: item.score(), reverse=True)
        previous = self._focus.key if self._focus else None
        self._focus = ranked[0] if ranked else None
        payload = {
            "focus": self._focus.__dict__ if self._focus else None,
            "score": self._focus.score() if self._focus else 0.0,
            "queue": [item.__dict__ | {"score": item.score()} for item in ranked[:10]],
        }
        if self._focus and self._focus.key != previous:
            emit("attention_shifted", payload, priority=3, source="attention")
        return payload

    def should_interrupt(self, item: AttentionItem) -> bool:
        current_score = self._focus.score() if self._focus else 0.0
        return item.interruptive and item.score() >= max(0.75, current_score + 0.2)

    def context_switch(self, key: str) -> bool:
        item = self._items.get(key)
        if not item:
            return False
        self._focus = item
        emit("attention_shifted", {"focus": item.__dict__, "manual": True}, priority=2, source="attention")
        return True

    def notification_allowed(self, urgency: float, importance: float) -> bool:
        world = get_world_state().get_state()
        if world.get("operating_mode") == "FOCUS":
            return urgency >= 0.8 or importance >= 0.8
        return (urgency + importance) / 2 >= 0.35

    def status(self) -> Dict[str, Any]:
        return self.evaluate_attention()

    def _on_goal(self, event) -> None:
        goal = event.payload
        self.add_item(AttentionItem(
            key=f"goal:{goal.get('id', goal.get('title'))}",
            label=goal.get("title", "Active goal"),
            urgency=float(goal.get("urgency", 0.4)),
            importance=float(goal.get("priority", 5)) / 10.0,
            context_relevance=0.8,
            source="goal",
        ))

    def _on_task_failed(self, event) -> None:
        self.add_item(AttentionItem(
            key=f"failure:{int(time.time())}",
            label=event.payload.get("label", "Task failed"),
            urgency=0.8,
            importance=0.7,
            context_relevance=0.6,
            interruptive=True,
            source="execution",
        ))

    def _on_world_state(self, event) -> None:
        updates = event.payload.get("updates", {})
        if "focused_application" in updates:
            self.add_item(AttentionItem(
                key="focus:application",
                label=f"Focused: {updates['focused_application']}",
                urgency=0.2,
                importance=0.3,
                context_relevance=0.6,
                source="world_state",
            ))

    def _on_autonomy(self, event) -> None:
        self.add_item(AttentionItem(
            key="mode:autonomous",
            label="Autonomous cognition online",
            urgency=0.5,
            importance=0.8,
            context_relevance=0.9,
            source="mode",
        ))


_manager: Optional[AttentionManager] = None


def get_attention_manager() -> AttentionManager:
    global _manager
    if _manager is None:
        _manager = AttentionManager()
    return _manager
