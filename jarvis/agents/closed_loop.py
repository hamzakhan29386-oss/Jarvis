"""Planner -> executor -> verifier -> recovery loop for JARVIS actions."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from event_bus import emit


@dataclass
class ActionExecution:
    execution_id: str
    action: str
    args: dict[str, Any]
    attempts: int = 0
    result: Any = None
    ok: bool = False
    verification: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExecutionPlanner:
    def plan_action(self, action: str, args: dict[str, Any] | None = None) -> ActionExecution:
        return ActionExecution(
            execution_id=f"exec-{uuid.uuid4().hex[:10]}",
            action=action,
            args=args or {},
        )


class ActionVerifier:
    FAILURE_TERMS = ("failed", "error", "could not", "unknown action", "timed out", "not found")

    def verify(self, execution: ActionExecution) -> dict[str, Any]:
        text = execution.result if isinstance(execution.result, str) else json.dumps(execution.result, default=str)
        lowered = text.lower()
        ok = not any(term in lowered for term in self.FAILURE_TERMS)
        return {
            "ok": ok,
            "checked_at": time.time(),
            "evidence": text[:1000],
            "failure_terms": [term for term in self.FAILURE_TERMS if term in lowered],
        }


class RecoveryPolicy:
    def __init__(self, max_attempts: int = 2):
        self.max_attempts = max_attempts

    def should_retry(self, execution: ActionExecution) -> bool:
        if execution.attempts >= self.max_attempts:
            return False
        if "unknown action" in execution.verification.get("failure_terms", []):
            return False
        return not execution.verification.get("ok", False)


class ClosedLoopAgentExecutor:
    def __init__(self):
        self.planner = ExecutionPlanner()
        self.verifier = ActionVerifier()
        self.recovery = RecoveryPolicy()

    def run_action(self, action: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        execution = self.planner.plan_action(action, args)
        emit("agent_execution_planned", execution.to_dict(), source="closed_loop_executor")

        while True:
            execution.attempts += 1
            emit("agent_execution_started", execution.to_dict(), source="closed_loop_executor")
            try:
                from actions import execute_action

                execution.result = execute_action(execution.action, execution.args)
            except Exception as exc:
                execution.result = f"Action '{execution.action}' failed: {exc}"

            execution.verification = self.verifier.verify(execution)
            execution.ok = bool(execution.verification.get("ok"))
            if execution.ok or not self.recovery.should_retry(execution):
                break
            emit("agent_execution_retrying", execution.to_dict(), priority=2, source="closed_loop_executor")
            time.sleep(0.25)

        execution.finished_at = time.time()
        payload = execution.to_dict()
        self._record(payload)
        emit(
            "agent_execution_completed" if execution.ok else "agent_execution_failed",
            payload,
            priority=2 if not execution.ok else 5,
            source="closed_loop_executor",
        )
        return payload

    def _record(self, payload: dict[str, Any]) -> None:
        try:
            from execution_replay import get_execution_replay

            get_execution_replay().record("closed_loop_action", payload, outcome="ok" if payload.get("ok") else "failed")
        except Exception:
            pass
        try:
            from world_state import get_world_state

            get_world_state().push_action(
                {"action": payload.get("action"), "args": payload.get("args"), "verification": payload.get("verification")},
                outcome="verified" if payload.get("ok") else "failed",
                source="closed_loop_executor",
            )
        except Exception:
            pass


_executor: ClosedLoopAgentExecutor | None = None


def get_closed_loop_executor() -> ClosedLoopAgentExecutor:
    global _executor
    if _executor is None:
        _executor = ClosedLoopAgentExecutor()
    return _executor
