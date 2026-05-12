"""Reusable workflow automation actions."""

from __future__ import annotations

from actions import AgentPlan, PlanStep, action, execute_plan


WORKFLOWS = {
    "coding_setup": [
        PlanStep("open_app", {"name": "vscode"}),
        PlanStep("open_app", {"name": "terminal"}, delay_ms=700),
        PlanStep("open_url", {"url": "http://localhost:5000"}, delay_ms=700),
    ],
    "study_mode": [
        PlanStep("open_app", {"name": "notion"}),
        PlanStep("browser_google_search", {"query": "study timer pomodoro"}, delay_ms=700),
        PlanStep("control_media", {"action_name": "mute"}, delay_ms=300),
    ],
    "entertainment_mode": [
        PlanStep("open_app", {"name": "spotify"}),
        PlanStep("open_app", {"name": "discord"}, delay_ms=700),
    ],
}


@action("run_workflow", "Run a named desktop workflow")
def run_workflow(name: str) -> str:
    key = name.strip().lower().replace(" ", "_")
    steps = WORKFLOWS.get(key)
    if not steps:
        return f"Unknown workflow: {name}"
    plan = AgentPlan(plan_id=key, label=key.replace("_", " ").title(), steps=steps)
    summary = execute_plan(plan)
    return summary["report"]

