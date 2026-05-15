"""
safety.py - Guardrails for autonomous desktop and browser actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable

DANGEROUS_KEYWORDS = (
    "delete", "remove", "format", "shutdown", "reboot", "registry", "payment",
    "purchase", "send email", "submit", "transfer", "password", "credential",
)


@dataclass
class SafetyDecision:
    allowed: bool
    requires_confirmation: bool = False
    reason: str = "allowed"
    level: str = "low"


@dataclass
class SafetyManager:
    allowlist: set[str] = field(default_factory=set)
    blocklist: set[str] = field(default_factory=set)
    safe_mode: bool = False
    max_loop_iterations: int = 50

    def assess(self, action: str, context: Dict | None = None, *, safety_level: str = "medium") -> SafetyDecision:
        text = f"{action} {context or {}}".lower()
        if action in self.blocklist:
            return SafetyDecision(False, False, "blocked by policy", "critical")
        if self.safe_mode and safety_level in {"high", "critical"}:
            return SafetyDecision(False, True, "safe mode blocks high-risk action", safety_level)
        if action in self.allowlist:
            return SafetyDecision(True, False, "allowlisted", safety_level)
        if any(keyword in text for keyword in DANGEROUS_KEYWORDS):
            return SafetyDecision(False, True, "confirmation required for dangerous action", "high")
        if safety_level in {"high", "critical"}:
            return SafetyDecision(False, True, "confirmation required", safety_level)
        return SafetyDecision(True, False, "allowed", safety_level)

    def add_allowlist(self, actions: Iterable[str]) -> None:
        self.allowlist.update(actions)

    def add_blocklist(self, actions: Iterable[str]) -> None:
        self.blocklist.update(actions)


_safety = SafetyManager()


def get_safety_manager() -> SafetyManager:
    return _safety
