"""
operating_modes.py - JARVIS autonomy and personality modes.
"""

from __future__ import annotations

from typing import Dict

from event_bus import emit
from world_state import get_world_state

MODES: Dict[str, Dict] = {
    "PASSIVE": {"autonomy": 0.0, "verbosity": "minimal", "planning_depth": 1, "interruptions": "critical_only"},
    "ASSIST": {"autonomy": 0.35, "verbosity": "concise", "planning_depth": 2, "interruptions": "important"},
    "AUTONOMOUS": {"autonomy": 0.8, "verbosity": "brief_status", "planning_depth": 4, "interruptions": "filtered"},
    "FOCUS": {"autonomy": 0.25, "verbosity": "minimal", "planning_depth": 2, "interruptions": "critical_only"},
    "RESEARCH": {"autonomy": 0.55, "verbosity": "analytical", "planning_depth": 4, "interruptions": "relevant"},
    "CODER": {"autonomy": 0.6, "verbosity": "technical", "planning_depth": 4, "interruptions": "errors_and_completion"},
}


def set_operating_mode(mode: str) -> Dict:
    mode = mode.upper()
    if mode not in MODES:
        raise ValueError(f"Unknown operating mode: {mode}")
    config = {"mode": mode, **MODES[mode]}
    get_world_state().update_state({"operating_mode": mode, "mode_config": config}, source="operating_modes")
    emit("operating_mode_changed", config, source="operating_modes")
    return config


def get_operating_mode() -> Dict:
    world = get_world_state().get_state()
    mode = world.get("operating_mode", "ASSIST")
    return {"mode": mode, **MODES.get(mode, MODES["ASSIST"])}
