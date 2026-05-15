"""
skills.skill_manager - Hot-loadable JARVIS skill/plugin architecture.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from event_bus import emit

SKILLS_DIR = Path(__file__).resolve().parent


class SkillManager:
    def __init__(self, root: Path = SKILLS_DIR):
        self.root = root
        self._skills: Dict[str, Dict[str, Any]] = {}

    def discover(self) -> List[Dict[str, Any]]:
        self._skills = {}
        for manifest in self.root.glob("*/manifest.json"):
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                data["_path"] = str(manifest.parent)
                data.setdefault("enabled", True)
                self._skills[data["name"]] = data
            except Exception as exc:
                emit("skill_load_failed", {"path": str(manifest), "error": str(exc)}, source="skills")
        emit("skills_discovered", {"count": len(self._skills)}, source="skills")
        return list(self._skills.values())

    def enable(self, name: str) -> bool:
        return self._set_enabled(name, True)

    def disable(self, name: str) -> bool:
        return self._set_enabled(name, False)

    def load(self, name: str) -> Dict[str, Any]:
        manifest = self._skills.get(name) or next((s for s in self.discover() if s.get("name") == name), None)
        if not manifest:
            return {"ok": False, "error": "skill not found"}
        if not manifest.get("enabled", True):
            return {"ok": False, "error": "skill disabled"}
        entry = manifest.get("entrypoint", "skill.py")
        path = Path(manifest["_path"]) / entry
        if not path.exists():
            return {"ok": True, "manifest": manifest, "loaded": False}
        spec = importlib.util.spec_from_file_location(f"jarvis_skill_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        if hasattr(module, "register"):
            module.register()
        emit("skill_loaded", {"name": name}, source="skills")
        return {"ok": True, "manifest": manifest, "loaded": True}

    def list_skills(self) -> List[Dict[str, Any]]:
        return self.discover()

    def _set_enabled(self, name: str, enabled: bool) -> bool:
        manifest = self._skills.get(name) or next((s for s in self.discover() if s.get("name") == name), None)
        if not manifest:
            return False
        manifest["enabled"] = enabled
        path = Path(manifest["_path"]) / "manifest.json"
        clean = {k: v for k, v in manifest.items() if not k.startswith("_")}
        path.write_text(json.dumps(clean, indent=2), encoding="utf-8")
        emit("skill_enabled" if enabled else "skill_disabled", {"name": name}, source="skills")
        return True


_manager: Optional[SkillManager] = None


def get_skill_manager() -> SkillManager:
    global _manager
    if _manager is None:
        _manager = SkillManager()
    return _manager
