"""
vision_memory.py - Screenshot, UI state, and workflow visual memory.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import threading

from core.paths import user_data_dir
from event_bus import emit

VISION_DIR = user_data_dir() / "vision_memory"
VISION_FILE = VISION_DIR / "index.json"


class VisionMemory:
    def __init__(self):
        self._lock = threading.RLock()
        self._items: List[Dict[str, Any]] = []
        VISION_DIR.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        try:
            if VISION_FILE.exists():
                self._items = json.loads(VISION_FILE.read_text(encoding="utf-8"))
        except Exception:
            self._items = []

    def _save(self) -> None:
        tmp = VISION_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._items[-1000:], indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(VISION_FILE)

    def remember_screenshot(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        src = Path(path)
        dest = VISION_DIR / f"{uuid.uuid4().hex}{src.suffix or '.png'}"
        try:
            if src.exists():
                shutil.copy2(src, dest)
            item = {"id": uuid.uuid4().hex, "path": str(dest), "metadata": metadata or {}, "timestamp": time.time()}
            with self._lock:
                self._items.append(item)
                self._save()
            emit("vision_memory_updated", item, source="vision_memory")
            return item
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def retrieve(self, query: str = "", limit: int = 10) -> List[Dict[str, Any]]:
        terms = [t.lower() for t in query.split() if len(t) > 2]
        with self._lock:
            items = list(self._items)
        if not terms:
            return items[-limit:]
        scored = []
        for item in items:
            hay = json.dumps(item.get("metadata", {})).lower()
            score = sum(1 for term in terms if term in hay)
            if score:
                scored.append((score, item))
        return [item for _, item in sorted(scored, key=lambda pair: pair[0], reverse=True)[:limit]]


_vision: Optional[VisionMemory] = None


def get_vision_memory() -> VisionMemory:
    global _vision
    if _vision is None:
        _vision = VisionMemory()
    return _vision
