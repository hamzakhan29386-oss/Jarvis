"""
knowledge_base.py - Persistent personal knowledge and lightweight RAG index.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import threading

from core.paths import user_data_dir
from event_bus import emit

KB_FILE = user_data_dir() / "knowledge_base.json"


class KnowledgeBase:
    def __init__(self, path: Path = KB_FILE):
        self.path = path
        self._lock = threading.RLock()
        self._docs: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        try:
            if self.path.exists():
                self._docs = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            self._docs = []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._docs, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def add_document(self, path: str, content: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        p = Path(path)
        if content is None and p.exists() and p.suffix.lower() in {".txt", ".md", ".py", ".js", ".html", ".css", ".json"}:
            content = p.read_text(encoding="utf-8", errors="ignore")
        content = content or ""
        chunks = self._chunk(content)
        doc = {
            "id": uuid.uuid4().hex,
            "path": str(p),
            "metadata": metadata or {},
            "chunks": chunks,
            "summary": content[:500],
            "indexed_at": time.time(),
        }
        with self._lock:
            self._docs.append(doc)
            self._save()
        emit("knowledge_indexed", {"path": str(p), "chunks": len(chunks)}, source="knowledge_base")
        return doc

    def search(self, query: str, limit: int = 5, project: Optional[str] = None) -> List[Dict[str, Any]]:
        terms = [t.lower() for t in query.split() if len(t) > 2]
        hits = []
        with self._lock:
            for doc in self._docs:
                if project and doc.get("metadata", {}).get("project") != project:
                    continue
                for i, chunk in enumerate(doc.get("chunks", [])):
                    text = chunk.lower()
                    score = sum(1 for term in terms if term in text)
                    if score:
                        hits.append({"doc_id": doc["id"], "path": doc["path"], "chunk_index": i, "text": chunk, "score": score})
        return sorted(hits, key=lambda h: h["score"], reverse=True)[:limit]

    def _chunk(self, content: str, size: int = 1200, overlap: int = 150) -> List[str]:
        chunks = []
        i = 0
        while i < len(content):
            chunks.append(content[i:i + size])
            i += max(1, size - overlap)
        return chunks


_kb: Optional[KnowledgeBase] = None


def get_knowledge_base() -> KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
    return _kb
