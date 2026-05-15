"""
ChromaDB-backed local vector memory for JARVIS.

This is the Phase 1 replacement path for the flat memory_store.json episode
search. The legacy JSON file can still exist as a migration source, while new
event-driven memory writes land in a local Chroma collection.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.paths import user_data_dir


DEFAULT_CHROMA_DIR = user_data_dir() / "chroma"
DEFAULT_COLLECTION = "jarvis_memory"
LEGACY_MEMORY_FILE = Path.cwd() / "memory_store.json"


@dataclass(slots=True)
class MemoryRecord:
    id: str
    text: str
    metadata: dict[str, Any]


class ChromaVectorMemory:
    """Async facade around the local ChromaDB persistent client."""

    def __init__(self, persist_dir: Path = DEFAULT_CHROMA_DIR, collection_name: str = DEFAULT_COLLECTION):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    async def initialize(self) -> None:
        await asyncio.to_thread(self._initialize_sync)

    async def remember(
        self,
        text: str,
        *,
        record_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        record_id = record_id or f"mem_{int(time.time() * 1000)}"
        await asyncio.to_thread(self._remember_sync, record_id, text, metadata or {})
        return record_id

    async def search(self, query: str, top_k: int = 5) -> list[MemoryRecord]:
        return await asyncio.to_thread(self._search_sync, query, top_k)

    async def migrate_legacy_json(self, path: Path = LEGACY_MEMORY_FILE, limit: int | None = None) -> int:
        return await asyncio.to_thread(self._migrate_legacy_json_sync, path, limit)

    def _initialize_sync(self) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("ChromaDB is not installed. Run: pip install chromadb") from exc

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._client.get_or_create_collection(name=self.collection_name)

    def _collection_sync(self):
        if self._collection is None:
            self._initialize_sync()
        return self._collection

    def _remember_sync(self, record_id: str, text: str, metadata: dict[str, Any]) -> None:
        collection = self._collection_sync()
        clean_metadata = {
            key: value if isinstance(value, (str, int, float, bool)) or value is None else json.dumps(value)
            for key, value in metadata.items()
        }
        collection.upsert(ids=[record_id], documents=[text], metadatas=[clean_metadata])

    def _search_sync(self, query: str, top_k: int) -> list[MemoryRecord]:
        collection = self._collection_sync()
        result = collection.query(query_texts=[query], n_results=top_k)
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        return [
            MemoryRecord(id=record_id, text=document or "", metadata=metadata or {})
            for record_id, document, metadata in zip(ids, documents, metadatas)
        ]

    def _migrate_legacy_json_sync(self, path: Path, limit: int | None) -> int:
        if not path.exists():
            return 0
        data = json.loads(path.read_text(encoding="utf-8"))
        episodes = data.get("episodes", [])
        if limit:
            episodes = episodes[-limit:]
        count = 0
        for episode in episodes:
            record_id = str(episode.get("id") or f"legacy_{count}")
            text = episode.get("summary") or episode.get("user_msg") or json.dumps(episode, ensure_ascii=False)
            metadata = {
                "source": "legacy_memory_store",
                "timestamp": episode.get("timestamp", ""),
                "tags": episode.get("tags", []),
                "importance_score": episode.get("importance_score", 0.0),
            }
            self._remember_sync(record_id, text, metadata)
            count += 1
        return count
