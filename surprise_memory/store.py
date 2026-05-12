from __future__ import annotations

import uuid
from dataclasses import dataclass

import chromadb


@dataclass
class StoreConfig:
    persist_directory: str = "./.chroma"
    collection_name: str = "memories"
    top_k: int = 5


class MemoryStore:
    def __init__(self, config: StoreConfig | None = None) -> None:
        self._config = config or StoreConfig()
        self._client = chromadb.PersistentClient(path=self._config.persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=self._config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        text: str,
        embedding: list[float],
        memory_id: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        mid = memory_id or str(uuid.uuid4())
        meta = metadata or {}
        meta.setdefault("deprecated", False)
        meta.setdefault("strength", 1.0)
        self._collection.add(
            ids=[mid],
            embeddings=[embedding],
            documents=[text],
            metadatas=[meta],
        )
        return mid

    def query(
        self,
        embedding: list[float],
        top_k: int | None = None,
    ) -> list[dict]:
        k = top_k or self._config.top_k
        n = self._collection.count()
        if n == 0:
            return []
        k = min(k, n)
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where={"deprecated": {"$ne": True}},
            include=["documents", "distances", "metadatas"],
        )
        neighbors = []
        for i in range(len(result["ids"][0])):
            neighbors.append({
                "id": result["ids"][0][i],
                "text": result["documents"][0][i],
                "distance": result["distances"][0][i],
                "metadata": result["metadatas"][0][i],
                "strength": result["metadatas"][0][i].get("strength", 1.0),
            })
        return neighbors

    def deprecate(self, memory_id: str) -> None:
        self._collection.update(
            ids=[memory_id],
            metadatas=[{"deprecated": True}],
        )

    def strengthen(self, memory_id: str, increment: float = 0.2, max_strength: float = 2.0) -> float:
        result = self._collection.get(ids=[memory_id], include=["metadatas"])
        current = result["metadatas"][0].get("strength", 1.0)
        new_strength = min(current + increment, max_strength)
        self._collection.update(ids=[memory_id], metadatas=[{"strength": new_strength}])
        return new_strength

    def weaken(
        self,
        memory_id: str,
        factor: float = 0.5,
        deprecation_threshold: float = 0.2,
    ) -> bool:
        """Decay strength by factor. Returns True if memory was deprecated."""
        result = self._collection.get(ids=[memory_id], include=["metadatas"])
        current = result["metadatas"][0].get("strength", 1.0)
        new_strength = current * factor
        if new_strength < deprecation_threshold:
            self._collection.update(ids=[memory_id], metadatas=[{"deprecated": True, "strength": new_strength}])
            return True
        self._collection.update(ids=[memory_id], metadatas=[{"strength": new_strength}])
        return False

    def count(self) -> int:
        all_items = self._collection.get(where={"deprecated": {"$ne": True}})
        return len(all_items["ids"])

    def reset(self) -> None:
        self._client.delete_collection(self._config.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
