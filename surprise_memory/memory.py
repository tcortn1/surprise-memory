from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_settings import BaseSettings, SettingsConfigDict

from .filters import Embedder, NLIScorer, LLMContradictionChecker, is_novel, find_contradictions, find_contradictions_llm
from .store import MemoryStore, StoreConfig


class MemoryConfig(BaseSettings):
    persist_directory: str = "./.chroma"
    collection_name: str = "memories"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    nli_model: str = "cross-encoder/nli-deberta-v3-large"
    novelty_threshold: float = 0.4
    contradiction_threshold: float = 0.7
    top_k: int = 5
    use_strength: bool = False
    strength_increment: float = 0.2
    strength_decay_factor: float = 0.5
    deprecation_threshold: float = 0.2
    use_llm_contradiction: bool = False

    model_config = SettingsConfigDict(env_prefix="SURPRISE_MEMORY_")


@dataclass
class WriteResult:
    written: bool
    reason: str  # "novel" | "contradiction" | "confirmation" | "redundant" | "filtered"
    memory_id: str | None = None
    deprecated_ids: list[str] = field(default_factory=list)
    strengthened_ids: list[str] = field(default_factory=list)
    nearest_distance: float | None = None
    nli_score: float | None = None


@dataclass
class MemoryResult:
    id: str
    text: str
    distance: float
    strength: float = 1.0


class MemoryManager:
    def __init__(self, config: MemoryConfig | None = None) -> None:
        self._config = config or MemoryConfig()
        store_config = StoreConfig(
            persist_directory=self._config.persist_directory,
            collection_name=self._config.collection_name,
            top_k=self._config.top_k,
        )
        self._store = MemoryStore(store_config)
        self._embedder = Embedder(self._config.embed_model)
        self._nli = NLIScorer(self._config.nli_model)
        self._llm_checker = LLMContradictionChecker() if self._config.use_llm_contradiction else None

    def write(self, observation: str) -> WriteResult:
        embedding = self._embedder.encode(observation)
        neighbors = self._store.query(embedding, top_k=self._config.top_k)
        distances = [n["distance"] for n in neighbors]

        nearest_distance = min(distances) if distances else None

        # NLI runs first — contradictions are semantically different (novel) so must
        # be checked before the novelty gate, otherwise they bypass contradiction detection.
        if self._config.use_llm_contradiction and self._llm_checker:
            contradicted = find_contradictions_llm(observation, neighbors[:1], self._llm_checker)
            nli_scores = []
            max_nli = None
        elif neighbors:
            nearest = neighbors[:1]
            nli_scores = [
                self._nli.contradiction_score(n["text"], observation)
                for n in nearest
            ]
            max_nli = max(nli_scores) if nli_scores else 0.0
            contradicted = [
                n["id"] for n, score in zip(nearest, nli_scores)
                if score >= self._config.contradiction_threshold
            ]
        else:
            contradicted = []
            nli_scores = []
            max_nli = None

        if contradicted:
            deprecated_ids = []
            for old_id in contradicted:
                if self._config.use_strength:
                    was_deprecated = self._store.weaken(
                        old_id,
                        factor=self._config.strength_decay_factor,
                        deprecation_threshold=self._config.deprecation_threshold,
                    )
                    if was_deprecated:
                        deprecated_ids.append(old_id)
                else:
                    self._store.deprecate(old_id)
                    deprecated_ids.append(old_id)
            memory_id = self._store.add(observation, embedding)
            return WriteResult(
                written=True,
                reason="contradiction",
                memory_id=memory_id,
                deprecated_ids=deprecated_ids,
                nearest_distance=nearest_distance,
                nli_score=max_nli,
            )

        if is_novel(distances, self._config.novelty_threshold):
            memory_id = self._store.add(observation, embedding)
            return WriteResult(written=True, reason="novel", memory_id=memory_id, nearest_distance=nearest_distance, nli_score=max_nli)

        if self._config.use_strength and neighbors:
            nearest = min(neighbors, key=lambda n: n["distance"])
            self._store.strengthen(
                nearest["id"],
                increment=self._config.strength_increment,
            )
            return WriteResult(
                written=False,
                reason="confirmation",
                strengthened_ids=[nearest["id"]],
                nearest_distance=nearest_distance,
                nli_score=max_nli,
            )

        return WriteResult(written=False, reason="redundant", nearest_distance=nearest_distance, nli_score=max_nli)

    def retrieve(self, query: str, top_k: int | None = None) -> list[MemoryResult]:
        embedding = self._embedder.encode(query)
        neighbors = self._store.query(embedding, top_k=top_k or self._config.top_k)
        return [
            MemoryResult(
                id=n["id"],
                text=n["text"],
                distance=n["distance"],
                strength=n.get("strength", 1.0),
            )
            for n in neighbors
        ]

    def store_size(self) -> int:
        return self._store.count()
