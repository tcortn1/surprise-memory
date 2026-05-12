from __future__ import annotations

import tempfile
import uuid
from dataclasses import dataclass
from typing import Protocol

from surprise_memory import MemoryConfig, MemoryManager, WriteResult, RelevanceFilter
from surprise_memory.filters import Embedder
from surprise_memory.store import MemoryStore, StoreConfig
from eval.synthetic import generate_dataset, generate_stability_dataset, SyntheticDataset


class Baseline(Protocol):
    name: str

    def write(self, text: str) -> WriteResult: ...
    def retrieve(self, query: str) -> list: ...
    def store_size(self) -> int: ...


class StoreEverythingBaseline:
    """Writes every observation unconditionally — bypasses all gates."""
    name = "store-everything"

    def __init__(self, persist_dir: str) -> None:
        self._embedder = Embedder()
        store_config = StoreConfig(persist_directory=persist_dir)
        self._store = MemoryStore(store_config)

    def write(self, text: str) -> WriteResult:
        embedding = self._embedder.encode(text)
        mid = self._store.add(text, embedding)
        return WriteResult(written=True, reason="novel", memory_id=mid)

    def retrieve(self, query: str) -> list:
        embedding = self._embedder.encode(query)
        return self._store.query(embedding)

    def store_size(self) -> int:
        return self._store.count()


class FilteredStrengthBaseline:
    """LLM pre-filter + novelty gate + strength-based contradiction decay."""
    name = "filtered-strength"

    def __init__(self, persist_dir: str) -> None:
        config = MemoryConfig(
            persist_directory=persist_dir,
            novelty_threshold=0.4,
            contradiction_threshold=0.7,
            use_strength=True,
            strength_increment=0.2,
            strength_decay_factor=0.5,
            deprecation_threshold=0.2,
        )
        self._memory = MemoryManager(config)
        self._filter = RelevanceFilter()

    def write(self, text: str) -> WriteResult:
        if not self._filter.is_worth_storing(text):
            return WriteResult(written=False, reason="filtered")
        return self._memory.write(text)

    def retrieve(self, query: str) -> list:
        return self._memory.retrieve(query)

    def store_size(self) -> int:
        return self._memory.store_size()


@dataclass
class EvalResult:
    baseline_name: str
    store_size: int
    fact_recall: float
    confirmation_filter_rate: float
    contradiction_resolution_rate: float
    noise_filter_rate: float


def run_eval(dataset: SyntheticDataset, baseline: Baseline) -> EvalResult:
    written_by_fact_key: dict[str, str] = {}
    contradictions_resolved = 0
    confirmation_filtered = 0
    noise_filtered = 0

    for obs in dataset.observations:
        result = baseline.write(obs.text)

        if obs.label == "fact" and obs.fact_key:
            if result.written:
                written_by_fact_key[obs.fact_key] = result.memory_id or ""

        elif obs.label == "confirmation":
            if not result.written:
                confirmation_filtered += 1

        elif obs.label == "contradiction":
            # count as resolved if the contradiction gate fired (deprecated or weakened)
            if result.reason == "contradiction":
                contradictions_resolved += 1

        elif obs.label == "noise":
            if not result.written:
                noise_filtered += 1

    return EvalResult(
        baseline_name=baseline.name,
        store_size=baseline.store_size(),
        fact_recall=len(written_by_fact_key) / dataset.fact_count if dataset.fact_count else 0.0,
        confirmation_filter_rate=confirmation_filtered / dataset.confirmation_count if dataset.confirmation_count else 0.0,
        contradiction_resolution_rate=contradictions_resolved / dataset.contradiction_count if dataset.contradiction_count else 0.0,
        noise_filter_rate=noise_filtered / dataset.noise_count if dataset.noise_count else 0.0,
    )


def print_results(results: list[EvalResult]) -> None:
    col_w = 28
    headers = ["baseline", "store_size", "fact_recall", "conf_filter", "contra_resolve", "noise_filter"]
    print("  ".join(h.ljust(col_w) for h in headers))
    print("-" * (col_w * len(headers) + 2 * (len(headers) - 1)))
    for r in results:
        row = [
            r.baseline_name,
            str(r.store_size),
            f"{r.fact_recall:.2f}",
            f"{r.confirmation_filter_rate:.2f}",
            f"{r.contradiction_resolution_rate:.2f}",
            f"{r.noise_filter_rate:.2f}",
        ]
        print("  ".join(v.ljust(col_w) for v in row))


def run_stability_test(persist_dir: str) -> None:
    dataset = generate_stability_dataset()

    print("\n--- Stability test: 3 confirmations then 1 contradiction ---")
    print("Sequence: fact → confirm → confirm → confirm → contradict\n")

    configs = [
        ("instant-deprecate", MemoryConfig(
            persist_directory=f"{persist_dir}/stability-instant",
            novelty_threshold=0.4,
            contradiction_threshold=0.7,
            use_strength=False,
        )),
        ("strength-decay", MemoryConfig(
            persist_directory=f"{persist_dir}/stability-strength",
            novelty_threshold=0.4,
            contradiction_threshold=0.7,
            use_strength=True,
            strength_increment=0.2,
            strength_decay_factor=0.5,
            deprecation_threshold=0.2,
        )),
    ]

    for name, config in configs:
        memory = MemoryManager(config)
        for obs in dataset.observations:
            result = memory.write(obs.text)
            note = ""
            if result.strengthened_ids:
                note = " (strengthened existing)"
            elif result.deprecated_ids:
                note = " (deprecated old)"
            elif result.reason == "contradiction":
                note = " (weakened, not yet deprecated)"
            print(f"  [{name}] {obs.label:15s} → {result.reason}{note}")

        retrieved = memory.retrieve("Where does Alex live?", top_k=3)
        amsterdam = any("Amsterdam" in r.text for r in retrieved)
        rotterdam = any("Rotterdam" in r.text for r in retrieved)
        print(f"  [{name}] Amsterdam survives: {amsterdam}  Rotterdam stored: {rotterdam}\n")


def main() -> None:
    dataset = generate_dataset(n_facts=10, m_confirmations=5, k_contradictions=5, j_noise=10)
    print(f"Dataset: {dataset.fact_count} facts, {dataset.confirmation_count} confirmations, "
          f"{dataset.contradiction_count} contradictions, {dataset.noise_count} noise\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        baselines: list[Baseline] = [
            StoreEverythingBaseline(f"{tmpdir}/store-everything"),
            FilteredStrengthBaseline(f"{tmpdir}/filtered-strength"),
        ]
        results = []
        for baseline in baselines:
            print(f"Running {baseline.name}...")
            results.append(run_eval(dataset, baseline))

        print()
        print_results(results)
        run_stability_test(tmpdir)


if __name__ == "__main__":
    main()
