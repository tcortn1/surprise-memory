"""
Targeted demo: three scenarios where filtered-strength beats store-everything.

Run:
    poetry run python eval/demo_vs_store_everything.py
"""
from __future__ import annotations

import tempfile

from openai import OpenAI

from surprise_memory import MemoryConfig, MemoryManager, RelevanceFilter
from surprise_memory.filters import Embedder
from surprise_memory.store import MemoryStore, StoreConfig

TOP_K = 5


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class StoreEverything:
    def __init__(self, persist_dir: str, embedder: Embedder) -> None:
        cfg = StoreConfig(persist_directory=persist_dir)
        self._store = MemoryStore(cfg)
        self._embedder = embedder

    def write(self, text: str) -> None:
        emb = self._embedder.encode(text)
        self._store.add(text, emb)

    def retrieve(self, query: str) -> list[str]:
        emb = self._embedder.encode(query)
        return [r["text"] for r in self._store.query(emb, top_k=TOP_K)]

    def store_size(self) -> int:
        return self._store.count()


class FilteredStrength:
    def __init__(self, persist_dir: str, relevance_filter: RelevanceFilter, use_strength: bool = True) -> None:
        cfg = MemoryConfig(
            persist_directory=persist_dir,
            novelty_threshold=0.4,
            contradiction_threshold=0.7,
            use_strength=use_strength,
            strength_increment=0.2,
            strength_decay_factor=0.5,
            deprecation_threshold=0.2,
            use_llm_contradiction=False,
        )
        self._memory = MemoryManager(cfg)
        self._filter = relevance_filter

    def write(self, text: str) -> None:
        if self._filter.is_worth_storing(text):
            self._memory.write(text)

    def retrieve(self, query: str) -> list[str]:
        return [r.text for r in self._memory.retrieve(query, top_k=TOP_K)]

    def store_size(self) -> int:
        return self._memory.store_size()


# ---------------------------------------------------------------------------
# LLM answer generation
# ---------------------------------------------------------------------------

def ask(retrieved: list[str], question: str, client: OpenAI) -> str:
    if not retrieved:
        return "(no memories retrieved)"
    context = "\n".join(f"- {t}" for t in retrieved)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content":
            f"Use the following memories to answer the question. "
            f"Give a short, direct answer — just the key fact, no extra words.\n\n"
            f"Memories:\n{context}\n\nQuestion: {question}"
        }],
        max_tokens=50,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def run_scenario(
    name: str,
    utterances: list[str],
    question: str,
    gold: str,
    se: StoreEverything,
    fs: FilteredStrength,
    client: OpenAI,
    top_k: int = TOP_K,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"SCENARIO: {name}")
    print(f"{'=' * 60}")
    print(f"Gold answer: {gold}")
    print(f"\nUtterances written ({len(utterances)}):")
    for u in utterances:
        se.write(u)
        fs.write(u)
        print(f"  > {u}")

    print(f"\nStore sizes — store-everything: {se.store_size()}  |  filtered-strength: {fs.store_size()}")

    se_emb = se._embedder.encode(question)
    fs_emb = se._embedder.encode(question)
    se_retrieved = [r["text"] for r in se._store.query(se_emb, top_k=top_k)]
    fs_retrieved = fs.retrieve(question)

    print(f"\nQuestion: {question}")
    print(f"\nstore-everything retrieved (top {top_k}):")
    for r in se_retrieved:
        print(f"  [{r[:90]}]")
    se_answer = ask(se_retrieved, question, client)
    print(f"  → answer: {se_answer}")

    print(f"\nfiltered-strength retrieved (top {top_k}):")
    for r in fs_retrieved:
        print(f"  [{r[:90]}]")
    fs_answer = ask(fs_retrieved, question, client)
    print(f"  → answer: {fs_answer}")


def main() -> None:
    client = OpenAI()
    embedder = Embedder()
    rf = RelevanceFilter()

    with tempfile.TemporaryDirectory() as tmpdir:

        # ------------------------------------------------------------------
        # Scenario 1: Contradiction — stale fact vs. updated fact
        # ------------------------------------------------------------------
        se1 = StoreEverything(f"{tmpdir}/se1", embedder)
        fs1 = FilteredStrength(f"{tmpdir}/fs1", rf, use_strength=False)

        run_scenario(
            name="Contradiction — stale fact replaced (top-1 retrieval)",
            utterances=[
                "I live in Amsterdam.",
                "I love the canals here in Amsterdam — it's my home.",
                "By the way, I moved to Rotterdam last month for a new job.",
            ],
            question="Where does this person currently live?",
            gold="Rotterdam",
            se=se1,
            fs=fs1,
            client=client,
            top_k=1,
        )

        # ------------------------------------------------------------------
        # Scenario 2a: Strength — weak belief deprecated by single contradiction
        # ------------------------------------------------------------------
        se2a = StoreEverything(f"{tmpdir}/se2a", embedder)
        fs2a = FilteredStrength(f"{tmpdir}/fs2a", rf, use_strength=True)

        print(f"\n{'=' * 60}")
        print("SCENARIO: Belief strength — WEAK belief deprecated by contradiction")
        print(f"{'=' * 60}")
        print("Gold answer: nurse (but system should return doctor — belief too weak)")
        for u in ["I work as a nurse.", "Actually, I was told you became a doctor."]:
            se2a.write(u)
            fs2a.write(u)
            print(f"  > {u}")
        print(f"\nStore sizes — store-everything: {se2a.store_size()}  |  filtered-strength: {fs2a.store_size()}")
        q = "What is this person's job?"
        se2a_r = [r["text"] for r in se2a._store.query(se2a._embedder.encode(q), top_k=1)]
        fs2a_r = fs2a.retrieve(q)
        print(f"\nstore-everything top-1: {se2a_r} → {ask(se2a_r, q, client)}")
        print(f"filtered-strength top-1: {fs2a_r} → {ask(fs2a_r[:1], q, client)}")

        # ------------------------------------------------------------------
        # Scenario 2b: Strength — strong belief resists single contradiction
        # ------------------------------------------------------------------
        se2b = StoreEverything(f"{tmpdir}/se2b", embedder)
        fs2b = FilteredStrength(f"{tmpdir}/fs2b", rf, use_strength=True)

        print(f"\n{'=' * 60}")
        print("SCENARIO: Belief strength — STRONG belief resists contradiction")
        print(f"{'=' * 60}")
        print("Gold answer: nurse")
        for u in [
            "I work as a nurse at the local hospital.",
            "My nursing shift starts at 7am — I love this job.",
            "I've been a nurse for six years now.",
            "Nursing is my calling — I can't imagine doing anything else.",
            "Actually, I was told you became a doctor.",
        ]:
            se2b.write(u)
            fs2b.write(u)
            print(f"  > {u}")
        print(f"\nStore sizes — store-everything: {se2b.store_size()}  |  filtered-strength: {fs2b.store_size()}")
        se2b_r = [r["text"] for r in se2b._store.query(se2b._embedder.encode(q), top_k=TOP_K)]
        fs2b_r = fs2b.retrieve(q)
        print(f"\nstore-everything: {ask(se2b_r, q, client)}")
        print(f"filtered-strength: {ask(fs2b_r, q, client)}")

        # ------------------------------------------------------------------
        # Scenario 3: Noise — redundant facts crowd out relevant memory
        # ------------------------------------------------------------------
        se3 = StoreEverything(f"{tmpdir}/se3", embedder)
        fs3 = FilteredStrength(f"{tmpdir}/fs3", rf)

        morning_noise = [
            "Coffee is the first thing I have every morning.",
            "I can't start my day without a good cup of coffee.",
            "Nothing beats a fresh coffee to kick off the morning.",
            "Every morning begins with coffee for me — always has.",
            "My morning coffee routine is sacred, I never skip it.",
        ]
        morning_fact = [
            "I go for a 5km run every morning before work — it clears my head.",
        ]

        run_scenario(
            name="Noise — redundant habits crowd out specific morning routine",
            utterances=morning_noise + morning_fact,
            question="What does this person do every morning?",
            gold="go for a 5km run",
            se=se3,
            fs=fs3,
            client=client,
        )


if __name__ == "__main__":
    main()
