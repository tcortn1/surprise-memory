"""
Surprise-memory demo — shows each gate firing in real time.

A personal assistant receives observations about a user over time.
Watch: novelty gate, confirmation strengthening, contradiction decay, noise filtering.

Usage:
    poetry run python demo.py
"""

from __future__ import annotations

import tempfile
from surprise_memory import MemoryConfig, MemoryManager, RelevanceFilter

OBSERVATIONS = [
    # (text, description)
    ("I live in Amsterdam.",                                        "stating home city"),
    ("The weather here is lovely today.",                           "noise — ambient filler"),
    ("I work as a software engineer.",                              "new fact — job"),
    ("I'm based in Amsterdam.",                                     "confirmation of home city"),
    ("I eat meat regularly.",                                       "new fact — diet"),
    ("Yeah, Amsterdam is my city.",                                 "another confirmation of city"),
    ("Actually, I moved to Rotterdam last month.",                  "explicit contradiction — city changed"),
    ("I still work as a software engineer.",                        "same fact with 'still' — NOT a contradiction"),
    ("I used to work in finance before becoming an engineer.",      "temporal — past fact, NOT a contradiction of current job"),
    ("Rotterdam is great, loving it here.",                         "confirmation of new city — NOT a contradiction"),
    ("I've decided to go vegetarian.",                              "partial update — implicit contradiction of diet"),
    ("I moved to Rotterdam last month.",                            "second contradiction — Amsterdam should resist"),
]

ICONS = {
    "filtered":      "🚫",
    "novel":         "✅",
    "confirmation":  "💪",
    "contradiction": "⚡",
    "redundant":     "⏭️",
}


def print_store(memory: MemoryManager) -> None:
    results = memory.retrieve("where does the user live and work", top_k=10)
    if not results:
        print("  store: (empty)")
        return
    print("  store:")
    for r in results:
        bar = "█" * int(r.strength * 5)
        print(f"    [{bar:<10}] {r.strength:.1f}  {r.text}")


def main() -> None:
    print("=" * 70)
    print("  Surprise-Memory Demo")
    print("  Watching novelty, confirmation, contradiction, and noise gates")
    print("=" * 70)

    relevance_filter = RelevanceFilter()

    with tempfile.TemporaryDirectory() as tmpdir:
        config = MemoryConfig(
            persist_directory=tmpdir,
            nli_model="cross-encoder/nli-deberta-v3-large",
            novelty_threshold=0.4,
            contradiction_threshold=0.7,
            use_strength=True,
            strength_increment=0.2,
            strength_decay_factor=0.5,
            deprecation_threshold=0.2,
            use_llm_contradiction=False,
        )
        memory = MemoryManager(config)

        for text, description in OBSERVATIONS:
            print(f"\n{'─' * 70}")
            print(f"  Input ({description}):")
            print(f"  \"{text}\"")
            print()

            worth_storing = relevance_filter.is_worth_storing(text)

            if not worth_storing:
                print(f"  {ICONS['filtered']} LLM filter: DISCARD — transient noise, skip")
                continue

            print(f"  {ICONS['novel']} LLM filter: STORE — durable fact, proceeding...")

            result = memory.write(text)

            if result.nearest_distance is not None:
                print(f"     Novelty gate: nearest distance = {result.nearest_distance:.2f} "
                      f"(threshold 0.4 — {'NOVEL' if result.nearest_distance > 0.4 else 'not novel'})")

            if result.nli_score is not None:
                print(f"     Contradiction gate: NLI score = {result.nli_score:.2f} "
                      f"(threshold 0.7 — {'CONTRADICTION' if result.nli_score >= 0.7 else 'no contradiction'})")

            icon = ICONS.get(result.reason, "?")

            if result.reason == "novel":
                print(f"  {icon} Written as new memory [strength: 1.0]")
            elif result.reason == "confirmation":
                print(f"  {icon} Confirmation — existing memory strengthened")
            elif result.reason == "contradiction":
                if result.deprecated_ids:
                    print(f"  {icon} Contradiction — old memory deprecated, new belief written")
                else:
                    print(f"  {icon} Contradiction — old memory weakened (not yet deprecated), new belief written")
            elif result.reason == "redundant":
                print(f"  {icon} Redundant — skipped")

            print()
            print_store(memory)

    print(f"\n{'=' * 70}")
    print("  Demo complete.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
