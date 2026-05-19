# surprise-memory

Lightweight, surprise-gated memory for LLM agents. Keeps the store lean by only writing observations that are genuinely new or contradictory. Inspired by predictive coding and CLS theory from neuroscience.

## The problem

Store-everything RAG pipelines bloat over time. After hundreds of conversations, the memory store contains redundant entries, outdated beliefs, and noise. Retrieval quality degrades. This system filters aggressively before writing, and updates beliefs rather than accumulating contradictions.

## How it works

Each observation passes through four gates in sequence:

1. **LLM pre-filter** — is this a durable fact or transient noise?
2. **Contradiction gate** — does this conflict with the nearest stored belief? (runs before novelty — contradictions are semantically different from existing beliefs and would bypass NLI if checked after)
3. **Novelty gate** — is this already represented in the store?
4. **Strength mechanism** — confirmations reinforce, contradictions decay

```python
from surprise_memory import MemoryManager

memory = MemoryManager()
memory.write("Alex lives in Amsterdam")   # novel → stored, strength 1.0
memory.write("Alex is based in Amsterdam")  # confirmation → strength 1.2
memory.write("Alex moved to Rotterdam")   # contradiction → Amsterdam decays to 0.6
memory.retrieve("where does Alex live")   # returns Rotterdam
```

## Results (LoCoMo benchmark, n=10, 1980 questions)

Generation F1: retrieve top-5, LLM generates answer, token F1 vs gold.

| Baseline | Avg store size | Gen F1 |
|---|---|---|
| Store-everything | 588 | 0.203 |
| Filtered-strength | **70.6** | **0.209** |

**88% store compression. Equivalent retrieval quality.**

Filtering achieves 8× smaller store with no quality loss. Smaller stores mean lower latency, lower cost, and less retrieval noise at scale — without sacrificing accuracy.

## Setup

```bash
poetry install
export OPENAI_API_KEY=sk-...  # used for pre-filter and generation F1 scoring
```

## Run the demo

```bash
poetry run python demo.py
```

Shows each gate firing in real time on a sequence of observations about a fictional user.

## Run eval

```bash
# targeted demo: contradiction, belief strength, noise — side-by-side vs store-everything
poetry run python eval/demo_vs_store_everything.py

# synthetic dataset (fast, no API calls for write phase)
poetry run python eval/run_eval.py

# LoCoMo benchmark (real conversations, requires OpenAI API key)
poetry run python eval/locomo_eval.py --n 5
poetry run python eval/locomo_eval.py --n 11 --only-filtered  # skip store-everything baseline
```

## Run tests

```bash
poetry run pytest   # 13 tests, all mocked, no GPU needed
```

## Architecture

```
surprise_memory/
├── memory.py           # MemoryManager — orchestrates all gates
├── filters.py          # Embedder, NLIScorer, LLMContradictionChecker
├── relevance_filter.py # LLM pre-filter (Groq or OpenAI)
└── store.py            # ChromaDB wrapper with strength tracking
```

## Configuration

```python
from surprise_memory import MemoryManager, MemoryConfig

memory = MemoryManager(MemoryConfig(
    novelty_threshold=0.4,        # cosine distance — higher = stricter novelty
    contradiction_threshold=0.7,  # NLI score — lower = more sensitive
    use_strength=True,            # enable strength-based belief decay
    strength_increment=0.2,       # confirmation reinforcement
    strength_decay_factor=0.5,    # contradiction decay multiplier
    deprecation_threshold=0.2,    # deprecate when strength drops below this
))
```

All thresholds are tunable. `SURPRISE_MEMORY_` env prefix overrides any config field.

## Comparison to related work

| System | Reduces store? | Resolves contradictions? | Training required? |
|---|---|---|---|
| SimpleMem | Yes (compression) | No | No |
| D-MEM | Yes (gating) | Yes | Yes |
| This system | Yes (gating) | Yes | No |


## Stack

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (local)
- Vector store: ChromaDB (local, persistent)
- Contradiction: `cross-encoder/nli-deberta-v3-large` (local)
- Pre-filter: `gpt-4o-mini` via OpenAI
