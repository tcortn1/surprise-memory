# surprise-memory

Lightweight, surprise-gated memory for LLM agents. Keeps the store lean by only writing observations that are genuinely new or contradictory. Inspired by predictive coding and CLS theory from neuroscience.

## The problem

Store-everything RAG pipelines bloat over time. After hundreds of conversations, the memory store contains redundant entries, outdated beliefs, and noise. Retrieval quality degrades. This system filters aggressively before writing, and updates beliefs rather than accumulating contradictions.

## How it works

Each observation passes through four gates:

1. **LLM pre-filter** — is this a durable fact or transient noise?
2. **Novelty gate** — is this already represented in the store?
3. **Contradiction gate** — does this conflict with the nearest stored belief?
4. **Strength mechanism** — confirmations reinforce, contradictions decay

```python
from surprise_memory import MemoryManager

memory = MemoryManager()
memory.write("Alex lives in Amsterdam")   # novel → stored, strength 1.0
memory.write("Alex is based in Amsterdam")  # confirmation → strength 1.2
memory.write("Alex moved to Rotterdam")   # contradiction → Amsterdam decays to 0.6
memory.retrieve("where does Alex live")   # returns Rotterdam
```

## Results (LoCoMo benchmark, n=1)

| Baseline | Store size | LLM hit rate |
|---|---|---|
| Store-everything | 419 | 0.24 |
| Filtered-strength | 51 | 0.37 |

88% store compression, 54% better retrieval quality.

## Setup

```bash
poetry install
export GROQ_API_KEY=gsk_...   # free at console.groq.com — used for pre-filter and eval judge
# or: export OPENAI_API_KEY=sk-...
```

## Run the demo

```bash
poetry run python demo.py
```

Shows each gate firing in real time on a sequence of observations about a fictional user.

## Run eval

```bash
# synthetic dataset (fast, no API calls for write phase)
poetry run python eval/run_eval.py

# LoCoMo benchmark (real conversations, requires API key)
poetry run python eval/locomo_eval.py --n 5
poetry run python eval/locomo_eval.py --n 11
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

D-MEM uses a learned Critic Router and knowledge graph. This system uses pre-trained models only — no training pipeline, no graph DB. Drop in and run.

## Stack

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (local)
- Vector store: ChromaDB (local, persistent)
- Contradiction: `cross-encoder/nli-deberta-v3-large` (local)
- Pre-filter: `llama-3.3-70b-versatile` via Groq or `gpt-4o-mini` via OpenAI
