# Surprise-Memory: Design Document

## What We Built

A lightweight Python library for surprise-gated LLM agent memory. Instead of storing every observation (standard store-everything RAG), only write observations that are genuinely novel or contradictory relative to what is already known. Confirmations and noise are filtered before reaching the store.

Grounded in neuroscience (predictive coding, CLS theory) but built for practical use — not a research system.

---

## The Problem

Most agents use store-everything RAG pipelines:
- Memory stores bloat over time with redundant near-duplicate entries
- Contradictions accumulate rather than triggering belief updates
- Retrieval quality degrades as store size grows
- No principled distinction between high-value and low-value observations

---

## Architecture

Four layers in sequence:

```
observation arrives
      │
      ▼
1. LLM pre-filter (RelevanceFilter — llama-3.3-70b / gpt-4o-mini)
   "Is this a durable fact worth storing at all?"
      ├── NO → discard (reason="filtered")
      └── YES → continue
            │
            ▼
      2. Novelty gate (embedding distance, threshold 0.4)
         "Do we already know something like this?"
            ├── distance > 0.4 → novel → write at strength 1.0
            └── distance ≤ 0.4 → something nearby exists → continue
                  │
                  ▼
            3. Contradiction gate (NLI cross-encoder, nearest neighbor only)
               nli-deberta-v3-large, threshold 0.7
               "Does this conflict with the closest stored memory?"
                  ├── score ≥ 0.7 → decay old memory strength × 0.5
                  │    → deprecate if strength < 0.2, write new
                  └── score < 0.7 → strengthen existing memory + 0.2
                       reason="confirmation", no write
```

Layer 1 uses LLM API (Groq or OpenAI). Layers 2 and 3 run fully locally.

---

## Key Design Decisions

### LLM pre-filter

Screens each observation before it reaches the store. Prompt asks: is this a durable fact (person, preference, belief, correction) or transient noise (weather, filler, one-off events)?

Result: noise_filter jumped from 0.00 to 1.00 with maintained fact_recall of 0.90.

### NLI nearest-neighbor only

Original design checked all top-k neighbors for contradiction. This caused cross-topic false positives — city change contradicting job memory. Fix: only check the single nearest neighbor. Reduces false positives while maintaining recall on genuine contradictions.

Models tested:
- `nli-deberta-v3-small` — misses narrative contradictions ("I moved to Rotterdam" vs "I live in Amsterdam"), NLI score 0.03
- `nli-deberta-v3-large` — catches the same case with score 1.00

Large model is the correct choice.

### Strength-based belief updating

Each memory has a `strength` score (initialized 1.0).
- Confirmation → `strength += 0.2` (cap 2.0), no new write
- Contradiction → `strength *= 0.5`, deprecate only if `strength < 0.2`

A fact confirmed 3x (strength 1.6) resists a single contradiction (→ 0.8, survives). Requires multiple contradictions to fully deprecate a well-confirmed belief.

Maps to CLS theory: synaptic consolidation requires repeated reinforcement. Surprise triggers a candidate update, not an immediate one.

### Soft deprecation via metadata

Deprecated memories stay in ChromaDB with `deprecated=True`. Never returned in queries, but preserved for audit.

---

## Stack

| Component | Choice | Notes |
|---|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Local, fast, 384-dim, L2-normalized |
| Vector store | ChromaDB (persistent) | Local, no infra, cosine distance |
| Entailment | `cross-encoder/nli-deberta-v3-large` | ~400MB, CPU-runnable, nearest neighbor only |
| LLM pre-filter | `llama-3.3-70b-versatile` via Groq (or `gpt-4o-mini`) | ~$0.00001/observation or free on Groq |
| LLM judge (eval) | `llama-3.3-70b-versatile` via Groq | For scoring retrieval quality |
| Eval dataset | LoCoMo (`KhangPTT373/locomo` on HuggingFace) | Real multi-session conversations |

---

## Repo Structure

```
surprise-memory/
├── surprise_memory/
│   ├── __init__.py             # public exports
│   ├── memory.py               # MemoryManager, MemoryConfig, WriteResult
│   ├── filters.py              # Embedder, NLIScorer, LLMContradictionChecker
│   ├── relevance_filter.py     # RelevanceFilter (LLM pre-filter, Groq/OpenAI)
│   └── store.py                # ChromaDB wrapper with strengthen/weaken
├── eval/
│   ├── synthetic.py            # dataset generator + stability scenario
│   ├── run_eval.py             # synthetic eval: store-everything vs filtered-strength
│   └── locomo_eval.py          # LoCoMo benchmark: cosine + LLM judge scorer
├── demo.py                     # step-by-step gate visualization
├── tests/
│   └── test_memory.py          # 13 unit tests, all mocked
└── pyproject.toml
```

---

## Tuned Parameters

| Parameter | Value | Why |
|---|---|---|
| Novelty threshold | 0.4 | 0.7 caused 50% fact recall — Alex facts cluster below 0.7 |
| Contradiction threshold | 0.7 | More sensitive than 0.85, stable on synthetic data |
| NLI top-k for contradiction | 1 (nearest only) | top-5 caused cross-topic false positives |
| Strength increment | 0.2 | Confirmation adds 0.2 per reinforcement |
| Strength decay factor | 0.5 | One contradiction halves strength |
| Deprecation threshold | 0.2 | Requires ~3 unresisted contradictions to deprecate |

---

## Eval Results

### Synthetic (10 facts, 5 confirmations, 5 contradictions, 10 noise about "Alex")

| Baseline | store_size | fact_recall | noise_filter |
|---|---|---|---|
| store-everything | 26 | 1.00 | 0.40 |
| filtered-strength | ~53 | 0.90 | 1.00 |

### LoCoMo (n=1, conv-26, 419 turns, 199 questions)

| Baseline | avg_store_size | cosine_hit_rate | llm_hit_rate |
|---|---|---|---|
| store-everything | 419 | 0.10 | 0.24 |
| filtered-strength | 51 | 0.17 | 0.37 |

**88% store compression. 54% better retrieval quality (LLM judge).**

### Stability test

Scenario: fact confirmed 3x (strength → 1.6), then contradicted once.

| Strategy | Amsterdam survives? |
|---|---|
| instant-deprecate | No — wiped immediately |
| strength-decay | Yes — strength 1.6 → 0.8, above threshold |

---

## Demo Edge Cases (large NLI + nearest-only)

| Input | Expected | Result |
|---|---|---|
| "The weather is nice" | filtered | ✅ |
| "I'm based in Amsterdam" | confirmation | ✅ NLI 0.00 |
| "Actually, I moved to Rotterdam" | contradiction | ✅ NLI 1.00 |
| "I still work as a software engineer" | confirmation (not contradiction) | ✅ NLI 0.00 |
| "I used to work in finance" | confirmation (temporal, not contradiction) | ✅ NLI 0.00 |
| "I've decided to go vegetarian" | contradiction of "I eat meat regularly" | ✅ NLI 1.00 |

---

## Comparison to Related Work

| System | Reduces size? | Resolves contradictions? | Training required? | Operational complexity |
|---|---|---|---|---|
| Store-everything | No | No | No | Low |
| SimpleMem | Yes (compression) | No | No | Medium |
| D-MEM | Yes (gating) | Yes | Yes (Critic Router) | High |
| This system | Yes (gating) | Partially | No | Low |

SimpleMem reports 43% F1 on LoCoMo-10. Our LLM judge hit rate of 0.37 on n=1 is in the same ballpark — different metric, different setup, not a direct comparison.

D-MEM uses a learned Critic Router (requires training) and knowledge graph backend. This system uses pre-trained models only, drops in without infrastructure.

---

## Known Limitations

**NLI recall ~40% on synthetic data.** Real conversation contradictions harder than template negations. Large model improves this significantly.

**Write latency.** Pre-filter: ~500ms API call. NLI: ~200-400ms local. Total per non-novel write: ~700ms-1s. Fine for async/background; too slow for real-time chat.

**Groq free tier: 30 RPM.** Limits eval speed. Pre-filter cache helps (repeated phrases free). Judge scoring paced at 5 concurrent with 2.1s sleep.

**LoCoMo n=1 results.** Full 11-conversation results pending. Pattern consistent across n=1 and n=5 runs.

**Cosine similarity scorer underestimates hit rate.** LLM judge is the right metric — cosine between retrieved utterance and short answer string is systematically low.

---

## Open Questions

- Does 0.37 LLM hit rate hold across all 11 LoCoMo conversations?
- Does strength-based updating improve hit rate vs instant-deprecate on real data?
- Is the novelty threshold stable across domains, or needs per-domain calibration?
- Fine-tuned local NLI on conversational data — how much improvement over large DeBERTa?
