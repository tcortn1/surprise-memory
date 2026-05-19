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
1. LLM pre-filter (RelevanceFilter — gpt-4o-mini)
   "Is this a durable fact worth storing at all?"
      ├── NO → discard (reason="filtered")
      └── YES → continue
            │
            ▼
      2. NLI contradiction gate (nli-deberta-v3-large, nearest neighbor only)
         "Does this conflict with the closest stored memory?"
         Runs BEFORE novelty gate — contradictions are semantically different
         from existing beliefs, so they would bypass NLI if checked after novelty.
            ├── score ≥ 0.7 → decay old memory strength × 0.5
            │    → deprecate if strength < 0.2, write new
            └── score < 0.7 → continue
                  │
                  ▼
            3. Novelty gate (embedding distance, threshold 0.4)
               "Do we already know something like this?"
                  ├── distance > 0.4 → novel → write at strength 1.0
                  └── distance ≤ 0.4 → something nearby exists
                       → strengthen existing memory + 0.2
                       reason="confirmation", no write
```

Layer 1 uses LLM API (Groq or OpenAI). Layers 2 and 3 run fully locally.

---

## Key Design Decisions

### LLM pre-filter

Screens each observation before it reaches the store. Prompt uses `STORE if it CONTAINS` framing (not `STORE if it is`) to correctly handle mixed utterances — reaction opener + embedded fact — e.g. "Wow, great! Yesterday I started a pottery class." The `CONTAINS` framing stores the fact while the old `IS` framing discarded the whole utterance.

Key rule: durable facts about named persons, first-person statements about life/preferences/events, stable world facts. Discards reactions, pleasantries, weather, generic emotional affirmations.

Result: noise_filter jumped from 0.00 to 1.00 with maintained fact_recall of 0.90.

### NLI runs before novelty gate

Original design ran NLI only for non-novel observations (distance ≤ threshold). Bug: contradictions are semantically different from existing beliefs, so they pass the novelty gate as "novel" and bypass NLI entirely. Fix: run NLI first, then novelty gate. Contradicting facts now correctly deprecate stale beliefs regardless of embedding distance.

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
│   ├── relevance_filter.py     # RelevanceFilter (LLM pre-filter, OpenAI)
│   └── store.py                # ChromaDB wrapper with strengthen/weaken
├── eval/
│   ├── synthetic.py            # dataset generator + stability scenario
│   ├── run_eval.py             # synthetic eval: store-everything vs filtered-strength
│   ├── locomo_eval.py          # LoCoMo benchmark: generation F1 scorer
│   └── demo_vs_store_everything.py  # three targeted scenarios vs store-everything
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

### LoCoMo (n=10, 10 conversations, 1980 questions — final clean run)

Generation F1: retrieve top-5, LLM generates answer, token F1 vs gold. Same methodology as SimpleMem. Zero fail-open events. OpenAI gpt-4o-mini for pre-filter and generation scoring.

| Baseline | avg_store_size | token_f1 | gen_f1 |
|---|---|---|---|
| store-everything | 588 | 0.093 | 0.203 |
| filtered-strength | **70.6** | **0.099** | **0.209** |

**88% store compression. Equivalent gen_f1 (+0.3%).**

Key insight: filtering gives 8× smaller store with no quality loss. The compression is the primary benefit — lower latency, lower cost, less retrieval noise at scale. Gen_f1 varies per conversation (conv-43, sports/career domain, store-everything wins by 0.10; conv-26, personal narrative, filtered-strength wins by 0.08). Average is a wash.

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

SimpleMem reports 43% F1 on LoCoMo-10. Our gen_f1 of 0.209 on n=10 is lower — likely because SimpleMem uses a compression step (summarizes conversation turns before storing), not raw utterances. Raw utterances limit extractable facts regardless of filter quality.

D-MEM uses a learned Critic Router (requires training) and knowledge graph backend. This system uses pre-trained models only, drops in without infrastructure.

Direct gen_f1 comparison is not valid across these systems due to different dataset subsets and scoring implementations. The meaningful comparison is store-everything vs filtered-strength on the same data — which this eval provides.

---

## Known Limitations

**NLI recall ~40% on synthetic data.** Real conversation contradictions harder than template negations. Large model improves this significantly.

**Write latency.** Pre-filter: ~500ms API call. NLI: ~200-400ms local. Total per non-novel write: ~700ms-1s. Fine for async/background; too slow for real-time chat.

**Groq free tier: 30 RPM.** Limits eval speed. Pre-filter cache helps (repeated phrases free). Judge scoring paced at 5 concurrent with 2.1s sleep.

**Gen_f1 varies by conversation domain.** Sports/career-heavy conversations (conv-43) favor store-everything because specific facts (scores, dates) appear in reactions/ambient utterances that the pre-filter correctly discards. Personal narrative conversations favor filtered-strength. Average is equivalent.

**Fundamental ceiling: raw utterances.** Both baselines are limited by the quality of information in raw conversational turns. Compression (extracting structured facts at write time) or query rewriting at retrieval would be the next meaningful improvement.

---

## Open Questions

- Does strength-based updating improve gen_f1 vs instant-deprecate on real data? (stability test validates mechanism, LoCoMo eval does not isolate this)
- Is the novelty threshold stable across domains, or needs per-domain calibration?
- Fine-tuned local NLI on conversational data — how much improvement over large DeBERTa?
- Structured fact extraction at write time — would this close the gap vs SimpleMem's compression approach?
