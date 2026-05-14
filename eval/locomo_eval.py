"""
LoCoMo dataset evaluation — real multi-session conversational memory benchmark.

Feeds real conversation turns through each memory strategy, then measures
retrieval quality on QA pairs using embedding similarity to ground truth answers.

Usage:
    poetry run python eval/locomo_eval.py            # first 5 conversations
    poetry run python eval/locomo_eval.py --n 20     # first 20
    poetry run python eval/locomo_eval.py --n all    # full dataset (slow)
"""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass, field

import numpy as np

from surprise_memory import MemoryConfig, MemoryManager, RelevanceFilter
from surprise_memory.filters import Embedder
from surprise_memory.store import MemoryStore, StoreConfig


HIT_THRESHOLD = 0.5  # cosine similarity to count retrieval as a hit
TOP_K_RETRIEVE = 5


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_locomo(n: int | None = 5) -> list[dict]:
    import json as _json
    from pathlib import Path

    cache_path = Path(".cache/locomo_raw.jsonl")
    cache_path.parent.mkdir(exist_ok=True)

    if not cache_path.exists():
        print("Downloading LoCoMo dataset via huggingface_hub...")
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
            files = list(list_repo_files("KhangPTT373/locomo", repo_type="dataset"))
            # prefer full dataset over lite version
            data_files = [f for f in files if f.endswith(".json") or f.endswith(".jsonl")]
            target = next((f for f in data_files if "processed_data.json" in f and "locomo_processed" in f), data_files[0] if data_files else None)
            if not target:
                raise ValueError(f"No JSON files found. Files: {files}")
            print(f"  downloading {target}...")
            raw_path = hf_hub_download(
                repo_id="KhangPTT373/locomo",
                filename=target,
                repo_type="dataset",
            )
            # parse and rewrite as JSONL
            with open(raw_path) as rf:
                content = rf.read().strip()
            if content.startswith("["):
                records = _json.loads(content)
            else:
                records = [_json.loads(line) for line in content.splitlines() if line.strip()]
            with open(cache_path, "w") as wf:
                for rec in records:
                    wf.write(_json.dumps(rec) + "\n")
            print(f"  cached {len(records)} conversations to {cache_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download LoCoMo: {e}")

    print("Loading LoCoMo from local cache...")
    examples = []
    seen_ids: set[str] = set()
    with open(cache_path) as f:
        for line in f:
            if not line.strip():
                continue
            example = _json.loads(line)
            conv_id = str(example.get("conv_id", ""))
            if conv_id in seen_ids:
                continue
            seen_ids.add(conv_id)
            examples.append(example)
            print(f"  loaded conversation {len(examples)}", end="\r")
            if n is not None and len(examples) >= n:
                break
    print()
    return examples


def extract_utterances(conversation: dict) -> list[str]:
    utterances = []
    for session in conversation.get("dialogs", []):
        for message in session.get("messages", []):
            text = message.get("content", "")
            if text and text.strip():
                utterances.append(text.strip())
    return utterances


def extract_qa(conversation: dict) -> list[dict]:
    return conversation.get("qas", []) or []


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class StoreEverythingBaseline:
    name = "store-everything"

    def __init__(self, persist_dir: str, embedder: Embedder) -> None:
        self._embedder = embedder
        self._store = MemoryStore(StoreConfig(persist_directory=persist_dir))

    def write(self, text: str) -> None:
        self._store.add(text, self._embedder.encode(text))

    def retrieve(self, query: str) -> list[str]:
        emb = self._embedder.encode(query)
        return [n["text"] for n in self._store.query(emb, top_k=TOP_K_RETRIEVE)]

    def store_size(self) -> int:
        return self._store.count()

    def reset(self, persist_dir: str) -> None:
        self._store = MemoryStore(StoreConfig(persist_directory=persist_dir))


class FilteredStrengthBaseline:
    name = "filtered-strength"

    def __init__(self, persist_dir: str, shared_filter: RelevanceFilter | None = None) -> None:
        config = MemoryConfig(
            persist_directory=persist_dir,
            novelty_threshold=0.4,
            contradiction_threshold=0.7,
            use_strength=True,
            strength_increment=0.2,
            strength_decay_factor=0.5,
            deprecation_threshold=0.2,
            use_llm_contradiction=False,
        )
        self._memory = MemoryManager(config)
        self._filter = shared_filter or RelevanceFilter()

    def write(self, text: str) -> None:
        if self._filter.is_worth_storing(text):
            self._memory.write(text)

    def retrieve(self, query: str) -> list[str]:
        return [r.text for r in self._memory.retrieve(query, top_k=TOP_K_RETRIEVE)]

    def store_size(self) -> int:
        return self._memory.store_size()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def token_f1(retrieved_texts: list[str], answer: str) -> float:
    """Max token-overlap F1 between any retrieved chunk and the gold answer string.

    Same formula as SQuAD extractive QA F1 — directly comparable to D-MEM (53.5)
    and A-MAC (58.3) LoCoMo numbers.
    """
    if not retrieved_texts or not answer.strip():
        return 0.0
    import re
    def _tokenize(s: str) -> set[str]:
        return set(re.sub(r"[^\w\s]", "", s.lower()).split())
    gold_tokens = _tokenize(answer.strip('"'))
    best = 0.0
    for text in retrieved_texts:
        pred_tokens = _tokenize(text)
        common = pred_tokens & gold_tokens
        if not common:
            continue
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best:
            best = f1
    return best


def generation_f1(retrieved_texts: list[str], question: str, gold_answer: str, client, model: str) -> tuple[float, str]:
    """Generate answer from retrieved context, compute token F1 against gold.
    Returns (f1_score, generated_answer)."""
    if not retrieved_texts or not gold_answer.strip():
        return 0.0, ""
    import time
    from openai import RateLimitError, APITimeoutError, APIConnectionError
    context = "\n".join(f"- {t}" for t in retrieved_texts)
    delay = 5.0
    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content":
                    f"Use the following memories to answer the question. "
                    f"Give a short, direct answer — just the key fact, no extra words.\n\n"
                    f"Memories:\n{context}\n\nQuestion: {question}"
                }],
                max_tokens=50,
                temperature=0,
                timeout=30.0,
            )
            generated = response.choices[0].message.content.strip()
            import re as _re
            def _tok(s: str) -> set[str]:
                return set(_re.sub(r"[^\w\s]", "", s.lower()).split())
            pred_tokens = _tok(generated)
            gold_tokens = _tok(gold_answer.strip('"'))
            if not pred_tokens or not gold_tokens:
                return 0.0, generated
            common = pred_tokens & gold_tokens
            if not common:
                return 0.0, generated
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(gold_tokens)
            f1 = 2 * precision * recall / (precision + recall)
            return f1, generated
        except (RateLimitError, APITimeoutError, APIConnectionError):
            if attempt == 3:
                return 0.0, ""
            time.sleep(delay)
            delay *= 2
    return 0.0, ""


def retrieval_hit(retrieved_texts: list[str], answer: str, embedder: Embedder) -> bool:
    if not retrieved_texts or not answer.strip():
        return False
    answer_emb = np.array(embedder.encode(answer))
    for text in retrieved_texts:
        mem_emb = np.array(embedder.encode(text))
        sim = float(np.dot(answer_emb, mem_emb))
        if sim >= HIT_THRESHOLD:
            return True
    return False


def retrieval_hit_llm(retrieved_texts: list[str], answer: str, client, model: str = "gpt-4o-mini") -> bool:
    if not retrieved_texts or not answer.strip():
        return False
    context = "\n".join(f"- {t}" for t in retrieved_texts)
    import time
    from openai import RateLimitError, APITimeoutError, APIConnectionError
    delay = 5.0
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content":
                    f"Retrieved memories:\n{context}\n\n"
                    f"Expected answer: {answer}\n\n"
                    f"Do the retrieved memories contain enough information to produce this answer? "
                    f"Reply YES or NO only."
                }],
                max_tokens=5,
                temperature=0,
                timeout=30.0,
            )
            return response.choices[0].message.content.strip().upper() == "YES"
        except (RateLimitError, APITimeoutError, APIConnectionError):
            if attempt == 4:
                return False
            time.sleep(delay)
            delay *= 2




# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------

@dataclass
class ConversationResult:
    conversation_id: str
    store_size: int
    n_questions: int
    hits: int
    llm_hits: int = 0
    token_f1_sum: float = 0.0
    gen_f1_sum: float = 0.0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.n_questions if self.n_questions else 0.0

    @property
    def llm_hit_rate(self) -> float:
        return self.llm_hits / self.n_questions if self.n_questions else 0.0

    @property
    def avg_token_f1(self) -> float:
        return self.token_f1_sum / self.n_questions if self.n_questions else 0.0

    @property
    def avg_gen_f1(self) -> float:
        return self.gen_f1_sum / self.n_questions if self.n_questions else 0.0


@dataclass
class LoCoMoResult:
    baseline_name: str
    conversations: list[ConversationResult] = field(default_factory=list)

    @property
    def avg_store_size(self) -> float:
        if not self.conversations:
            return 0.0
        return sum(c.store_size for c in self.conversations) / len(self.conversations)

    @property
    def overall_hit_rate(self) -> float:
        total_q = sum(c.n_questions for c in self.conversations)
        total_hits = sum(c.hits for c in self.conversations)
        return total_hits / total_q if total_q else 0.0

    @property
    def overall_llm_hit_rate(self) -> float:
        total_q = sum(c.n_questions for c in self.conversations)
        total_hits = sum(c.llm_hits for c in self.conversations)
        return total_hits / total_q if total_q else 0.0

    @property
    def overall_token_f1(self) -> float:
        total_q = sum(c.n_questions for c in self.conversations)
        total_f1 = sum(c.token_f1_sum for c in self.conversations)
        return total_f1 / total_q if total_q else 0.0

    @property
    def overall_gen_f1(self) -> float:
        total_q = sum(c.n_questions for c in self.conversations)
        total_f1 = sum(c.gen_f1_sum for c in self.conversations)
        return total_f1 / total_q if total_q else 0.0


def run_locomo_eval(
    dataset,
    baseline_class,
    embedder: Embedder,
    tmpdir: str,
    shared_filter: RelevanceFilter | None = None,
    llm_judge_client=None,
    judge_model: str = "gpt-4o-mini",
    gen_client=None,
    gen_model: str = "gpt-4o-mini",
) -> LoCoMoResult:
    is_store_everything = isinstance(baseline_class, type) and baseline_class.__name__ == "StoreEverythingBaseline"
    result = LoCoMoResult(baseline_name="store-everything" if is_store_everything else "filtered-strength")

    for i, conversation in enumerate(dataset):
        conv_id = str(conversation.get("conv_id", i))
        persist_dir = f"{tmpdir}/{result.baseline_name}/{i}"

        if is_store_everything:
            baseline = StoreEverythingBaseline(persist_dir, embedder)
        else:
            baseline = FilteredStrengthBaseline(persist_dir, shared_filter=shared_filter)

        utterances = extract_utterances(conversation)
        qa_pairs = extract_qa(conversation)
        valid_qa = [qa for qa in qa_pairs if qa.get("answer") and qa.get("question") and isinstance(qa.get("answer"), str)]
        n_turns = len(utterances)

        print(f"  [{result.baseline_name}] conv {conv_id}: {n_turns} turns, {len(valid_qa)} questions")

        for j, utterance in enumerate(utterances):
            baseline.write(utterance)
            print(f"    writing turn {j + 1}/{n_turns}", end="\r")

        print(f"    wrote {n_turns} turns → store size: {baseline.store_size()}    ")

        hits = 0
        llm_hits = 0
        token_f1_sum = 0.0
        gen_f1_sum = 0.0
        n_qa = len(valid_qa)
        for j, qa in enumerate(valid_qa):
            retrieved = baseline.retrieve(qa["question"])
            if retrieval_hit(retrieved, qa["answer"], embedder):
                hits += 1
            token_f1_sum += token_f1(retrieved, qa["answer"])
            if gen_client:
                f1_score, generated = generation_f1(retrieved, qa["question"], qa["answer"], gen_client, gen_model)
                gen_f1_sum += f1_score
                import json as _json
                with open(f"eval/qa_log_{result.baseline_name}.jsonl", "a") as _f:
                    _f.write(_json.dumps({
                        "conv_id": conv_id,
                        "question": qa["question"],
                        "gold": qa["answer"],
                        "generated": generated,
                        "retrieved": retrieved,
                        "gen_f1": f1_score,
                    }) + "\n")
            if llm_judge_client:
                if retrieval_hit_llm(retrieved, qa["answer"], llm_judge_client, model=judge_model):
                    llm_hits += 1
            gen_str = f"  gen_f1: {gen_f1_sum / (j + 1):.2f}" if gen_client else ""
            print(f"    scoring {j + 1}/{n_qa}  token_f1: {token_f1_sum / (j + 1):.2f}{gen_str}", end="\r")
        gen_str = f"  gen_f1: {gen_f1_sum / n_qa:.2f}" if gen_client else ""
        print(f"    scored {n_qa} questions → token_f1: {token_f1_sum / n_qa:.2f}{gen_str}    ")

        result.conversations.append(ConversationResult(
            conversation_id=conv_id,
            store_size=baseline.store_size(),
            n_questions=len(valid_qa),
            hits=hits,
            llm_hits=llm_hits,
            token_f1_sum=token_f1_sum,
            gen_f1_sum=gen_f1_sum,
        ))

    return result


def print_locomo_results(results: list[LoCoMoResult], has_gen_f1: bool = False) -> None:
    col_w = 24
    headers = ["baseline", "avg_store_size", "token_f1"]
    if has_gen_f1:
        headers.append("gen_f1")
    headers.append("total_questions")
    print("  ".join(h.ljust(col_w) for h in headers))
    print("-" * (col_w * len(headers) + 2 * (len(headers) - 1)))
    for r in results:
        total_q = sum(c.n_questions for c in r.conversations)
        row = [
            r.baseline_name,
            f"{r.avg_store_size:.1f}",
            f"{r.overall_token_f1:.3f}",
        ]
        if has_gen_f1:
            row.append(f"{r.overall_gen_f1:.3f}")
        row.append(str(total_q))
        print("  ".join(v.ljust(col_w) for v in row))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", default="5", help="Number of conversations (int or 'all')")
    parser.add_argument("--only-filtered", action="store_true", help="Skip store-everything, run filtered-strength only")
    args = parser.parse_args()

    n = None if args.n == "all" else int(args.n)
    only_filtered = args.only_filtered
    dataset = load_locomo(n=n)
    print(f"Loaded {len(dataset)} conversations\n")

    embedder = Embedder()
    import sys
    log_suffix = "_test" if only_filtered and n == 1 else ""
    shared_filter = RelevanceFilter(log_path=f"eval/filter_decisions{log_suffix}.tsv")

    from openai import OpenAI
    gen_client = OpenAI(timeout=30.0)
    gen_model = "gpt-4o-mini"

    with tempfile.TemporaryDirectory() as tmpdir:
        results = []

        if not only_filtered:
            print("Running store-everything...")
            results.append(run_locomo_eval(dataset, StoreEverythingBaseline, embedder, tmpdir, gen_client=gen_client, gen_model=gen_model))

        print("\nRunning filtered-strength...")
        results.append(run_locomo_eval(dataset, FilteredStrengthBaseline, embedder, tmpdir, shared_filter=shared_filter, gen_client=gen_client, gen_model=gen_model))
        print(f"  filter fail-open events: {shared_filter.fail_open_count} (should be 0 for clean results)")

    print()
    print_locomo_results(results, has_gen_f1=True)

    import json
    from datetime import datetime
    log = {
        "timestamp": datetime.now().isoformat(),
        "n_conversations": n,
        "results": [
            {
                "baseline": r.baseline_name,
                "avg_store_size": r.avg_store_size,
                "token_f1": r.overall_token_f1,
                "gen_f1": r.overall_gen_f1,
                "total_questions": sum(c.n_questions for c in r.conversations),
                "conversations": [
                    {
                        "conv_id": c.conversation_id,
                        "store_size": c.store_size,
                        "token_f1": c.avg_token_f1,
                        "gen_f1": c.avg_gen_f1,
                        "n_questions": c.n_questions,
                    }
                    for c in r.conversations
                ],
            }
            for r in results
        ],
    }
    log_path = f"eval/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nResults saved to {log_path}")


if __name__ == "__main__":
    main()
