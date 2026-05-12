from __future__ import annotations

import os

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self._model = SentenceTransformer(model_name, device="cpu")

    def encode(self, text: str) -> list[float]:
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()


class NLIScorer:
    # Label order for cross-encoder/nli-deberta-v3-small: contradiction=0, entailment=1, neutral=2
    CONTRADICTION_IDX = 0

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small") -> None:
        self._model = CrossEncoder(model_name)

    def contradiction_score(self, premise: str, hypothesis: str) -> float:
        scores = self._model.predict([[premise, hypothesis]])
        probs = _softmax(scores[0])
        return float(probs[self.CONTRADICTION_IDX])


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


def is_novel(distances: list[float], threshold: float = 0.7) -> bool:
    if not distances:
        return True
    return min(distances) > threshold


def find_contradictions(
    new_text: str,
    neighbors: list[dict],
    nli_scorer: NLIScorer,
    threshold: float = 0.85,
) -> list[str]:
    contradicted_ids = []
    for neighbor in neighbors:
        if neighbor["metadata"].get("deprecated"):
            continue
        score = nli_scorer.contradiction_score(
            premise=neighbor["text"],
            hypothesis=new_text,
        )
        if score >= threshold:
            contradicted_ids.append(neighbor["id"])
    return contradicted_ids


_LLM_CONTRADICTION_PROMPT = """Does statement B contradict statement A?

A contradiction means B asserts something that cannot simultaneously be true with A.
Confirming, extending, rephrasing, or adding detail does NOT count as contradiction.

Statement A: {premise}
Statement B: {hypothesis}

Reply with exactly one word: YES (contradiction) or NO (not a contradiction)."""


class LLMContradictionChecker:
    """LLM-based contradiction checker. More accurate than NLI for conversational text."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        groq_key = os.environ.get("GROQ_API_KEY")
        if groq_key:
            from groq import Groq
            self._client = Groq(api_key=groq_key, timeout=10.0)
            self._model = "llama-3.1-8b-instant"
        else:
            from openai import OpenAI
            self._client = OpenAI(timeout=10.0)
            self._model = model

    def is_contradiction(self, premise: str, hypothesis: str) -> bool:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{
                "role": "user",
                "content": _LLM_CONTRADICTION_PROMPT.format(
                    premise=premise,
                    hypothesis=hypothesis,
                ),
            }],
            max_tokens=5,
            temperature=0,
        )
        verdict = response.choices[0].message.content.strip().upper()
        return verdict == "YES"


def find_contradictions_llm(
    new_text: str,
    neighbors: list[dict],
    checker: LLMContradictionChecker,
) -> list[str]:
    contradicted_ids = []
    for neighbor in neighbors:
        if neighbor["metadata"].get("deprecated"):
            continue
        if checker.is_contradiction(premise=neighbor["text"], hypothesis=new_text):
            contradicted_ids.append(neighbor["id"])
    return contradicted_ids
