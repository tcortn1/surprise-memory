from __future__ import annotations

import time

from openai import OpenAI, APITimeoutError, APIConnectionError, RateLimitError

_SYSTEM_PROMPT = """You are a memory relevance classifier for an AI agent.

Decide whether an observation is worth storing in long-term memory.

STORE if it is:
- A durable fact about a person (name, location, job, preference, belief)
- A stated correction or update to something previously said
- A personal preference or opinion the user expressed
- A fact about the world that is stable and reusable

DISCARD if it is:
- Weather, time, ambient environment ("it's raining", "it's 3pm")
- One-off events with no lasting relevance ("the train was late")
- Pleasantries or filler ("sounds good", "thanks", "sure")
- Transient emotional reactions to momentary events

Reply with exactly one word: STORE or DISCARD."""


def _make_client(model: str) -> tuple[object, str]:
    import os
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        from groq import Groq
        return Groq(api_key=groq_key, timeout=10.0), "llama-3.1-8b-instant"
    return OpenAI(timeout=10.0), model


class RelevanceFilter:
    def __init__(self, model: str = "gpt-4o-mini", max_retries: int = 3) -> None:
        self._client, self._model = _make_client(model)
        self._max_retries = max_retries
        self._cache: dict[str, bool] = {}
        self.fail_open_count: int = 0

    def is_worth_storing(self, text: str) -> bool:
        if text in self._cache:
            return self._cache[text]
        result = self._classify(text)
        self._cache[text] = result
        return result

    def _classify(self, text: str) -> bool:
        delay = 1.0
        for attempt in range(self._max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": text},
                    ],
                    max_tokens=5,
                    temperature=0,
                )
                verdict = response.choices[0].message.content.strip().upper()
                return verdict == "STORE"
            except (APITimeoutError, APIConnectionError, RateLimitError):
                if attempt == self._max_retries - 1:
                    self.fail_open_count += 1
                    return True
                time.sleep(delay)
                delay *= 2
