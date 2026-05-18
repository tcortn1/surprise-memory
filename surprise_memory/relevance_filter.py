from __future__ import annotations

import json
import os
import time

from openai import OpenAI, APITimeoutError, APIConnectionError, RateLimitError

_CACHE_PATH = ".cache/relevance_filter_cache.json"

_SYSTEM_PROMPT = """You are a memory relevance classifier for an AI agent.

Decide whether an observation is worth storing in long-term memory.

STORE if it CONTAINS:
- A first-person statement about the speaker's own life, facts, events, or preferences ("I went to...", "I work as...", "I chose...")
- A durable fact about a named person (location, job, relationship, identity, belief)
- A stated correction or update to something previously said
- A specific personal event, decision, or experience
- A fact about the world that is stable and reusable (historical events, scientific facts, named places)

DISCARD if it is:
- A reaction or question directed at someone else ("Wow, that's cool!", "What happened?", "How are you?")
- Weather, time, ambient environment ("it's raining", "it's 3pm")
- One-off events with no lasting relevance ("the train was late")
- Pleasantries, greetings, or filler ("sounds good", "thanks", "sure", "hey!")
- Generic emotional affirmations without specific facts ("I cherish family", "that means a lot")
- Transient emotional reactions to momentary events

Reply with exactly one word: STORE or DISCARD."""


def _make_client(model: str) -> tuple[object, str]:
    return OpenAI(timeout=10.0), model


class RelevanceFilter:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        log_path: str | None = None,
    ) -> None:
        self._client, self._model = _make_client(model)
        self._max_retries = max_retries
        self._cache: dict[str, bool] = self._load_cache()
        self.fail_open_count: int = 0
        self._log_path = log_path

    def _load_cache(self) -> dict[str, bool]:
        if os.path.exists(_CACHE_PATH):
            with open(_CACHE_PATH) as f:
                return json.load(f)
        return {}

    def _save_cache(self) -> None:
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        with open(_CACHE_PATH, "w") as f:
            json.dump(self._cache, f)

    def is_worth_storing(self, text: str) -> bool:
        if text in self._cache:
            return self._cache[text]
        result = self._classify(text)
        self._cache[text] = result
        self._save_cache()
        if self._log_path:
            with open(self._log_path, "a") as f:
                verdict = "STORE" if result else "DISCARD"
                f.write(f"{verdict}\t{text}\n")
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
