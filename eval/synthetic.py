from __future__ import annotations

import random
from dataclasses import dataclass, field


_FACTS: list[tuple[str, str]] = [
    ("location", "Alex lives in Amsterdam."),
    ("job", "Alex works as an engineer."),
    ("pet", "Alex owns a cat."),
    ("hobby", "Alex enjoys cycling."),
    ("language", "Alex speaks Dutch and English."),
    ("food", "Alex prefers vegetarian food."),
    ("transport", "Alex commutes by bike."),
    ("education", "Alex studied computer science."),
    ("music", "Alex likes jazz music."),
    ("sport", "Alex plays tennis on weekends."),
]

_CONTRADICTIONS: dict[str, str] = {
    "location": "Alex lives in Rotterdam.",
    "job": "Alex works as a teacher.",
    "pet": "Alex owns a dog.",
    "hobby": "Alex enjoys swimming.",
    "language": "Alex only speaks English.",
    "food": "Alex loves eating meat.",
    "transport": "Alex drives to work.",
    "education": "Alex studied economics.",
    "music": "Alex prefers classical music.",
    "sport": "Alex plays football on weekends.",
}

_CONFIRMATIONS: dict[str, list[str]] = {
    "location": ["Alex is based in Amsterdam.", "Amsterdam is where Alex lives."],
    "job": ["Alex is an engineer by profession.", "Alex has a job in engineering."],
    "pet": ["Alex has a cat at home.", "Alex is a cat owner."],
    "hobby": ["Alex loves going for bike rides.", "Cycling is Alex's hobby."],
    "language": ["Alex is bilingual in Dutch and English.", "Alex can speak both Dutch and English."],
    "food": ["Alex sticks to vegetarian meals.", "Alex does not eat meat."],
    "transport": ["Alex gets to work by bicycle.", "Alex bikes to the office."],
    "education": ["Alex has a degree in computer science.", "Alex studied CS at university."],
    "music": ["Alex is a fan of jazz.", "Alex enjoys listening to jazz."],
    "sport": ["Alex plays tennis at the weekend.", "Alex is a tennis player."],
}

_NOISE: list[str] = [
    "The weather was nice today.",
    "Coffee is a popular morning drink.",
    "The meeting ran a bit long.",
    "It rained heavily last Tuesday.",
    "The train was delayed by ten minutes.",
    "Someone left dishes in the sink.",
    "The park was busy this afternoon.",
    "A new coffee shop opened downtown.",
    "The movie got mixed reviews.",
    "Traffic was unusually light this morning.",
]


@dataclass
class Observation:
    text: str
    label: str  # "fact" | "confirmation" | "contradiction" | "noise"
    fact_key: str | None = None
    contradicts_fact_key: str | None = None


@dataclass
class SyntheticDataset:
    observations: list[Observation]
    fact_count: int
    confirmation_count: int
    contradiction_count: int
    noise_count: int


def generate_stability_dataset() -> SyntheticDataset:
    """Fixed scenario: one fact confirmed 3x then contradicted once.
    Tests whether confirmation accumulation protects against a single contradiction."""
    observations = [
        Observation(text="Alex lives in Amsterdam.", label="fact", fact_key="location"),
        Observation(text="Alex is based in Amsterdam.", label="confirmation", fact_key="location"),
        Observation(text="Amsterdam is where Alex lives.", label="confirmation", fact_key="location"),
        Observation(text="Alex is based in Amsterdam.", label="confirmation", fact_key="location"),
        Observation(text="Alex lives in Rotterdam.", label="contradiction", contradicts_fact_key="location"),
    ]
    return SyntheticDataset(
        observations=observations,
        fact_count=1,
        confirmation_count=3,
        contradiction_count=1,
        noise_count=0,
    )


def generate_dataset(
    n_facts: int = 10,
    m_confirmations: int = 5,
    k_contradictions: int = 5,
    j_noise: int = 10,
    seed: int = 42,
) -> SyntheticDataset:
    rng = random.Random(seed)

    if n_facts > len(_FACTS):
        raise ValueError(f"Max {len(_FACTS)} facts available, requested {n_facts}")

    selected_facts = rng.sample(_FACTS, n_facts)
    fact_keys = [key for key, _ in selected_facts]

    observations: list[Observation] = []

    for key, text in selected_facts:
        observations.append(Observation(text=text, label="fact", fact_key=key))

    for _ in range(m_confirmations):
        key = rng.choice(fact_keys)
        text = rng.choice(_CONFIRMATIONS[key])
        observations.append(Observation(text=text, label="confirmation", fact_key=key))

    for _ in range(k_contradictions):
        key = rng.choice(fact_keys)
        text = _CONTRADICTIONS[key]
        observations.append(
            Observation(text=text, label="contradiction", contradicts_fact_key=key)
        )

    for _ in range(j_noise):
        text = rng.choice(_NOISE)
        observations.append(Observation(text=text, label="noise"))

    rng.shuffle(observations)

    return SyntheticDataset(
        observations=observations,
        fact_count=n_facts,
        confirmation_count=m_confirmations,
        contradiction_count=k_contradictions,
        noise_count=j_noise,
    )
