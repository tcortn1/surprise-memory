from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from surprise_memory import MemoryConfig, MemoryManager


EMBED_DIM = 384
FAR_DISTANCE = 0.9   # > novelty_threshold (0.7) → novel
NEAR_DISTANCE = 0.1  # < novelty_threshold → not novel


def _make_embedding(value: float = 0.1) -> list[float]:
    return [value] * EMBED_DIM


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.encode.return_value = _make_embedding()
    return embedder


@pytest.fixture
def mock_nli():
    nli = MagicMock()
    nli.contradiction_score.return_value = 0.0
    return nli


@pytest.fixture
def memory(mock_embedder, mock_nli, tmp_path):
    config = MemoryConfig(
        persist_directory=str(tmp_path / "chroma"),
        novelty_threshold=0.7,
        contradiction_threshold=0.85,
        top_k=5,
    )
    with (
        patch("surprise_memory.memory.Embedder", return_value=mock_embedder),
        patch("surprise_memory.memory.NLIScorer", return_value=mock_nli),
    ):
        return MemoryManager(config)


def _stub_neighbors(memory: MemoryManager, distances: list[float]) -> None:
    memory._store.query = MagicMock(return_value=[
        {"id": f"id-{i}", "text": f"text-{i}", "distance": d, "metadata": {}}
        for i, d in enumerate(distances)
    ])


# --- novel write ---

def test_write_to_empty_store_is_always_novel(memory):
    result = memory.write("some observation")
    assert result.written is True


def test_write_novel_returns_written_true(memory):
    _stub_neighbors(memory, [FAR_DISTANCE])
    result = memory.write("some novel observation")
    assert result.written is True


def test_write_novel_reason_is_novel(memory):
    _stub_neighbors(memory, [FAR_DISTANCE])
    result = memory.write("some novel observation")
    assert result.reason == "novel"


def test_write_novel_increases_store_size(memory):
    before = memory.store_size()
    _stub_neighbors(memory, [FAR_DISTANCE])
    memory.write("some novel observation")
    assert memory.store_size() == before + 1


# --- redundant write ---

def test_write_redundant_returns_written_false(memory, mock_nli):
    mock_nli.contradiction_score.return_value = 0.0
    _stub_neighbors(memory, [NEAR_DISTANCE])
    result = memory.write("redundant observation")
    assert result.written is False


def test_write_redundant_reason_is_redundant(memory, mock_nli):
    mock_nli.contradiction_score.return_value = 0.0
    _stub_neighbors(memory, [NEAR_DISTANCE])
    result = memory.write("redundant observation")
    assert result.reason == "redundant"


def test_write_redundant_does_not_increase_store_size(memory, mock_nli):
    mock_nli.contradiction_score.return_value = 0.0
    before = memory.store_size()
    _stub_neighbors(memory, [NEAR_DISTANCE])
    memory.write("redundant observation")
    assert memory.store_size() == before


# --- contradiction write ---

def test_write_contradiction_returns_written_true(memory, mock_nli):
    mock_nli.contradiction_score.return_value = 0.95
    _stub_neighbors(memory, [NEAR_DISTANCE])
    result = memory.write("contradicting observation")
    assert result.written is True


def test_write_contradiction_reason_is_contradiction(memory, mock_nli):
    mock_nli.contradiction_score.return_value = 0.95
    _stub_neighbors(memory, [NEAR_DISTANCE])
    result = memory.write("contradicting observation")
    assert result.reason == "contradiction"


def test_write_contradiction_returns_deprecated_ids(memory, mock_nli):
    mock_nli.contradiction_score.return_value = 0.95
    _stub_neighbors(memory, [NEAR_DISTANCE])
    result = memory.write("contradicting observation")
    assert len(result.deprecated_ids) == 1


def test_write_contradiction_calls_deprecate(memory, mock_nli):
    mock_nli.contradiction_score.return_value = 0.95
    _stub_neighbors(memory, [NEAR_DISTANCE])
    memory._store.deprecate = MagicMock()
    memory.write("contradicting observation")
    memory._store.deprecate.assert_called_once()


# --- retrieve ---

def test_retrieve_returns_empty_when_store_empty(memory):
    results = memory.retrieve("some query")
    assert results == []


def test_retrieve_returns_results_after_write(memory):
    memory.write("Alex lives in Amsterdam")
    results = memory.retrieve("where does Alex live")
    assert len(results) >= 1
