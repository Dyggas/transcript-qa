"""Unit tests for the NumPy cosine-similarity retrieval index."""

import pytest

from app.chunking import Chunk
from app.retrieval import RetrievalIndex


def _chunk(i: int) -> Chunk:
    return Chunk(timestamp=f"00:00:0{i}", text=f"chunk {i}", index=i)


def _index() -> RetrievalIndex:
    chunks = [_chunk(0), _chunk(1), _chunk(2)]
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    return RetrievalIndex(chunks, embeddings)


def test_empty_inputs_raise():
    with pytest.raises(ValueError):
        RetrievalIndex([], [])


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        RetrievalIndex([_chunk(0)], [[1.0, 0.0], [0.0, 1.0]])


def test_embedding_dim():
    assert _index().embedding_dim == 3


def test_retrieve_ranks_by_similarity():
    results = _index().retrieve([0.9, 0.1, 0.0], k=3)
    scores = [s for _, s in results]
    # Most-aligned chunk first, scores in descending order.
    assert results[0][0].index == 0
    assert scores == sorted(scores, reverse=True)


def test_retrieve_respects_k():
    results = _index().retrieve([1.0, 0.0, 0.0], k=2)
    assert len(results) == 2


def test_retrieve_clamps_k_to_chunk_count():
    results = _index().retrieve([1.0, 0.0, 0.0], k=99)
    assert len(results) == 3


def test_retrieve_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        _index().retrieve([1.0, 0.0], k=1)


def test_retrieve_rejects_non_1d_query():
    with pytest.raises(ValueError):
        _index().retrieve([[1.0, 0.0, 0.0]], k=1)


def test_zero_norm_query_does_not_crash():
    results = _index().retrieve([0.0, 0.0, 0.0], k=1)
    assert len(results) == 1  # returns something rather than dividing by zero
