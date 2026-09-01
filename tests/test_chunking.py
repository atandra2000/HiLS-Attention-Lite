"""Chunk tiling and causal candidate-set semantics (DESIGN §7.5, EXECUTION-PLAN §1.1).

`causal_chunk_candidates` is the reference set semantics models/router.py
must reproduce as exact set equality: chunk i selects [i, i-1, i-2, …][:k]
— own chunk first, remaining slots fill most-recent-first so that tied
scores collapse to this same answer (ties break toward higher index)."""
import pytest

from models.chunking import chunk_index_bounds, causal_chunk_candidates, validate_chunkable


def test_chunk_partition_boundaries():
    for seq_len, n in [(512, 4), (4096, 32), (16384, 128)]:
        start, end = chunk_index_bounds(seq_len, 128)
        assert (start, end) == (0, n)
        assert end * 128 == seq_len  # chunks tile [0, T) exactly
    validate_chunkable(4096, 128)  # no raise


def test_nondivisible_seq_rejected():
    with pytest.raises(ValueError):
        validate_chunkable(500, 128)
    with pytest.raises(ValueError):
        chunk_index_bounds(4095, 128)  # off-by-one must not silently floor


def test_causal_candidates():
    cands = causal_chunk_candidates(5, 3)
    assert cands == [[0], [1, 0], [2, 1, 0], [3, 2, 1], [4, 3, 2]]
    for i, row in enumerate(cands):
        assert row[0] == i  # own chunk always first
        assert len(row) == min(i + 1, 3)
        assert set(row) <= set(range(i + 1))  # candidates of chunk i ⊆ {0..i}
        assert len(set(row)) == len(row)  # no duplicates


def test_candidates_k_semantics():
    # k=1 → own chunk only; k ≥ i+1 → every admissible chunk, own first
    assert causal_chunk_candidates(4, 1) == [[0], [1], [2], [3]]
    assert causal_chunk_candidates(3, 99) == [[0], [1, 0], [2, 1, 0]]