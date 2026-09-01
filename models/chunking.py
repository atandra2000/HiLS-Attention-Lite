"""Chunk tiling and causal candidate sets for HiLS chunk attention.

Pure-shape utilities (no tensors): the sequence must tile into equal chunks
of C tokens, and the causal candidate structure is the score-free reference
that models/router.py:select_chunks must reproduce as exact set equality —
chunk i selects [i, i-1, i-2, …][:k], so tied scores (break by index,
higher wins) collapse onto the same sets."""
from __future__ import annotations


def validate_chunkable(seq_len: int, chunk_len: int) -> None:
    """Enforce exact chunk tiling: positive lengths, seq_len % chunk_len == 0.

    Assertion, not floor: N = T/C drives the router, landmark meanpool, and
    KV gather; a ragged tail chunk would desynchronize all three."""
    if chunk_len <= 0:
        raise ValueError(f"chunk_len must be positive, got {chunk_len}")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if seq_len % chunk_len != 0:
        raise ValueError(
            f"seq_len {seq_len} is not divisible by chunk_len {chunk_len}; "
            "HiLS requires exact chunk tiling (N = T/C)"
        )


def chunk_index_bounds(seq_len: int, chunk_len: int) -> tuple[int, int]:
    """Half-open valid chunk-index range (0, N) for a tileable sequence."""
    validate_chunkable(seq_len, chunk_len)
    return 0, seq_len // chunk_len


def causal_chunk_candidates(n_chunks: int, k: int) -> list[list[int]]:
    """Reference causal candidate sets: candidates[i] = [i, i-1, …, 0][:k].

    Own chunk first, remaining slots most-recent-first (ties break toward
    higher chunk index). Deliberately slow and explicit — the ground-truth
    semantics the router's top-k must match exactly. Length is min(i+1, k)."""
    if n_chunks <= 0:
        raise ValueError(f"n_chunks must be positive, got {n_chunks}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    out: list[list[int]] = []
    for i in range(n_chunks):
        out.append([i] + list(range(i - 1, max(-1, i - k), -1)))
    return out