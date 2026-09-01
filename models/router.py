"""Landmark router: retrieval scores, causal top-k, fusion, balance (Phase 1.3).

The router is trained by the LM loss through the fusion weights — there is no
separate retrieval loss. Selection (top-k indices) is a discrete, eager,
head-shared choice; the *scores* of selected chunks stay differentiable and
re-enter the forward pass as per-head fusion weights (DESIGN §2.4).

Ties break toward higher chunk index via flip + stable argsort (no epsilon):
equal scores resolve to the most-recent chunks, matching the
models/chunking.py:causal_chunk_candidates reference exactly.
"""
import math

import torch
from torch import Tensor


def retrieval_scores(
    pooled_q: Tensor,      # (B, H, N, d_head)
    landmarks: Tensor,     # (B, n_kv, N, d_head)
    kv_group_map: Tensor,  # (H,) int — query head → KV group
    head_dim: int,
) -> Tensor:
    """s_ijh = <q̄_ih, ℓ_j·kv(h)> / sqrt(d_head): (B, H, N, N)."""
    lm = landmarks.index_select(1, kv_group_map)  # (B, H, N, d_head)
    return pooled_q @ lm.transpose(-2, -1) / math.sqrt(head_dim)


def select_chunks(
    scores: Tensor,    # (B, H, Nq, Nk), scores[b, h, r, j]; Nk >= Nq
    n_selected: int,   # k: own chunk + top-(k-1)
    own_chunk: bool = True,
) -> tuple[Tensor, Tensor]:
    """Causal head-shared top-k. Returns (selected (B, Nq, k) int64, fused
    per-head scores (B, H, Nq, k)). Own chunk sits at slot 0. Rows with fewer
    admissible chunks pad by repeating the last valid index (the own chunk)
    with -inf fused scores — zero fusion weight, still causal.

    Query rows are right-aligned: slice-row r is absolute query chunk
    Nk-Nq+r, admissible candidates are j <= that index. Training passes a
    square matrix (Nq == Nk); decode passes one query (Nq=1) against the
    N = i+1 landmarks cached so far — all admissible by construction."""
    B, H, Nq, Nk = scores.shape
    if Nk < Nq:
        raise ValueError(f"need at least one candidate column per query row, got ({Nq}, {Nk})")
    k = n_selected
    if not 1 <= k <= Nk:
        raise ValueError(f"n_selected must be in [1, {Nk}], got {k}")
    dev = scores.device
    pos_c = torch.arange(Nk, device=dev)              # candidate chunk indices
    pos_q = torch.arange(Nk - Nq, Nk, device=dev)     # absolute query indices

    # selection is shared across query heads: score chunks by mean_h s_ij
    mean_scores = scores.mean(dim=1)  # (B, Nq, Nk)
    causal = pos_c[None, :] <= pos_q[:, None]  # admissible: j <= i
    if own_chunk:
        selectable = causal & (pos_c[None, :] != pos_q[:, None])
        fill, n_fill = k - 1, torch.clamp_max(pos_q, k - 1)  # valid others per row
    else:
        selectable = causal
        fill, n_fill = k, torch.clamp_max(pos_q + 1, k)

    masked = mean_scores.masked_fill(~selectable, float("-inf"))
    # descending score with ties toward higher j: flip the j axis, stable
    # ascending sort of the negated scores, map positions back
    order = torch.argsort(masked.flip(-1).neg(), dim=-1, stable=True)
    ranked = (Nk - 1) - order  # (B, Nq, Nk) chunk indices, best first
    slots = torch.arange(fill, device=dev)
    others = torch.where(
        slots[None, None, :] < n_fill[:, None],
        ranked[..., :fill],
        pos_q.view(1, Nq, 1).expand(B, Nq, fill),  # pad with own index
    )

    if own_chunk:
        selected = torch.cat([pos_q.view(1, Nq, 1).expand(B, Nq, 1), others], dim=-1)
        pad = torch.cat(
            [torch.zeros(Nq, 1, dtype=torch.bool, device=dev), slots >= n_fill[:, None]],
            dim=-1,
        )
    else:
        selected = others
        pad = slots >= n_fill[:, None]

    sel = selected[:, None, :, :].expand(B, H, Nq, selected.shape[-1])
    fused = scores.gather(3, sel).masked_fill(pad[None, None], float("-inf"))
    return selected, fused


def fusion_weights(selected_scores: Tensor) -> Tensor:
    """Per-head softmax over the k selected scores: (B, H, N, k).

    -inf padding slots get exactly zero weight; the own chunk competes like
    every other chunk (DESIGN §2.3)."""
    return torch.softmax(selected_scores, dim=-1)


def balance_loss(selection_counts: Tensor, n_chunks: int) -> Tensor:
    """L_bal = N · Σ_j p_j², p_j = c_j / Σ c_j; scalar.

    Herfindahl index: = 1 at perfectly uniform usage, unbounded above
    (collapse ⇒ N). Counts may be (N,) or batched (B, N); the distribution is
    taken over the whole tensor's leading axes."""
    c = selection_counts.sum(dim=0) if selection_counts.dim() > 1 else selection_counts
    p = c / c.sum()
    return n_chunks * (p * p).sum()


def selection_stats(selected: Tensor, n_chunks: int) -> dict:
    """{entropy, max_share, per_layer} for the collapse watchdog.

    Membership is deduplicated per query chunk (padding repeats the own
    index), so counts measure distinct chunk usage. per_layer is the layer's
    per-chunk-index selection histogram."""
    idx = torch.arange(n_chunks, device=selected.device)
    member = (selected[..., None] == idx).any(dim=-2)  # (B, N, n_chunks)
    counts = member.flatten(0, 1).sum(dim=0).to(torch.float64)
    p = counts / counts.sum()
    nz = p[p > 0]
    entropy = float(-(nz * nz.log()).sum())
    return {
        "entropy": entropy,
        "max_share": float(p.max()),
        "per_layer": counts.tolist(),
    }