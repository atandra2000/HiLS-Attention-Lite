"""Landmark router: scoring, causal top-k selection, fusion, balance (Phase 1.3).

Contract (plan §1.3, DESIGN §2.3): S(i) = {i} ∪ top-(k−1) of mean_h s_ij over
j ≤ i; selection is head-shared, fusion weights are per head; ties break by
chunk index (higher wins — matches models/chunking.py:causal_chunk_candidates).
Padding semantics for rows with fewer than k admissible chunks: padding slots
repeat the own-chunk index and carry -inf gathered scores, so they get exactly
zero fusion weight and never violate causality.
"""
import math

import pytest
import torch

from models.chunking import causal_chunk_candidates
from models.landmarks import LandmarkProjector, pooled_query
from models.router import (
    balance_loss,
    fusion_weights,
    retrieval_scores,
    select_chunks,
    selection_stats,
)


def _gqa_map(n_heads: int = 4, n_kv: int = 2) -> torch.Tensor:
    return (torch.arange(n_heads) // (n_heads // n_kv)).long()


def test_topk_exact_semantics():
    torch.manual_seed(1)
    B, H, N, k = 2, 4, 8, 3
    scores = torch.randn(B, H, N, N, dtype=torch.float64)
    selected, fused = select_chunks(scores, k)
    mean = scores.mean(dim=1)
    for b in range(B):
        for i in range(N):
            # hand-sorted reference: score desc, ties toward higher chunk index
            adm = sorted(
                ((mean[b, i, j].item(), j) for j in range(i + 1) if j != i),
                key=lambda t: (-t[0], -t[1]),
            )
            expect = [i] + [j for _, j in adm[: k - 1]]
            assert selected[b, i, : len(expect)].tolist() == expect
            assert (selected[b, i, len(expect) :] == i).all()  # padding = own
            # fused scores are the gathered originals; padding slots are -inf
            for s, j in enumerate(expect):
                assert fused[b, :, i, s].tolist() == pytest.approx(
                    scores[b, :, i, j].tolist()
                )
            assert torch.isinf(fused[b, :, i, len(expect) :]).all()

    # adversarial all-tied scores collapse onto causal_chunk_candidates exactly
    sel_t, _ = select_chunks(torch.zeros(B, H, N, N, dtype=torch.float64), k)
    ref = causal_chunk_candidates(N, k)
    for b in range(B):
        for i in range(N):
            assert sel_t[b, i, : len(ref[i])].tolist() == ref[i]
            assert (sel_t[b, i, len(ref[i]) :] == i).all()


def test_own_chunk_always_selected():
    torch.manual_seed(2)
    N, k = 16, 8
    selected, _ = select_chunks(torch.randn(2, 4, N, N, dtype=torch.float64), k)
    assert (selected[:, :, 0] == torch.arange(N)).all()  # slot 0 is own, every row


def test_selection_is_causal():
    torch.manual_seed(3)
    N, k = 16, 8
    selected, _ = select_chunks(torch.randn(2, 4, N, N, dtype=torch.float64), k)
    assert (selected <= torch.arange(N)[:, None]).all()  # no future chunk, any row/slot


def test_fusion_weights_sum_to_one():
    torch.manual_seed(4)
    B, H, N, k = 2, 4, 16, 8
    scores = torch.randn(B, H, N, N, dtype=torch.float64)
    selected, fused = select_chunks(scores, k)
    g = fusion_weights(fused)
    assert g.shape == (B, H, N, k)
    assert torch.allclose(g.sum(dim=-1), torch.ones(B, H, N, dtype=torch.float64))
    # sparse scatter: exactly zero weight outside the selected set, positive inside
    full = torch.zeros(B, H, N, N, dtype=torch.float64).scatter_add_(
        3, selected.unsqueeze(1).expand(B, H, N, k), g
    )
    member = (selected[..., None] == torch.arange(N)).any(dim=-2)  # (B, N, N)
    member4 = member.unsqueeze(1).expand(B, H, N, N)
    assert full.masked_fill(member4, 0.0).abs().max() == 0
    assert (full[member4] > 0).all()


def test_retrieval_score_gradient_flow():
    # the "natively trainable" property: fusion loss grads reach W_ℓ and q̄
    torch.manual_seed(5)
    B, H, KV, T, D, C, k = 2, 4, 2, 512, 8, 128, 4
    proj = LandmarkProjector(n_kv_heads=KV, head_dim=D).double()
    q = torch.randn(B, H, T, D, dtype=torch.float64, requires_grad=True)
    key = torch.randn(B, KV, T, D, dtype=torch.float64, requires_grad=True)
    pooled = pooled_query(q, C)
    scores = retrieval_scores(pooled, proj(key, C), _gqa_map(H, KV), D)
    assert scores.shape == (B, H, T // C, T // C)
    _, fused = select_chunks(scores, k)
    g = fusion_weights(fused)
    (g * torch.linspace(1.0, 0.5, k, dtype=torch.float64)).sum().backward()
    assert proj.proj.weight.grad is not None
    assert proj.proj.weight.grad.abs().sum() > 0  # compression side (W_ℓ)
    assert q.grad is not None and q.grad.abs().sum() > 0  # query side (q̄)


def test_balance_loss_penalizes_collapse():
    N = 8
    uniform = torch.full((N,), 4.0, dtype=torch.float64)
    collapse = torch.zeros(N, dtype=torch.float64)
    collapse[0] = uniform.sum()  # every selection lands on chunk 0
    l_uniform = balance_loss(uniform, N)
    l_collapse = balance_loss(collapse, N)
    assert abs(l_uniform.item() - 1.0) < 1e-12  # closed form: = 1 at uniform
    assert l_collapse.item() > 1.0
    assert abs(l_collapse.item() - N) < 1e-12  # fully collapsed ⇒ L_bal = N
    # (B, N) batch counts aggregate to the same scalar
    assert torch.allclose(balance_loss(uniform.repeat(3, 1), N), l_uniform)


def test_selection_stats_logged():
    N, k = 4, 2
    rotated = torch.tensor([[[0, 1], [1, 2], [2, 3], [3, 0]]])
    stats = selection_stats(rotated, N)
    assert stats["entropy"] == pytest.approx(math.log(4))
    assert stats["max_share"] == pytest.approx(0.25)
    assert stats["per_layer"] == [2, 2, 2, 2]
    collapsed = torch.zeros(1, N, k, dtype=torch.long)
    stats_c = selection_stats(collapsed, N)
    assert stats_c["entropy"] == pytest.approx(0.0)
    assert stats_c["max_share"] == pytest.approx(1.0)
    assert stats_c["per_layer"] == [4, 0, 0, 0]  # padding duplicates dedup