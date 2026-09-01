"""HiLS hierarchical attention: sparse ≡ eager reference (Phase 2.1).

The critical test pair: the vectorized gather+SDPA path must equal the
explicit per-chunk-loop eager reference implementing the same hierarchical
factorization (DESIGN §2.4, §7.1) — fp64 CPU, atol=1e-5.

Note on the plan's tiny cfg (T=512, C=128, k=8): N=4 < k, so the attention
path clamps n_selected to N — hierarchical all-chunk selection, explicitly
NOT a plain-dense fallback (AGENTS.md rule 6; test_hils_k_equals_n_selects_all
pins the operator identity). test_hils_matches_eager_reference additionally
runs a true-sparse variant (N=8, k=4) that exercises subset gathering and
padded rows.
"""
import math

import pytest
import torch

from models.attention import (
    EagerHiLSAttention,
    HiLSAttention,
    eager_hils_attention_core,
    hils_attention_core,
    precompute_freqs_cis,
)
from models.landmarks import LandmarkProjector, pooled_query
from models.router import fusion_weights, retrieval_scores, select_chunks

B, T, C, K, H, KV, D = 2, 512, 128, 8, 4, 2, 32  # plan §2.1 tiny cfg


def _routed(q, k, v, n_sel):
    """Shared router stage so both cores see identical selection artifacts."""
    proj = LandmarkProjector(n_kv_heads=KV, head_dim=D).double()
    kv_map = (torch.arange(H) // (H // KV)).long()
    scores = retrieval_scores(pooled_query(q, C), proj(k, C), kv_map, D)
    selected, fused = select_chunks(scores, n_sel)
    return selected, fusion_weights(fused), kv_map


def _rand_qkv(B_, T_, dtype=torch.float64):
    torch.manual_seed(7)
    return (
        torch.randn(B_, H, T_, D, dtype=dtype),
        torch.randn(B_, KV, T_, D, dtype=dtype),
        torch.randn(B_, KV, T_, D, dtype=dtype),
    )


def test_hils_matches_eager_reference():
    for T_, n_sel in ((T, min(K, T // C)), (1024, 4)):  # plan cfg + true-sparse
        q, k, v = _rand_qkv(B, T_)
        selected, g, kv_map = _routed(q, k, v, n_sel)
        out_sdpa = hils_attention_core(q, k, v, selected, g, C, kv_map)
        out_eager = eager_hils_attention_core(q, k, v, selected, g, C, kv_map)
        assert out_sdpa.shape == (B, H, T_, D)
        assert torch.allclose(out_sdpa, out_eager, atol=1e-5), (T_, n_sel)


def test_hils_k_equals_n_selects_all():
    N = T // C
    q, k, v = _rand_qkv(B, T)
    selected, g, kv_map = _routed(q, k, v, N)
    for b in range(B):
        for i in range(N):
            assert set(selected[b, i, : i + 1].tolist()) == set(range(i + 1))
    out_sdpa = hils_attention_core(q, k, v, selected, g, C, kv_map)
    out_eager = eager_hils_attention_core(q, k, v, selected, g, C, kv_map)
    assert torch.allclose(out_sdpa, out_eager, atol=1e-5)
    # the factorization is the operator: k=N hierarchical attention is NOT
    # plain full attention (per-chunk softmax + fusion ≠ joint softmax)
    k_h = k.index_select(1, kv_map)
    v_h = v.index_select(1, kv_map)
    full = torch.softmax(q @ k_h.transpose(-2, -1) / math.sqrt(D), dim=-1) @ v_h
    assert (out_sdpa - full).abs().max() > 1e-3


def test_aux_loss_shape_and_scale():
    torch.manual_seed(13)
    mod = HiLSAttention(d_model=128, n_heads=H, n_kv_heads=KV, chunk_len=C,
                        n_selected=K, aux_balance_weight=0.01,
                        rope_theta=500000.0, max_seq_len=T).double()
    x = torch.randn(B, T, 128, dtype=torch.float64)
    freqs = precompute_freqs_cis(D, T, 500000.0, dtype=torch.float64)
    out, aux = mod(x, freqs, return_aux=True)
    assert out.shape == (B, T, 128)
    assert aux.dim() == 0  # scalar
    l_bal = aux.item() / 0.01
    assert 1.0 <= l_bal <= 1.5  # ≈ 1.0 at init: near-uniform soft selection mass
    assert aux.item() < 0.1  # two orders below the ~10.8-nat init CE scale


def test_attn_impl_switch_paths_agree():
    torch.manual_seed(17)
    args = dict(d_model=128, n_heads=H, n_kv_heads=KV, chunk_len=C, n_selected=K,
                aux_balance_weight=0.01, rope_theta=500000.0, max_seq_len=T)
    mod_s = HiLSAttention(**args).double()  # production path
    torch.manual_seed(17)
    mod_e = EagerHiLSAttention(**args).double()  # reference path
    x = torch.randn(B, T, 128, dtype=torch.float64)
    freqs = precompute_freqs_cis(D, T, 500000.0, dtype=torch.float64)
    assert torch.allclose(mod_s(x, freqs), mod_e(x, freqs), atol=1e-5)