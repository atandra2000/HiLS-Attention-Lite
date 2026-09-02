"""HiLS hierarchical chunk attention (Phase 2.1).

Operator (DESIGN §2.4): for query chunk i, each selected chunk j ∈ S(i) gets
an independent softmax over its own C keys, and the outputs fuse by the
per-head retrieval-score softmax g_ij. Selection is head-shared and eager
(data-dependent top-k indices break torch.compile); everything else runs
batched. Two cores implement the same math:

  - models/attention.py:hils_attention_core    — production: gather selected
    KV to (B, N, k, KV, C, D), one SDPA call per selected slot over (B·N, ·)
  - models/attention.py:eager_hils_attention_core — O(T²) ground truth:
    full (T, T) score matrix, per-chunk softmax reconstructed by masking
    within chunk boundaries; explicit per-query-chunk loop

Causality is a selection mask only (j ≤ i); within a retrieved chunk all C
keys are visible, so no token-level mask exists anywhere in this operator.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.landmarks import LandmarkProjector, pooled_query
from models.router import balance_loss, fusion_weights, retrieval_scores, select_chunks


def precompute_freqs_cis(head_dim: int, max_seq_len: int, theta: float,
                         dtype: torch.dtype = torch.float32) -> Tensor:
    """Llama-style complex rotation table: (max_seq_len, head_dim//2)."""
    half = head_dim // 2
    inv_freq = theta ** (-2.0 * torch.arange(half, dtype=dtype) / head_dim)
    t = torch.arange(max_seq_len, dtype=dtype)
    angle = torch.outer(t, inv_freq)
    return torch.polar(torch.ones_like(angle), angle)


def apply_rope(x: Tensor, freqs_cis: Tensor) -> Tensor:
    """Absolute-position RoPE: x (B, H, T, D), freqs_cis (T, D//2) complex.

    Trig runs in ≥ fp32 (promoted), then casts back — bf16 inputs never see
    bf16 trig precision."""
    work = torch.promote_types(x.dtype, torch.float32)
    xc = torch.view_as_complex(x.to(work).reshape(*x.shape[:-1], -1, 2))
    out = torch.view_as_real(xc * freqs_cis.view(1, 1, *freqs_cis.shape)).flatten(3)
    return out.to(x.dtype)


def hils_attention_core(q: Tensor, k: Tensor, v: Tensor, selected: Tensor,
                        g: Tensor, chunk_len: int, kv_group_map: Tensor) -> Tensor:
    """Production core: per-selected-slot SDPA over (B·N, ·), fused by g.

    q: roped (B, H, T, D); k/v: (B, KV, T, D); selected: (B, N, k);
    g: (B, H, N, k) → (B, H, T, D)."""
    B, H, T, D = q.shape
    KV = k.shape[1]
    N, C = T // chunk_len, chunk_len
    n_sel = selected.size(-1)
    dev = q.device

    q_flat = (q.view(B, H, N, C, D).permute(0, 2, 1, 3, 4)
              .reshape(B * N, H, C, D))
    k_perm = k.view(B, KV, N, C, D).permute(0, 2, 1, 3, 4)  # (B, N, KV, C, D)
    v_perm = v.view(B, KV, N, C, D).permute(0, 2, 1, 3, 4)
    b_idx = torch.arange(B, device=q.device)[:, None, None]
    K_sel = k_perm[b_idx, selected]  # advanced indexing → (B, N, k, KV, C, D)
    V_sel = v_perm[b_idx, selected]

    out = q.new_zeros(B, H, N, C, D)
    for s in range(n_sel):  # one SDPA call per selected slot — the per-chunk
        # softmax is the operator; a single call over k·C keys would be a
        # different (joint) factorization
        o = F.scaled_dot_product_attention(
            q_flat,
            K_sel[:, :, s].reshape(B * N, KV, C, D),
            V_sel[:, :, s].reshape(B * N, KV, C, D),
            enable_gqa=True,
        )  # (B·N, H, C, D)
        o = o.view(B, N, H, C, D).permute(0, 2, 1, 3, 4)  # (B, H, N, C, D)
        out = out + g[..., s].unsqueeze(-1).unsqueeze(-1) * o
    return out.reshape(B, H, T, D)


def eager_hils_attention_core(q: Tensor, k: Tensor, v: Tensor, selected: Tensor,
                              g: Tensor, chunk_len: int, kv_group_map: Tensor) -> Tensor:
    """Reference core, deliberately slow: explicit per-query-chunk loop over a
    full (T, T) score matrix; per-chunk softmax reconstructed by masking within
    chunk boundaries. Ground truth only — O(T²) memory and time."""
    B, H, T, D = q.shape
    N, C = T // chunk_len, chunk_len
    n_sel = selected.size(-1)
    k_h = k.index_select(1, kv_group_map)  # (B, H, T, D)
    v_h = v.index_select(1, kv_group_map)
    S = (q @ k_h.transpose(-2, -1)) / math.sqrt(D)  # (B, H, T, T)

    ar = torch.arange(C, device=q.device)
    out = torch.zeros_like(q)
    for i in range(N):  # one query chunk per iteration
        rows = slice(i * C, (i + 1) * C)
        cols = (selected[:, i, :, None] * C + ar).reshape(B, n_sel * C)  # (B, k·C)
        S_sel = S[:, :, rows].gather(
            3, cols[:, None, None, :].expand(B, H, C, n_sel * C))  # (B, H, C, k·C)
        V_sel = v_h.gather(
            2, cols[:, None, :, None].expand(B, H, n_sel * C, D))
        # softmax within each selected chunk's C keys (independent per slot);
        # padding slots carry g == 0 exactly, so duplicated columns vanish
        p = S_sel.view(B, H, C, n_sel, C).softmax(dim=-1)
        p = (p * g[:, :, i].view(B, H, 1, n_sel, 1)).reshape(B, H, C, n_sel * C)
        out[:, :, rows] = p @ V_sel
    return out


class HiLSAttention(nn.Module):
    """Production path: landmark scoring → causal top-k → gather selected KV →
    per-chunk SDPA (one call per selected slot, batched over B·N) →
    score-weighted fusion. attn_impl='sdpa'.

    n_selected > n_chunks clamps to n_chunks: hierarchical all-chunk selection,
    not a dense fallback (AGENTS.md rule 6 — the operator stays per-chunk)."""

    attn_impl = "sdpa"

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 chunk_len: int, n_selected: int,
                 aux_balance_weight: float, rope_theta: float, max_seq_len: int):
        super().__init__()
        assert n_heads % n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.head_dim = d_model // n_heads
        self.chunk_len, self.n_selected = chunk_len, n_selected
        self.aux_balance_weight = aux_balance_weight
        self.max_seq_len = max_seq_len
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(n_heads * self.head_dim, d_model)
        self.landmarks = LandmarkProjector(n_kv_heads, self.head_dim)
        self.register_buffer(
            "kv_group_map",
            (torch.arange(n_heads) // (n_heads // n_kv_heads)).long(),
            persistent=False)

    def forward(self, x: Tensor, freqs_cis: Tensor,
                return_aux: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """(B, T, d) → (B, T, d) [+ λ·L_bal scalar]. freqs_cis: (T, D//2)."""
        B, T, _ = x.shape
        H, KV, D, C = self.n_heads, self.n_kv_heads, self.head_dim, self.chunk_len
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, KV, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, KV, D).transpose(1, 2)
        q, k = apply_rope(q, freqs_cis), apply_rope(k, freqs_cis)

        n_chunks = T // C
        n_sel = min(self.n_selected, n_chunks)  # all-chunk, never plain-dense
        scores = retrieval_scores(pooled_query(q, C), self.landmarks(k, C),
                                  self.kv_group_map, D)
        selected, fused = select_chunks(scores, n_sel)
        self.last_selected = selected.detach()  # watchdog surface (training loop)
        g = fusion_weights(fused)
        core = hils_attention_core if self.attn_impl == "sdpa" \
            else eager_hils_attention_core
        out = core(q, k, v, selected, g, C, self.kv_group_map)
        out = self.out_proj(out.transpose(1, 2).reshape(B, T, H * D))

        if not return_aux:
            return out
        # soft selection mass per chunk — differentiable into the score path
        # (hard counts would block gradient entirely)
        idx = selected[:, None].expand(B, H, n_chunks, n_sel).reshape(-1)
        counts = g.reshape(-1).new_zeros(n_chunks).index_add(0, idx, g.reshape(-1))
        return out, self.aux_balance_weight * balance_loss(counts, n_chunks)


class EagerHiLSAttention(HiLSAttention):
    """Reference path: identical math via models/attention.py:
    eager_hils_attention_core (O(T²)). Ground truth only — attn_impl='eager'."""

    attn_impl = "eager"