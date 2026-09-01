"""HiLS decoder block: RMSNorm → HiLS attention (+residual) → RMSNorm → SwiGLU (+residual)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import EagerHiLSAttention, HiLSAttention


class RMSNorm(nn.Module):
    """Root-mean-square norm with a learned (initially one) scale, no bias.

    F.rms_norm is the fused stdlib kernel of the manual rsqrt(mean(x^2)) formula."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.weight.size(0),), self.weight, self.eps)


class HiLSBlock(nn.Module):
    """One HiLS layer: chunk-sparse attention sublayer + SwiGLU sublayer, both
    residual. attn_impl selects the kernel (sdpa production / eager reference);
    the block always reports its attention's λ-weighted balance term upward."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int,
                 chunk_len: int, n_selected: int, aux_balance_weight: float,
                 ffn_dim: int, rms_norm_eps: float, attn_impl: str,
                 rope_theta: float, max_seq_len: int):
        super().__init__()
        assert attn_impl in ("sdpa", "eager"), f"unknown attn_impl: {attn_impl!r}"
        self.attn_norm = RMSNorm(d_model, rms_norm_eps)
        attn_cls = EagerHiLSAttention if attn_impl == "eager" else HiLSAttention
        self.attn = attn_cls(d_model, n_heads, n_kv_heads, chunk_len, n_selected,
                             aux_balance_weight, rope_theta, max_seq_len)
        self.ffn_norm = RMSNorm(d_model, rms_norm_eps)
        self.w13 = nn.Linear(d_model, 2 * ffn_dim)  # fused gate/up SwiGLU projection
        self.w2 = nn.Linear(ffn_dim, d_model)

    def forward(self, hidden: "torch.Tensor", freqs_cis):
        att, aux = self.attn(self.attn_norm(hidden), freqs_cis, return_aux=True)
        h = hidden + att
        gate, up = self.w13(self.ffn_norm(h)).chunk(2, dim=-1)
        h = h + self.w2(F.silu(gate) * up)
        return h, aux