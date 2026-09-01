"""Landmark projector and chunk-query pooling (HiLS Phase 1.2).

ℓ_j = W_ℓ · meanpool(K_j): one linear compression per KV head, identity-init
so the model starts as a pure chunk-key meanpool and the LM loss shapes W_ℓ.
q̄_i = meanpool(Q_i) is the router's chunk-query estimate. Both are unfold-free
reshapes: chunks are contiguous tiles, so meanpool is a view + mean.
"""
import torch
from torch import Tensor, nn

from models.chunking import chunk_index_bounds


class LandmarkProjector(nn.Module):
    """Per-layer, per-KV-head linear map (head_dim × head_dim), identity init."""

    def __init__(self, n_kv_heads: int, head_dim: int):
        super().__init__()
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.proj = nn.Linear(head_dim, head_dim, bias=False)
        nn.init.eye_(self.proj.weight)

    def forward(self, k: Tensor, chunk_len: int) -> Tensor:
        """k: (B, n_kv, T, d_head) → landmarks: (B, n_kv, N, d_head)."""
        B, KV, T, D = k.shape
        _, n_chunks = chunk_index_bounds(T, chunk_len)
        pooled = k.view(B, KV, n_chunks, chunk_len, D).mean(dim=3)
        return self.proj(pooled)


def pooled_query(q: Tensor, chunk_len: int) -> Tensor:
    """Chunk-query estimate q̄_i = meanpool(Q_i): (B, H, T, d_head) → (B, H, N, d_head)."""
    B, H, T, D = q.shape
    _, n_chunks = chunk_index_bounds(T, chunk_len)
    return q.view(B, H, n_chunks, chunk_len, D).mean(dim=3)