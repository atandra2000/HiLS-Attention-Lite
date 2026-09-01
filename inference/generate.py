"""Sparse decode (Phase 3): per-layer KV + landmark cache, chunk-synchronous
decode steps, KV access counter.

One decode step consumes one full chunk of C tokens (the in-flight chunk is
complete when it is attended — the same block-synchronous pattern as
DiffusionGemma's canvas decode). Selection reuses the training path verbatim:
models/landmarks.py:pooled_query for the chunk-mass q̄_i, models/router.py:
retrieval_scores / select_chunks / fusion_weights for causal top-k and fusion,
models/attention.py:hils_attention_core's per-chunk factorization for
attention. No separate decode math: at every layer the chunk step's state is
identical to a teacher-forced forward over the same tokens (fp64-exact; the
§7.6 load-bearing gate).

Causality is selection-level (j ≤ i). Because a chunk step attends its own
chunk only once it is complete, no attention mask is needed anywhere in
decode; the in-flight chunk's landmark is finalized in the same step that
consumes it, so the running-mean buffer degenerates to a single projection.
"""
import torch
import torch.nn.functional as F
from torch import Tensor

from models.attention import apply_rope, hils_attention_core
from models.landmarks import pooled_query
from models.router import fusion_weights, retrieval_scores, select_chunks
from models.transformer import HiLSAttentionLM


class HiLSCache:
    """Per-layer K/V cache + landmark cache + KV access counter.

    K/V are chunk-major (L, B, KV, N_max, C, D) so per-step gathers use the
    same advanced indexing as training. Landmark slot j is frozen when chunk j
    is consumed; the counter reads gathered K/V keys (n_sel·C per layer-step)
    plus the landmark table (N per layer-step)."""

    def __init__(self, model: HiLSAttentionLM, batch: int, max_seq: int):
        cfg = model.cfg
        C = cfg.chunk_len
        if max_seq % C != 0:
            raise ValueError(f"max_seq {max_seq} not divisible by chunk_len {C}")
        p = next(model.parameters())
        dev, dt = p.device, p.dtype
        attn0 = model.blocks[0].attn
        self.chunk_len = C
        self.max_chunks = max_seq // C
        self.n_layers = cfg.n_layers
        self.batch = batch
        self.seq_len = 0
        self.steps = 0
        self.kv_keys_read = 0    # key-tensor units: n_sel·C per layer-step
        self.landmark_reads = 0  # N per layer-step
        L, B, KV, D = self.n_layers, batch, attn0.n_kv_heads, attn0.head_dim
        self.k_cache = torch.zeros(L, B, KV, self.max_chunks, C, D, device=dev, dtype=dt)
        self.v_cache = torch.zeros_like(self.k_cache)
        self.landmarks = torch.zeros(L, B, KV, self.max_chunks, D, device=dev, dtype=dt)


def new_cache(model: HiLSAttentionLM, batch: int, max_seq: int) -> HiLSCache:
    """Empty cache sized for max_seq tokens (max_seq % chunk_len == 0)."""
    return HiLSCache(model, batch, max_seq)


@torch.no_grad()
def step_decode(model: HiLSAttentionLM, cache: HiLSCache, tokens: Tensor,
                position: int) -> Tensor:
    """Decode one chunk: tokens (B, C) are the C tokens of the next chunk
    (chunk start at `position`, multiple of C, continuing the cache). Returns
    logits (B, C, V) — teacher-forced within the chunk, so callers can sample
    greedily chunk-wise.

    Per layer: rope the chunk, finalize its landmark, score the chunk-mass
    q̄_i vs the N = i+1 cached landmarks → causal top-k (same router as
    training) → gather k·C keys → k per-chunk SDPA calls over the C query
    tokens → score-weighted fusion. Increments the access counter."""
    B, T = tokens.shape
    C = cache.chunk_len
    if T != C:
        raise ValueError(f"step_decode consumes one chunk ({C} tokens), got {T}")
    if position % C != 0:
        raise ValueError(f"position {position} is not a chunk boundary")
    if position != cache.seq_len:
        raise ValueError(f"position {position} does not continue cache at {cache.seq_len}")
    if position + T > cache.max_chunks * C:
        raise ValueError(f"position {position} exceeds cache capacity")
    i = position // C
    N = i + 1
    freqs = model._freqs_cis(position + T, tokens.device)[position:]
    h = model.embed(tokens)
    b_idx = torch.arange(B, device=h.device)
    for l, block in enumerate(model.blocks):
        attn = block.attn
        H, KV, D = attn.n_heads, attn.n_kv_heads, attn.head_dim
        xn = block.attn_norm(h)
        q = attn.q_proj(xn).view(B, T, H, D).transpose(1, 2)   # (B, H, C, D)
        k = attn.k_proj(xn).view(B, T, KV, D).transpose(1, 2)
        v = attn.v_proj(xn).view(B, T, KV, D).transpose(1, 2)
        q, k = apply_rope(q, freqs), apply_rope(k, freqs)

        # finalize this chunk's landmark, then select against all N landmarks
        lm_i = attn.landmarks(k, C)[:, :, 0]  # (B, KV, D)
        cache.k_cache[l, :, :, i] = k
        cache.v_cache[l, :, :, i] = v
        cache.landmarks[l, :, :, i] = lm_i
        n_sel = min(attn.n_selected, N)
        scores = retrieval_scores(pooled_query(q, C), cache.landmarks[l, :, :, :N],
                                  attn.kv_group_map, D)  # (B, H, 1, N)
        selected, fused = select_chunks(scores, n_sel)    # (B, 1, k), (B, H, 1, k)
        g = fusion_weights(fused)
        k_all = cache.k_cache[l].permute(0, 2, 1, 3, 4)   # (B, N_max, KV, C, D) view
        v_all = cache.v_cache[l].permute(0, 2, 1, 3, 4)
        K_sel = k_all[b_idx, selected[:, 0]]              # (B, k, KV, C, D)
        V_sel = v_all[b_idx, selected[:, 0]]

        out = q.new_zeros(B, H, T, D)
        for s in range(n_sel):  # one SDPA call per selected slot — the per-chunk
            # softmax is the operator; identical factorization as training
            o = F.scaled_dot_product_attention(
                q, K_sel[:, s], V_sel[:, s], enable_gqa=True)  # (B, H, C, D)
            out = out + g[..., s].unsqueeze(-1) * o
        h = h + attn.out_proj(out.transpose(1, 2).reshape(B, T, H * D))
        gate, up = block.w13(block.ffn_norm(h)).chunk(2, dim=-1)
        h = h + block.w2(F.silu(gate) * up)

        cache.kv_keys_read += n_sel * C
        cache.landmark_reads += N
    cache.seq_len = position + T
    cache.steps += 1
    return model.head(model.final_norm(h))


@torch.no_grad()
def prefill(model: HiLSAttentionLM, cache: HiLSCache, tokens: Tensor) -> Tensor:
    """Chunk-aligned prompt fill: a thin loop of chunk steps (P % C == 0), so
    prompt and decode share one code path. Returns last-chunk logits (B, C, V)."""
    B, P = tokens.shape
    C = cache.chunk_len
    if P % C != 0:
        raise ValueError(f"prefill length {P} not divisible by chunk_len {C}")
    logits = None
    for s in range(0, P, C):
        logits = step_decode(model, cache, tokens[:, s:s + C], s)
    return logits


def kv_access_fraction(cache: HiLSCache) -> float:
    """Bytes of K/V read per decode step ÷ full-attention cache bytes.

    Per step per layer: 2·n_sel·C·KV·D bytes of selected K/V plus N·KV·D
    landmark bytes, vs 2·T·KV·D for full attention over the same context —
    i.e. in key units (2·kv_keys_read + landmark_reads) / (2·layers·steps·T).
    At 16K with k=8, C=128: 0.0625 + the landmark read (≈ 0.066)."""
    if cache.steps == 0:
        return 0.0
    T = cache.seq_len
    return (2.0 * cache.kv_keys_read + cache.landmark_reads) \
        / (2.0 * cache.n_layers * cache.steps * T)