"""Chunked LM cross-entropy: the full (B, T, V) logits tensor is never
materialized (DESIGN §4.0). Vocab-chunked logits live one slice at a time with
fp32 normalization; each chunk's logits are retained for backward via a custom
autograd Function that derives the softmax from them, so the head GEMM runs
once instead of being checkpoint-recomputed. The eager reference for the
two-path pair is plain ``F.cross_entropy`` over full logits.
"""
import torch


class _ChunkTerms(torch.autograd.Function):
    """One vocab chunk's fp32 logsumexp and target logit, saving logits for
    backward. The chunk GEMM runs once; backward computes the softmax from the
    saved logits. The trade: ~2 bytes/elem/chunk retained buys back a full
    head-GEMM forward per step."""

    @staticmethod
    def forward(ctx, hidden, weight, local_targets):
        logits = hidden @ weight.to(hidden.dtype).t()
        logits_f = logits.float()
        lse = torch.logsumexp(logits_f, dim=-1)
        local = local_targets.clamp(0, weight.size(0) - 1).unsqueeze(-1)
        tgt = torch.gather(logits_f, -1, local).squeeze(-1)
        ctx.save_for_backward(hidden, weight, logits, local)
        return lse, tgt

    @staticmethod
    def backward(ctx, grad_lse, grad_tgt):
        hidden, weight, logits, local = ctx.saved_tensors
        p = logits.float().softmax(dim=-1)
        g_logits = p * grad_lse.unsqueeze(-1)
        # d(target_logit)/dlogits = onehot(target); grad_tgt is already signed
        # and zeroed for out-of-chunk targets by the caller's in-chunk mask, so
        # the clamped scatter only ever adds zeros there.
        g_logits.scatter_add_(-1, local, grad_tgt.unsqueeze(-1))
        h_dtype = hidden.dtype
        g_hidden = g_logits.to(h_dtype) @ weight.to(h_dtype)
        B, T, C = g_logits.shape
        g_weight = (g_logits.reshape(-1, C).t() @ hidden.reshape(-1, hidden.size(-1))).to(weight.dtype)
        return g_hidden, g_weight, None


def chunked_lm_ce(hidden, embed_weight, targets, vocab_chunk: int = 8192):
    """LM cross-entropy one vocab chunk at a time; never materializes (B, T, V).

    Chunk logsumexps combine into the global denominator. Eager reference:
    ``F.cross_entropy`` over the full logit matrix (tests/test_loss.py)."""
    V = embed_weight.size(0)
    step = V if vocab_chunk is None else min(int(vocab_chunk), V)

    lse_parts, tgt_parts = [], []
    for c0 in range(0, V, step):
        c1 = min(c0 + step, V)
        lse, tgt = _ChunkTerms.apply(hidden, embed_weight[c0:c1], targets - c0)
        in_chunk = (targets >= c0) & (targets < c1)
        lse_parts.append(lse)
        tgt_parts.append(torch.where(in_chunk, tgt, torch.zeros_like(tgt)))
    lse = torch.logsumexp(torch.stack(lse_parts, dim=-1), dim=-1)
    target_logit = sum(tgt_parts)
    return (lse - target_logit).mean()