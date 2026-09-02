# Concept: HiLS routing — landmarks, causal top-k, native trainability

The router is the repo's claimed cell: a learned, causal, chunk-level
selection policy trained by the LM loss alone — no retrieval loss, no
auxiliary head, no reward model. This page follows one query chunk through
the decision; the operator it feeds is [hierarchical attention](hierarchical-attention.md).

## Chunking (the granularity decision)

The sequence tiles into `C = 128`-token chunks (`N = T/C`: 32 chunks at the
4096 pretrain length, 128 at 16K). Tiling is exact, not ragged:
`models/chunking.py:validate_chunkable` rejects any `(seq_len, chunk_len)`
that does not divide — a ragged tail would desynchronize the router, the
landmark meanpool, and the KV gather, all of which index by `N = T/C`
(`models/chunking.py:chunk_index_bounds` is the shared bound helper).

Selection granularity is the **query chunk, not the query token**: all C
queries in chunk i share one selected set. That is what makes the gather
static-shaped (`k` is a Python int), SDPA-batched, and compile-friendly;
per-query selection is a documented ablation, not v1.

## Landmarks (the compressed side)

Each chunk j carries one landmark per KV head — a learned compression of its
keys:

```
ℓ_j = W_ℓ · meanpool_{t ∈ chunk j}(k_t)        # per KV head: (d_head) = (64×64)·(d_head)
```

`models/landmarks.py:LandmarkProjector` is that per-layer, per-KV-head linear
map, **identity-initialized** so the model starts as a pure chunk-key
meanpool and the LM loss shapes W_ℓ from there (~0.39M params total). The
router's query side is the same meanpool on Q:
`models/landmarks.py:pooled_query` produces q̄_i = meanpool(Q_i). Both are
unfold-free reshapes — chunks are contiguous tiles, so meanpool is a view +
mean.

## Scoring and causal top-k

`s_ijh = ⟨q̄_ih, ℓ_j·kv(h)⟩ / √d_head` — `models/router.py:retrieval_scores`
maps each query chunk against every landmark through the GQA group map,
yielding `(B, H, N, N)` scores. Selection then happens in
`models/router.py:select_chunks`:

- **Head-shared, eager.** Chunks are ranked by the mean score over query
  heads; the discrete top-k indices are computed in eager Python/tensor land
  because data-dependent gathers break `torch.compile` (house precedent).
- **Causal by construction.** Chunk i may select j ≤ i only. Ties break
  toward the higher chunk index (flip + stable argsort, no epsilon), so
  selection reproduces the score-free reference sets of
  `models/chunking.py:causal_chunk_candidates` exactly — `tests/test_router.py`
  checks set equality.
- **Own chunk always.** Slot 0 is the chunk's own keys (local continuity is
  never risked to the router); the remaining k−1 slots are the
  best-scoring admissible chunks.
- **Right-aligned padding.** Rows with fewer admissible chunks pad by
  repeating the own index with −inf fused scores — zero fusion weight, still
  causal. This is what lets decode score one query chunk against all N
  cached landmarks with the same function as training.

## Native trainability (the load-bearing fact)

Selection (the top-k indices) is discrete and non-differentiable. But the
**scores of selected chunks stay differentiable** and re-enter the forward
pass as per-head fusion weights — `models/router.py:fusion_weights` takes a
softmax over the k selected scores, and the attention output is
Σ_j g_ij · o_ij. The LM loss therefore reaches q̄ and W_ℓ through g — the
router is *trained by the LM loss through the fusion weights*, which is the
paper's native-trainability claim reproduced end-to-end
(`test_retrieval_score_gradient_flow` guards gradients to W_ℓ and q̄).
Unselected chunks get signal only via the balance loss below.

## Load balancing + the watchdog (collapse defense)

The known failure mode is routing collapse: every query chunk reads the same
few chunks. The aux term is the Herfindahl index of per-chunk selection mass,
`models/router.py:balance_loss`: `L_bal = N · Σ_j p_j²` (= 1 at uniform,
unbounded at collapse), λ = 0.01, per layer inside
`models/attention.py:HiLSAttention` and averaged across layers. The mass is
the *soft* per-chunk fusion weight (hard counts would block gradient).

Health is observable, not just optimized: `models/router.py:selection_stats`
reports entropy, max share, and the per-layer histogram; the training loop
(`training/pretrain.py:train`) trips a watchdog rollback when max share
exceeds 50% for 500 consecutive steps. The aux term touches only the score
path — `aux_balance_weight: 0.0` is a clean ablation flag.

## What the router does NOT do

- No per-token selection (documented ablation, not v1).
- No learned temperature or top-k schedule; k=8 is static.
- No fixed pattern fallback: `k = N` is a test case (all-chunk selection),
  never a "dense mode" (AGENTS.md rule 6).