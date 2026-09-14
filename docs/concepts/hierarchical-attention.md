# Concept: Hierarchical attention — per-chunk softmax and score fusion

Attention in this repo is *hierarchical*: for a query chunk i and each
selected chunk j, the softmax runs over that chunk's C keys **independently**,
and the per-chunk outputs fuse by the router's retrieval scores. This page is
the operator; the selection above it is [HiLS routing](hils-routing.md).

## The operator

For query chunk i with selected set S(i) (|S(i)| = k):

```
o_ij = SDPA(Q_i, K_j, V_j)          # one softmax over chunk j's C keys — per chunk
g_ij = softmax_j(s_ij)              # per-head softmax over the k selected scores
o_i  = Σ_{j ∈ S(i)} g_ij · o_ij     # score-weighted fusion
```

The per-chunk softmax is **the operator**, not an approximation of full
attention: a single SDPA over the concatenated k·C keys would compute a joint
softmax and change the math. The production core,
`models/attention.py:hils_attention_core`, gathers the selected KV to
`(B, N, k, KV, C, D)` and issues k batched SDPA calls — one per selected
slot, each over `(B·N, ·)` — then fuses by `g`. `k` is a static Python int,
so the slot loop is unrollable by `torch.compile`.

Causality is a **selection mask only** (j ≤ i); within a retrieved chunk all
C keys are visible, so no token-level mask exists anywhere in the operator.
A chunk step attends its own chunk only once it is complete — the invariant
that makes decode ≡ teacher-forced (see [long-context phases](long-context-phases.md)).

## Two paths, one math

- `models/attention.py:HiLSAttention` — the production module
  (`attn_impl="sdpa"`): landmark scoring → causal top-k → gather → per-chunk
  SDPA → fusion, batched.
- `models/attention.py:EagerHiLSAttention` — the reference path
  (`attn_impl="eager"`): the identical math via
  `models/attention.py:eager_hils_attention_core`, an O(T²) per-query-chunk
  loop over a full score matrix with per-chunk softmax reconstructed by
  masking. Ground truth only — `test_hils_matches_eager_reference` (fp64,
  atol 1e-5) guards every change to either path, and `k = N` is part of that
  gate: all-chunk selection stays the per-chunk-softmax operator, never a
  joint softmax (it is not plain dense attention).

`models/attention.py:HiLSAttention.forward` is where the pieces meet: rope →
landmarks → scores → `select_chunks` → fusion → core; it also stashes
`last_selected` for the watchdog and returns the λ-weighted balance term.

## RoPE (absolute, unstretched)

`models/attention.py:precompute_freqs_cis` builds the Llama-style complex
rotation table and `models/attention.py:apply_rope` rotates Q/K. Two facts
matter for long context:

- Positions are **absolute** and the table is computed for the actual
  sequence length on every forward (`models/transformer.py:HiLSAttentionLM`
  caches nothing) — so forward beyond `max_seq_len` extrapolates with no
  stretching, no YaRN, no scaling tricks (θ = 500K is the only lever).
- Trig runs in ≥ fp32 and casts back — bf16 inputs never see bf16 trig.

## The block around the attention

`models/block.py:HiLSBlock` is the house decoder block: pre-norm attention
sublayer + pre-norm SwiGLU sublayer, both residual. The norm is
`models/block.py:RMSNorm` (fused `F.rms_norm`, learned scale, no bias). The
backbone carries **none of the novelty** — the house block exists so the
chunked-CE / checkpointing / eval machinery stays drop-in and the sparse
attention is the only variable (DESIGN §1).

## The loss path (chunked CE, bounded fp32 peak)

`models/transformer.py:HiLSAttentionLM.forward` with targets computes the LM
loss through `training/losses.py:chunked_lm_ce`: vocab-chunked fp32
logsumexp + target logits, one vocab slice at a time, with a custom autograd
Function that keeps each chunk's logits for backward instead of recomputing
the head GEMM. The eager reference for that two-path pair is plain
`F.cross_entropy` over full logits (`tests/test_loss.py`, fp32 atol 1e-6).
At micro_bs 8 / seq 4096 the naive path would cost ~6.6 GB of fp32 logits;
the chunked path retains bf16 chunk logits (~3.3 GB) plus one transient fp32
slice (~1.1 GB) and buys back the head-GEMM recompute (utils/memory.py
encodes the budget).

## k = N is a test case, not a fallback

Setting `n_selected ≥ N` selects every admissible chunk: the operator stays
per-chunk SDPA with fusion weights ≈ uniform over all chunks — hierarchical
all-chunk selection, not a dense fallback. `test_hils_k_equals_n_selects_all`
guards the degenerate path, and the eval harness uses the same configuration
as its in-repo "all-chunk" throughput baseline (see
[inference/evaluate.py](../../inference/evaluate.py)).