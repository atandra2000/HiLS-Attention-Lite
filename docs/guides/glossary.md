# Glossary

Notation, acronyms, and config keys. Config keys cross-link to the
[config reference](../references/config.md), which walks the production YAML
field by field.

## Notation

| Symbol | Meaning | Where it lives in code |
|---|---|---|
| `B, H, T, D` | batch, query heads, sequence length, head dim | shapes in `models/attention.py:hils_attention_core` |
| `KV` | GQA key-value groups (4 in the production config) | `models/attention.py:HiLSAttention` |
| `C` | chunk length in tokens (`chunk_len = 128`) | `models/chunking.py:chunk_index_bounds` |
| `N` | number of chunks, `N = T / C` (32 at seq 4096, 128 at seq 16384) | `models/chunking.py:validate_chunkable` |
| `k` | chunks selected per query chunk (`n_selected = 8`: own chunk + top-7) | `models/router.py:select_chunks` |
| `S` | retrieval scores between pooled queries and landmarks | `models/router.py:retrieval_scores` |
| `g` | per-selected-chunk fusion weights (softmax over scores) | `models/router.py:fusion_weights` |
| `ℓ` | landmark vector per chunk (mean-pooled K at init: `W_ℓ = I`) | `models/landmarks.py:LandmarkProjector` |
| `λ` | aux balance-loss weight (`aux_balance_weight = 0.01`) | `models/router.py:balance_loss` |
| `L_bal` | Herfindahl-style load-balance term over selection counts | `models/router.py:balance_loss` |

## Acronyms

| Term | Expansion |
|---|---|
| HiLS | Hierarchical Learned Sparse (chunk attention) — the model's routing mechanism |
| GQA | grouped-query attention — `n_heads` query heads share `n_kv_heads` KV groups |
| RoPE | rotary positional embedding, applied with `rope_theta = 500000` (`models/attention.py:apply_rope`) |
| SDPA | `scaled_dot_product_attention` — the production per-chunk kernel path |
| CE | cross-entropy; the LM loss is vocab-chunked (`training/losses.py:chunked_lm_ce`) |
| MFU | model FLOPs utilization — a step-time target, see the [eval-scripts reference](../references/eval-scripts.md) |
| EOS | end-of-sequence token (GPT-2 id 50256, house parity with the packer) |

## Terms

- **Chunk** — a contiguous block of `C = 128` tokens; the granularity at which
  attention is routed. `models/chunking.py:causal_chunk_candidates` enumerates
  the causally admissible candidates.
- **Landmark** — the compressed per-chunk representation queries are scored
  against; pooled from keys (`models/landmarks.py:pooled_query` pooled the
  query side). Identity-initialized so the router is meaningful at step 0.
- **Score fusion** — selected chunks' per-chunk softmax outputs are combined by
  the same scores that selected them; `fusion = score_softmax` is the only
  implementation in v1 (`models/transformer.py:HiLSConfig.__post_init__`
  fails fast on anything else).
- **Native trainability** — selection is top-k over differentiable scores, so
  gradients flow through the fusion weights `g` (not through the hard top-k
  indices); see [HiLS routing concepts](../concepts/hils-routing.md).
- **Two-phase plan** — Phase A trains 7.0B tokens at seq 4096, Phase B 1.0B at
  seq 16384, tokens/optimizer-step held constant; see
  [long-context phases](../concepts/long-context-phases.md) and
  [training.md](../training.md).
- **Chunk-synchronous decode** — sparse decoding that keeps finalized chunks'
  cache frozen; `inference/generate.py:HiLSCache` and
  `inference/generate.py:step_decode` implement it, and
  `inference/generate.py:kv_access_fraction` counts the KV actually touched.

## Config keys (quick table)

Full field-by-field walkthrough: [config reference](../references/config.md).

| Key | Value in production | One-line meaning |
|---|---|---|
| `chunk_len` | 128 | tokens per chunk (`C`) |
| `n_selected` | 8 | chunks attended per query chunk (`k`) |
| `landmark_init` | `identity` | only init implemented in v1 (`W_ℓ = I`) |
| `fusion` | `score_softmax` | fuse selected outputs by score softmax |
| `selection_scope` | `per_query_chunk` | each query chunk selects independently |
| `aux_balance_weight` | 0.01 | λ scaling `L_bal` inside the total loss |
| `attn_impl` | `sdpa` | production path (`eager` = O(T²) reference) |
| `phase_switch_step` | 53407 | optimizer step where seq 4096 → 16384 |
| `phase_b_warmup_steps` | 500 | LR re-warm tent width at the switch |
| `grad_checkpoint_every` | 3 | checkpoint every 3rd block in the loop |
| `max_seq_len` | 16384 | Phase B ceiling; forward extrapolates beyond it |
