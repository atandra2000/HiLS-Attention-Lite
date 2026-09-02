# Concept: long-context phases — 7B @ 4096 then 1B @ 16K

HiLS-Attention-Lite trains in two phases on one schedule, then evaluates
long-context behavior with a single measurement harness. This page is the
map: phase shapes, the decode path that inherits them, and the eval protocol.

## The two-phase plan

| Phase | tokens | seq | chunks N | selected | micro-bs | grad-accum |
|---|---|---|---|---|---|---|
| A | 7.0B | 4096 | 32 | 8/32 = 25% | 8 | 4 |
| B | 1.0B | 16384 | 128 | 8/128 = 6.25% | 1 | 8 |

Tokens per optimizer step stay constant at 131,072:
`training/pretrain.py:phase_at` derives `(seq_len, micro_bs, grad_accum)`
from the config (never hardcoded — `scripts/microbench_a100.py` and
`scripts/step_time_a100.py` consume the same function). The LR schedule is
one cosine across both phases with a 500-step re-warm tent at the switch:
`training/pretrain.py:lr_at`.

The loop itself (`training/pretrain.py:train`) is a standard teacher-forced
loop over `shared_data` shards: fused AdamW with fp32 master weights
(`training/pretrain.py:build_optimizer`), BF16 autocast on GPU, grad-ckpt
every 3rd layer, NaN guard with checkpoint rollback, and the selection
watchdog. State is the small `training/pretrain.py:TrainState` dataclass
(step, tokens_seen, model, optimizer, losses). The config is validated
eagerly by `training/pretrain.py:load_config` (the model section must
construct a `models/transformer.py:HiLSConfig`).

## Sparse decode (Phase 3)

Decode is **chunk-synchronous**: one step consumes one full chunk of C tokens
and attends its own chunk only once complete — the same selection code path
as training, verbatim. State lives in `inference/generate.py:HiLSCache`
(per-layer chunk-major K/V + landmark table + KV access counter), created by
`inference/generate.py:new_cache`. `inference/generate.py:prefill` is just a
loop of chunk steps, so prompt fill and decode share one code path, and
`inference/generate.py:step_decode` finalizes each chunk's landmark in the
step that consumes it.

Because a chunk step's state equals a teacher-forced forward over the same
tokens (fp64-exact at every layer — the §7.6 load-bearing gate,
`test_sparse_decode_matches_teacher_forced`), there is no separate decode
math to drift. The KV-access counter is the v1 headline instrument:
`inference/generate.py:kv_access_fraction` reports selected-K/V + landmark
bytes read per step ÷ full-attention bytes — ≈ k·C/T + the landmark read
(≈ 0.066 at 16K with k=8, C=128). The v1 cache *physically* stores all K/V;
the claim is **effective access** (DESIGN §2.6).

## The evaluation harness (Phase 5)

`inference/evaluate.py:LongContextEvaluator` is the one harness behind all
three headline scripts: decode throughput@ctx, the KV-access fraction,
NLL-vs-length, and needle-in-a-haystack retrieval accuracy
(`inference/evaluate.py:Needle` — id-level (key, value) pairs with seeded
haystacks; `inference/evaluate.py:register_baseline` adds in-repo baselines
such as `"all-chunk"`). Headline gates are printed by the scripts as
PASS/DISCLOSED — never bare claims (AGENTS.md rule 4):

| gate | script | criterion |
|---|---|---|
| B1 decode throughput @ 16K | `scripts/longctx_eval.py` | ≥ 1.8× vs LLaMA-3-Lite (claim ~2×) |
| B2 KV-access fraction @ 16K | `scripts/longctx_eval.py` | ≤ 0.08 (claim ~6%) |
| B3 retrieval @ 64K | `scripts/retrieval_eval.py` | ≥ 85%, else disclose |
| B4 ΔNLL @ 4096 | `scripts/loss_parity_eval.py` | ≤ +5% vs LLaMA-3-Lite |

## Data + checkpoints + logging

- `data/prepare_data.py:main` delegates to the workspace `shared_data`
  pipeline with the GPT-2 tokenizer contract (50,257 vocab, EOS/PAD 50256)
  and pins `LLM_DATA_ROOT` — see the [quickstart](../guides/quickstart.md).
- Checkpoints are three-file sets (weights safetensors + optimizer pt + meta
  json); a step is resumable only when all three exist —
  `utils/checkpoint.py:CheckpointManager` (save/load/latest_step).
- `utils/logging.py:TrainingLogger` aggregates loss over the log interval
  and reports tokens/sec + ppl; optional WandB via `WANDB_PROJECT`.
- `utils/memory.py:estimate_model_memory_gb` encodes the A100 budget table
  (params + fp32 AdamW master + boundary activations + CE chain), and
  `utils/memory.py:assert_fits_in_available_gpu` enforces it before a run
  starts — the microbench script cross-checks the estimator against the
  measured allocator peak.