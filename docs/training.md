# Training pipeline — the applied walkthrough

This page is the code-keyed walkthrough of the training loop as it actually
exists: config wiring, the two-phase schedule, the step loop, guardrails, the
loss path, data, and checkpointing. The *why* lives in
[HILS.md](../HILS.md) §4–§5 and the concept pages
([routing](concepts/hils-routing.md),
[hierarchical attention](concepts/hierarchical-attention.md),
[long-context phases](concepts/long-context-phases.md)); runbook operations
live in the [A100 pod runbook](guides/a100-runbook.md). Every anchor below is
machine-checked by `scripts/check_docs.py`.

## 1. The production config

`configs/pretrain_a100_341m.yaml` defines a Chinchilla-class ~341M model
(configured target) for one A100 80GB. The headline structure:

| | Phase A | Phase B |
|---|---|---|
| window | 4096 | 16384 |
| tokens | 7.0B | 1.0B |
| chunks per window (`N`) | 32 | 128 |
| selected fraction (`k/N`) | 8/32 = 25% | 8/128 = 6.25% |
| optimizer steps | 53,407 | 7,630 |

Tokens per optimizer step are held constant at **131,072** (4096 × 8 × 4 =
16384 × 1 × 8 — derived from the config's batch fields). The phase switch
sits at `phase_switch_step = 53407`; the A100 duration and MFU figures in the
config header are **targets until the A100 run receipts say otherwise**
(PASS/DISCLOSED semantics — see the
[eval-scripts reference](references/eval-scripts.md)).

## 2. Wiring: config → model → optimizer

`training/pretrain.py:train` is the entrypoint. Startup:

1. `training/pretrain.py:load_config` parses the YAML and constructs
   `models/transformer.py:HiLSConfig` eagerly — its `__post_init__` fails fast
   on any knob with exactly one implementation (`landmark_init`, `fusion`,
   `selection_scope`, `straight_through_selection`), before the GPU is touched.
2. `models/transformer.py:HiLSAttentionLM` is built from the config;
   `grad_ckpt_every` is set from `training.grad_checkpoint_every` (every 3rd
   block when `grad_checkpoint: true`).
3. If `compile: true` on CUDA, compiled block handles are stored beside the
   raw `nn.ModuleList` as `model._fast_blocks` — checkpoints keep clean keys
   (no `_orig_mod.` prefix), so eval loads stay trivial.
4. `training/pretrain.py:build_optimizer` returns fused AdamW (CUDA only) with
   the house split: weight decay on `p.dim() >= 2`, none on norms/biases.
   Parameters stay fp32 — the master copy; bf16 lives only inside autocast.

## 3. The schedule

- `training/pretrain.py:phase_at` returns `(seq_len, micro_bs, grad_accum)`:
  `(4096, 8, 4)` before the switch, `(16384, 1, 8)` after. Phase B keeps one
  16K window per micro-batch (attention memory scales with window length) and
  restores the token budget via accumulation. The function asserts the
  token budget divides by the Phase-B sequence length.
- `training/pretrain.py:lr_at` is warmup → one cosine across **both** phases,
  with a 500-step re-warm tent added at the switch. The tent is continuous at
  both ends (zero-height at `phase_switch_step` and at switch + 500) and peaks
  back at the base LR mid-tent.
- When the phase changes, the loop rebuilds the DataLoader and, on CUDA,
  re-runs the memory preflight (§6).

## 4. The step loop

Standard teacher-forced loop over packed `shared_data` shards. Per
micro-batch, under `torch.autocast("cuda", bf16)`:

```python
# illustrative — trimmed from training/pretrain.py:train; the loss/backward core is verbatim
with autocast:
    total, aux = model(tokens, targets)   # (CE + mean λ·L_bal, aux-for-logging)
...
(total / accum).backward()
```

Per optimizer step: set LR from `training/pretrain.py:lr_at`, clip global
grad norm to 1.0, step, zero grads, advance `training/pretrain.py:TrainState`
(step, tokens_seen, loss history), log via `utils/logging.py:TrainingLogger`,
and save a checkpoint every `save_interval` steps with `tokens_seen` in the
extra metadata.

`models/transformer.py:HiLSAttentionLM.forward` returns
`(ce + aux, aux)`: the LM term comes from `training/losses.py:chunked_lm_ce`
(§7) and `aux` is the per-layer mean of the λ-weighted balance term produced
by `models/attention.py:HiLSAttention` in each `models/block.py:HiLSBlock`.

## 5. Guardrails

Three independent defenses, each with checkpoint rollback via
`utils/checkpoint.py:CheckpointManager`:

1. **NaN guard** — a non-finite loss zeroes grads and skips the step; a
   streak beyond `nan_guard_max_consecutive` rolls back to the latest
   checkpoint. If loss is still non-finite after the rollback, the loop
   raises instead of looping forever.
2. **Selection watchdog** — `models/router.py:selection_stats` computes the
   max selection share across blocks each step (the surface is
   `HiLSAttention.last_selected`, detached). Max-share above 50% for 500
   consecutive optimizer steps means router collapse; the loop rolls back to
   the last good checkpoint. A step at or below 50% resets the streak.
3. **Memory preflight** — at every phase rebuild,
   `utils/memory.py:estimate_model_memory_gb` estimates the phase's footprint
   and `utils/memory.py:assert_fits_in_available_gpu` refuses to start if it
   doesn't fit the visible GPU.

Rollback is single-shot per checkpoint: rolling back to the same step twice
raises, so a persistently broken run stops loudly rather than silently
thrashing.

## 6. Memory stack

- **bf16 autocast** on CUDA; fp32 master weights in the optimizer (§2).
- **Gradient checkpointing** every 3rd block (`use_reentrant=False`) in
  `models/transformer.py:HiLSAttentionLM.forward`.
- **The loss is the memory trick**: the full `(B, T, V)` logits tensor is
  never materialized — see §7.
- The sparse path itself bounds attention memory: only `k` chunks of keys and
  values per query chunk enter `models/attention.py:hils_attention_core`, one
  SDPA call per selected slot so the per-chunk softmax is preserved exactly.
  `models/attention.py:eager_hils_attention_core` is the O(T²) ground-truth
  reference used by the equivalence tests — never for training.
- VRAM budget derivation per config shape: [config reference](references/config.md).

## 7. The loss path

`training/losses.py:chunked_lm_ce` computes LM cross-entropy one vocab chunk
at a time: per-chunk fp32 logsumexp, target logit gathered in-chunk, chunk
logsumexps combined into the global denominator. The autograd function
behind it saves each chunk's logits so backward derives the softmax from them
— the head GEMM runs once per step instead of being recomputed. The eager
reference is plain `F.cross_entropy` over full logits (`tests/test_loss.py`
holds the parity test).

The router aux term: `models/router.py:balance_loss` scores the Herfindahl
concentration of selection counts; each attention block returns its
λ-weighted term, and `models/transformer.py:HiLSAttentionLM.forward` averages
across layers into `aux`. The λ itself is `aux_balance_weight = 0.01`.

## 8. Data

- Packing: `data/prepare_data.py:main` builds the GPT-2-tokenized packed
  shards (EOS/PAD id 50256, house parity with the workspace loader contract).
- The loop consumes them through the workspace `shared_data` loader
  (`build_training_data`) with per-phase `seq_len` / `batch_size`, a 5%
  validation split, and a fixed shuffle seed — the loader call is the only
  workspace coupling in the training path.
- Corpus composition and packing details: [HILS.md](../HILS.md) §6.

## 9. Checkpointing and resume

- `utils/checkpoint.py:CheckpointManager.save` writes model + optimizer +
  metadata (`step`, `tokens_seen`) on the interval; `latest_step` picks the
  newest at startup, and `train` resumes both counters from metadata before
  the first batch.
- Known limitation (also recorded in `docs/quality-review.md`): the manager
  persists neither RNG state nor the data cursor, so resume replays data from
  the shard start for the current phase — determinism across resume is not a
  claim the checkpoint makes.
- Sampling and sparse decode from a checkpoint: [API reference](references/api.md)
  and [long-context phases](concepts/long-context-phases.md).

## 10. Training-code → doc map

| Module | Documented in |
|---|---|
| `training/pretrain.py` (`train`, `phase_at`, `lr_at`, `build_optimizer`) | this page §2–§5 |
| `training/losses.py` (`chunked_lm_ce`) | this page §7; theory in [HILS.md](../HILS.md) §5 |
| `models/transformer.py` (`HiLSAttentionLM`, `HiLSConfig`) | [API reference](references/api.md); config fields in [config reference](references/config.md) |
| `models/attention.py`, `models/block.py` | [hierarchical attention](concepts/hierarchical-attention.md) |
| `models/router.py`, `models/landmarks.py`, `models/chunking.py` | [HiLS routing](concepts/hils-routing.md) |
| `utils/checkpoint.py`, `utils/logging.py`, `utils/memory.py` | this page §5, §9 |
| `data/prepare_data.py` | this page §8 |
| `inference/generate.py`, `inference/evaluate.py` | [long-context phases](concepts/long-context-phases.md); scripts in [eval-scripts reference](references/eval-scripts.md) |
