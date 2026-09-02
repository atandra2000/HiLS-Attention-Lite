# Guide: debugging playbook

Symptom-first recipes for the failure modes this architecture actually has.
Every row names the owning test or script; run it before and after any fix.
The two tests that guard the whole correctness story — rerun them after ANY
change to `models/` or `training/`:

```bash
python3 -m pytest tests/test_attention.py -q   # sparse ≡ eager (fp64, atol 1e-5)
python3 -m pytest tests/test_sampler.py -q     # decode ≡ teacher-forced
```

## Router collapse (max-share climbing)

**Symptom:** `[watchdog] max-share > 50%` in the log, or
`training/pretrain.py:train` rolls back to the last checkpoint after 500
consecutive steps.

**Checks, in order:**

1. Read the per-layer histogram — `models/router.py:selection_stats` prints
   entropy, max share, and per-chunk counts; one hot layer vs all layers
   distinguishes a local layer from global collapse.
2. Confirm the aux term is on: `aux_balance_weight: 0.01` in the config —
   `models/router.py:balance_loss` is the Herfindahl penalty; 0.0 disables it.
3. Confirm the fusion path is intact: if
   `models/router.py:fusion_weights` ever sees masked rows with −inf across
   all slots, weights degenerate — `test_fusion_weights_sum_to_one` guards.
4. If collapse appears right after the phase switch, suspect the re-warm:
   `training/pretrain.py:lr_at` adds a 500-step tent at
   `phase_switch_step`; check `training/pretrain.py:phase_at` is switching
   shapes where you think it is.

The watchdog is a *metric + rollback*, not a loss change. If it fires
repeatedly at the same step, the run is hitting a data/phase pathology —
inspect the shard windows around that step before touching λ.

## Non-finite loss

**Symptom:** `[nan-guard] non-finite loss` — the guard zeroes grads and rolls
back after `nan_guard_max_consecutive` hits.

**Checks:** grad clip (1.0) is applied by the loop; bf16 autocast only on
CUDA (`training/pretrain.py:train` uses `contextlib.nullcontext` on CPU);
the chunked-CE path normalizes in fp32
(`training/losses.py:chunked_lm_ce`), so a NaN usually comes from the
attention scale or LR — check `training/pretrain.py:lr_at` warmup
(`lr · (step+1)/warmup`, first optimizer step trains). A second rollback at
the same step raises — take the checkpoint inspection route
(`utils/checkpoint.py:CheckpointManager.latest_step` + meta json) rather
than disabling the guard.

## Phase A→B switch spike at step 53,407

**Symptom:** loss jumps when seq goes 4096 → 16384.

**Expected:** a small, transient bump; the 500-step re-warm absorbs it. Verify
with `training/pretrain.py:phase_at` (shape pair flips at
`phase_switch_step`) and watch `utils/logging.py:TrainingLogger` output —
tokens/step must stay 131,072 (8·4096·4 = 1·16384·8). A *growing* spike is
not the switch — treat as NaN-guard territory.

## KV-access fraction off target

**Symptom:** `scripts/longctx_eval.py` prints a fraction well above 0.08.

**Check the arithmetic first:** fraction ≈ k·C/T + N/(2T·k-ish) — at 16K with
k=8, C=128 it is 0.0625 + the landmark read ≈ 0.066; at 4K it is ~0.25 by
construction (8/32 chunks). The B2 gate is a 16K-context gate. If the
*counter* itself disagrees with k·C per step, `inference/generate.py:step_decode`
and `inference/generate.py:kv_access_fraction` are the two code points;
`test_kv_access_fraction_counter` pins them exactly (k·C keys + N landmarks
per layer-step).

## Decode disagrees with teacher-forced

**Symptom:** `test_sparse_decode_matches_teacher_forced` fails.

**Cause almost always:** landmark finalization order in
`inference/generate.py:step_decode` (a chunk's landmark must be finalized in
the same step that consumes it) or stale cache slots (chunk-major K/V layout
`(L, B, KV, N_max, C, D)`). The counter-instrument zeroing pattern in the
test shows how to isolate a single step. Never "fix" by recomputing
selection differently in decode — decode must reuse the training router
verbatim (`inference/generate.py:prefill` is a loop of the same step).

## Throughput below the MFU gate

**Symptom:** `scripts/step_time_a100.py` prints MFU < 33%.

**First suspect:** the per-chunk gather, not the router — the ~8% routing/
gather/aux line-item is the modeled overhead. Profile the advanced-indexing
gather in `models/attention.py:hils_attention_core` before touching the
router (`models/router.py:select_chunks` is deliberately eager). Confirm
`--compile` was passed (production compiles blocks, selection stays eager).

## VRAM above the budget gate

**Symptom:** `scripts/microbench_a100.py` prints peak ≥ 15/20 GB.

**Checks:** warmup step before measuring (the script already does — allocator
state must include AdamW moments); `utils/memory.py:estimate_model_memory_gb`
prediction vs measured peak (a large gap means the estimator's CE-chain
assumption drifted); `utils/memory.py:assert_fits_in_available_gpu` should
have failed the run early otherwise. The full-vocab CE trap (~6.6 GB fp32
logits) is exactly what `training/losses.py:chunked_lm_ce` avoids — make sure
no eval path materializes full logits (the evaluator's positional readout
exists for this reason).

## Config sanity

`models/transformer.py:HiLSConfig` validates eagerly —
`training/pretrain.py:load_config` fails fast on unknown keys. The knobs that
change selection semantics (`chunk_len`, `n_selected`,
`aux_balance_weight`, `landmark_init`, `fusion`) live in
[references/config.md](../references/config.md) with their invariants; after
any change there, rerun `tests/test_router.py` first (causality + own-chunk
invariants are load-bearing for decode).