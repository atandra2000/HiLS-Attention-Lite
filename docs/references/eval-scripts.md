# Reference: eval + boundary scripts

Every headline number comes from one of the scripts below, all built on the
same harness (`inference/evaluate.py:LongContextEvaluator`) and the same
honesty contract (AGENTS.md rule 4): a measured number is printed with an
explicit `[PASS …]` or `[DISCLOSED …]` marker — never a bare claim; a missing
checkpoint, baseline, tokenizer, or shard is disclosed with the reason, never
silently dropped; a measured miss is disclosed next to its gate. **The scripts
exit 0 in every reported case** — a disclosed miss is a result, not a crash.

Every script has a CPU self-check form (`--tiny`) and an A100 headline form;
run the self-check before spending pod time on the headline.

## The gates at a glance

| gate | script | measures | criterion | gate evaluated at |
|---|---|---|---|---|
| B1 | `scripts/longctx_eval.py` | decode throughput vs LLaMA-3-Lite | ≥ 1.8× (claim ~2×) | largest `--ctx` |
| B2 | `scripts/longctx_eval.py` | effective KV-access fraction | ≤ 0.08 (claim ~6%) | largest `--ctx` (16K) |
| B3 | `scripts/retrieval_eval.py` | NIAH retrieval accuracy | ≥ 85%, else disclose | largest `--lengths` (64K) |
| B4 | `scripts/loss_parity_eval.py` | held-out ΔNLL vs LLaMA-3-Lite | ≤ +5%, else disclose | `--seq` (4096) |
| B5 | `scripts/microbench_a100.py` | peak VRAM per phase | A < 15 GB, B < 20 GB | the phase run |
| B6 | `scripts/step_time_a100.py` | optimizer-step MFU | ≥ 33% (A100 80GB BF16) | Phase A shape |

## `scripts/longctx_eval.py` — B1 + B2

Per requested context length, times chunk-synchronous decode
(`inference/generate.py:prefill` + `inference/generate.py:step_decode`) for
our checkpoint; optionally re-times a LLaMA-3-Lite checkpoint (full forward
per token, no KV cache — its cost is the B1 denominator). The warmup chunk is
untimed and the KV-access instruments
(`inference/generate.py:kv_access_fraction`) are zeroed before the timed
window, so the fraction is steady-state, not warmup-contaminated. The same
weights are re-timed with `n_selected → N` as the built-in **all-chunk**
diagnostic — what learned top-k saves with identical weights, no external
model needed.

| flag | default | meaning |
|---|---|---|
| `--ours-ckpt` | none | our checkpoint dir (three-file sets; `utils/checkpoint.py:CheckpointManager`) |
| `--baseline-ckpt` | none | LLaMA-3-Lite `.pt` checkpoint; omitted ⇒ B1 is disclosed, B2 still prints |
| `--ctx` | config-derived | one or more context lengths |
| `--steps` / `--warmup` | 3 / 1 | timed / untimed decode chunks per length |
| `--tiny` | off | 2-layer random-init config, no checkpoints (self-check) |
| `--device` | auto | `cuda` when available |
| `--baseline-repo` | `../LLaMA-3-Lite` | baseline repo dir (config/tokenizer source for the baseline checkpoint) |

```bash
python scripts/longctx_eval.py --tiny --ctx 512 1024                     # CPU self-check
python scripts/longctx_eval.py --ours-ckpt checkpoints/pretrain_a100 \
    --baseline-ckpt ../LLaMA-3-Lite/checkpoints/model_final.pt --ctx 16384   # A100
```

## `scripts/retrieval_eval.py` — B3 (+ NLL-vs-length)

Needle-in-a-haystack through `inference/evaluate.py:Needle` rows: template
needles planted in a seeded haystack at alternating depths, the key repeated
at the row end, every value token argmax-checked (multi-token values are
teacher-forced by construction). Lengths beyond the 16K training window
extrapolate with **no RoPE stretching** — the rotation table is computed for
the actual length every forward. NLL-vs-length rides the same run when a
token shard is available (`--shard` / `--nll-tokens`). `--tiny` retrieval is
plumbing-only: a random-init model's number is meaningless and disclosed as
such.

| flag | default | meaning |
|---|---|---|
| `--ours-ckpt` | none | our checkpoint dir |
| `--lengths` | 16384 32768 65536 | evaluation lengths (B3 reads the largest) |
| `--needles` | harness default | retrieval rows per length |
| `--batch` | 8 | retrieval rows per forward |
| `--shard` / `--nll-tokens` | auto / 131072 | held-out uint32 shard + tail length for NLL-vs-length |
| `--steps` / `--warmup` / `--seed` / `--device` | 3 / 1 / 0 / auto | decode-loop and seeding controls |

```bash
python scripts/retrieval_eval.py --tiny                                  # CPU self-check
python scripts/retrieval_eval.py --ours-ckpt checkpoints/pretrain_a100 \
    --lengths 16384 32768 65536 --needles 20                             # A100
```

## `scripts/loss_parity_eval.py` — B4

Scores the same held-out text through each model's **own** loss path: ours is
total − aux from `models/transformer.py:HiLSAttentionLM.forward` (the
chunked-CE objective, `training/losses.py:chunked_lm_ce`, without the aux
term); the baseline is LLaMA-3-Lite scored with plain full-vocab
cross-entropy. Tokenizers differ (GPT-2 ~50k vs ~128k), so the honest
same-text comparison tokenizes per model — `--texts-file` (one UTF-8 file)
or pre-tokenized `--ours-shard` / `--baseline-shard` windows. The ~341M vs
~515M parameter-budget delta is disclosed in every run: parity is anchored,
not matched-budget.

| flag | default | meaning |
|---|---|---|
| `--ours-ckpt` | none | our checkpoint dir (optional — tiny form runs without it) |
| `--baseline-ckpt` | none | LLaMA-3-Lite `.pt` checkpoint |
| `--baseline-repo` | `../LLaMA-3-Lite` | baseline repo dir (config/tokenizer source) |
| `--baseline-tokenizer` | repo default | override tokenizer for the baseline model |
| `--texts-file` | none | UTF-8 held-out text; tokenized separately per model |
| `--ours-shard` / `--baseline-shard` | none | pre-tokenized uint32 held-out windows |
| `--seq` / `--windows` | 4096 / 8 | window length and count |
| `--tiny` | off | tiny random-init models; verdict is plumbing-only |

```bash
python scripts/loss_parity_eval.py --tiny                                # CPU plumbing check
python scripts/loss_parity_eval.py --ours-ckpt checkpoints/pretrain_a100 \
    --texts-file heldout.txt \
    --baseline-ckpt ../LLaMA-3-Lite/checkpoints/model_final.pt           # A100
```

## `scripts/microbench_a100.py` — B5 (peak VRAM)

One steady-state optimizer step at the phase's shape (fused AdamW, BF16
autocast, production attention path); reports the CUDA allocator peak. A
warmup step runs first so AdamW's lazily allocated moments are inside the
measurement; without it the gate would pass vacuously. Shapes come from
`training/pretrain.py:phase_at` — never hardcoded — so the benchmark cannot
drift from the loop, and the reading is cross-checked against
`utils/memory.py:estimate_model_memory_gb`. `torch.compile` stays off: eager
is the conservative allocator upper bound. On a CUDA-less machine the gate is
unmeasurable, not violated — the script prints why and exits 0.

| flag | default | meaning |
|---|---|---|
| `--phase` | required | `A` (micro_bs 8, seq 4096; gate < 15 GB) or `B` (micro_bs 1, seq 16384; gate < 20 GB) |
| `--config` | production YAML | shapes + model read from here |

```bash
python scripts/microbench_a100.py --phase A
python scripts/microbench_a100.py --phase B
```

## `scripts/step_time_a100.py` — B6 (MFU)

Times whole optimizer steps (grad-accum micro-batches + clip + fused AdamW)
at the Phase A shape and converts to MFU. FLOPs convention: 6·N per token
over unique (untied) parameters, ×(1 + 1/(3·E)) for every-Eth-block gradient
checkpointing; attention score FLOPs are excluded — they are sublinear under
chunk selection, which is the point of the architecture. `--compile` matches
the production loop's block compile (blocks only; selection stays eager).
On CPU the script prints a tokens/sec proxy and exits 0.

| flag | default | meaning |
|---|---|---|
| `--steps` / `--warmup` | 20 / 5 | timed / untimed optimizer steps |
| `--compile` | off | compile blocks, matching production |

```bash
python scripts/step_time_a100.py --compile      # gate: MFU ≥ 33%
```

## `scripts/e2e_gpu_smoke.py` — production-loop smoke

Runs `training/pretrain.py:train` twice over **fixed synthetic batches** (the
same `batches=` / `max_steps=` seam the unit tests use), exercising the real
loop machinery end to end: BF16 autocast, phase shapes, NaN guard, selection
watchdog, VRAM estimator, checkpoint save + resume. Per phase it checks the
loop reaches the step cap, loss decreases, and no guard marker fired; the
Phase A checkpoint is written and resumable. Fixed (repeated) batches are
deliberate — random iid tokens make a loss-decrease assertion vacuous near
the ln(vocab) floor, a memorizable stream makes it a real signal.

| flag | default | meaning |
|---|---|---|
| `--steps-a` / `--steps-b` | 200 / 20 | optimizer steps per phase |
| `--workdir` | temp dir | where the smoke checkpoints land |
| `--tiny` | off | 2-layer config; the whole harness self-checks on CPU in seconds |

```bash
python scripts/e2e_gpu_smoke.py --tiny          # CPU self-check
python scripts/e2e_gpu_smoke.py                 # A100: 200 A-steps + 20 B-steps + resume
```

## `scripts/launch_a100.sh` — the one-command pretrain

`bash scripts/launch_a100.sh` execs `training/pretrain.py` on the production
config; it auto-resumes from the latest **complete** checkpoint in
`checkpoints/pretrain_a100`. Prerequisites (data + the B5/B6 checks) are in
the [A100 runbook](../guides/a100-runbook.md).

## Interpreting output

- `[PASS …]` — the gate's criterion measured-and-met at the stated context.
- `[DISCLOSED …]` — the gate's status is a *result*, not a claim: either a
  measured miss, or a prerequisite (checkpoint/baseline/tokenizer/shard) that
  was not available, printed with the reason. Exit code stays 0 either way.
- Tiny/CPU runs are plumbing checks: their measured numbers are real, but the
  production gates are evaluated only at the production context (B2 is a 16K
  gate; B3 evaluates at the largest requested length).

Symptom-first triage for a failing gate lives in the
[debugging playbook](../guides/debugging-playbook.md); what the numbers feed
back into is the config in [config.md](config.md).