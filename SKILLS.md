# SKILLS.md — HiLS-Attention-Lite

> Companion to `AGENTS.md` (architecture + rules). This file holds day-to-day
> developer workflows. Workflows marked *(Phase N)* arrive with that phase of
> [`../../llm-research/EXECUTION-PLAN-hils-attention-lite.md`](../../llm-research/EXECUTION-PLAN-hils-attention-lite.md).

## Skill 1: Run the CPU-friendly test suite

```bash
cd LLM/HiLS-Attention-Lite
python3 -m pytest -m "not gpu and not slow"
```

Covers chunking semantics, landmark identity-init, causal top-k selection
invariants, sparse ≡ eager attention equivalence, model wiring, sampler
invariants (decode ≡ teacher-forced in fp64), chunked-CE ≡ eager CE, resume
bit-equality, and the doc-anchor gate. Must pass before any change.

## Skill 2: Two-phase pretrain *(Phase 4)*

```bash
bash scripts/launch_a100.sh
```

One config (`configs/pretrain_a100_341m.yaml`), one schedule: 53,407 steps @
seq 4096 (7.0B tokens) then 7,630 steps @ seq 16384 (1.0B tokens) with a
500-step re-warm at the switch.

## Skill 3: Long-context throughput + KV-fraction eval *(Phase 5)*

```bash
python scripts/longctx_eval.py --baseline llama3-lite-ckpt --ours hils-lite-ckpt --ctx 16384
```

Headline gates: throughput ≥ 1.8× (claim ~2×), KV-access fraction ≤ 0.08
(claim ~6%). Reports measured numbers with explicit PASS/DISCLOSED markers.

## Skill 4: Extrapolation retrieval eval *(Phase 5)*

```bash
python scripts/retrieval_eval.py --ours hils-lite-ckpt --lengths 16384 32768 65536
```

Gate: ≥ 85% retrieval @ 64K (4× the training length), else disclose.

## Skill 5: Quality-parity eval vs LLaMA-3-Lite *(Phase 5)*

```bash
python scripts/loss_parity_eval.py --shard fineweb-edu-heldout
```

Gate: ΔNLL ≤ +5% vs the 515M full-attention baseline (budget delta
disclosed); a larger gap is reported prominently, never hidden.

## Skill 6: A100 microbench + MFU *(Phase 4)*

```bash
python scripts/microbench_a100.py --phase A   # gate: < 15 GB
python scripts/microbench_a100.py --phase B   # gate: < 20 GB
python scripts/step_time_a100.py              # gate: MFU ≥ 33%
```