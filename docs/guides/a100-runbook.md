# Guide: A100 pod runbook

The operational order for the remaining pod work — data → boundary checks →
the 40–48 h pretrain → the headline evals. Commands live in the
[quickstart](quickstart.md) and the [eval-scripts reference](../references/eval-scripts.md);
this page is about sequence, resume semantics, and what to watch. The CPU
gate (`python3 -m pytest -m "not gpu and not slow"`, 68 tests) passes before
anything is uploaded.

## 0. What ships to the pod

- The repo (`LLM/HiLS-Attention-Lite/`) with `requirements.txt` installed.
- The packed shards: `data/pretrain_chinchilla/shards/` (uint32, 50M tokens
  each) — either build on the pod with `python3 data/prepare_data.py` or copy
  from a machine that already has `shared_data`.
- The LLaMA-3-Lite checkpoint (`.pt`) only for the B1/B4 headline forms.

Nothing else is needed: the loop is single-device, single process; no
distributed setup, no external services.

## 1. Boundary checks, in this order

```bash
python scripts/microbench_a100.py --phase A   # B5: peak < 15 GB
python scripts/microbench_a100.py --phase B   # B5: peak < 20 GB
python scripts/step_time_a100.py --compile    # B6: MFU ≥ 33%
python scripts/e2e_gpu_smoke.py               # 200 A-steps + 20 B-steps + resume
```

Why this order: VRAM fails fast (seconds per phase) and is the cheapest
thing to fix — a Phase B shape that doesn't fit never reaches the MFU
question. MFU is the money gate: at < 33% the 40–48 h estimate is wrong and
the profile-first rule applies — the per-chunk gather in
`models/attention.py:hils_attention_core` is the suspect, not the router
([debugging playbook](debugging-playbook.md) has the recipe). The e2e smoke
is last because it runs the real loop: if it completes with no
`[watchdog]` / `[nan-guard]` / rollback markers in the log and its Phase A
checkpoint resumes, the loop is pod-ready.

## 2. Launch and survive the long run

```bash
nohup bash scripts/launch_a100.sh > pretrain.log 2>&1 &
tail -f pretrain.log
```

(or a tmux session — anything that survives SSH drops). The launcher execs
`training/pretrain.py` on the production config and **auto-resumes** from the
latest complete checkpoint in `checkpoints/pretrain_a100`.

Resume semantics — the part that matters when the pod dies at hour 30:

- `utils/checkpoint.py:CheckpointManager` writes three files per step
  (weights safetensors + optimizer pt + meta json). A step is resumable only
  when all three exist; a checkpoint torn mid-write is ignored, so a crash
  can never resume into a half-written state.
- Re-running the launcher after a crash is safe by construction — it picks up
  at the last complete save (`save_interval: 4000`, so a worst-case loss is
  < 4000 steps ≈ 30 min at Phase A pace).
- Interrupting intentionally (Ctrl-C, timeout) is equally safe; just relaunch.

What to watch in the log (`utils/logging.py:TrainingLogger` every 50 steps):

- **tokens/step = 131,072** — the phase shapes derive from
  `training/pretrain.py:phase_at`; anything else means the config is not the
  one you think it is.
- **The phase flip at step 53,407** — a small transient loss bump is expected;
  the 500-step re-warm tent absorbs it (playbook: "Phase A→B switch spike").
- **Guard markers**: `[watchdog] max-share > 50%` and `[nan-guard]` should
  appear zero times in a healthy run; a single rollback is the guard working,
  repeated rollbacks at the same step are playbook territory, not knobs to
  turn.
- **Selection stats** — entropy and max share drifting toward collapse is
  visible here long before the watchdog fires (playbook: "Router collapse").

Disk: three files × ~16 saves + final ≈ tens of GB — check quota before the
run; the VRAM estimator (`utils/memory.py:assert_fits_in_available_gpu`)
guards memory before step 1, nothing guards disk.

## 3. The headline evals, after step 61,037

```bash
python scripts/longctx_eval.py --ours-ckpt checkpoints/pretrain_a100 \
    --baseline-ckpt ../LLaMA-3-Lite/checkpoints/model_final.pt --ctx 16384   # B1 + B2
python scripts/retrieval_eval.py --ours-ckpt checkpoints/pretrain_a100 \
    --lengths 16384 32768 65536 --needles 20                                  # B3
python scripts/loss_parity_eval.py --ours-ckpt checkpoints/pretrain_a100 \
    --texts-file heldout.txt --baseline-ckpt ../LLaMA-3-Lite/checkpoints/model_final.pt  # B4
```

Order: B1/B2 first (cheapest, and B2 needs no baseline checkpoint), then B3
(longest decode), then B4 (cross-repo import, needs the baseline + held-out
text). Every marker these print — `[PASS …]` or `[DISCLOSED …]` — is the
number that goes in the README table verbatim; a disclosed miss is a result
to report, not something to rerun until it passes. Until these run, the four
headline numbers stay *targets* everywhere in the docs.

If a gate is disclosed, the playbook's eval rows ("KV-access fraction off
target", "Throughput below the MFU gate") name the first suspects before any
rerun is spent.

## 4. After the runs

- Copy `checkpoints/pretrain_a100/` and the eval logs off the pod; the
  checkpoint set is the artifact the eval scripts and
  [quickstart §4](quickstart.md) sampling both consume.
- Update the README headline table (targets → measured) and the
  `> **Status**` block — measured numbers only, with the marker the script
  printed next to them.
- The eval scripts are re-runnable at any later checkpoint; nothing in them
  is one-shot.