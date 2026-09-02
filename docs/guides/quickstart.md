# Guide: quickstart

End-to-end: environment → data → training → sampling → headline evaluation.
Assumes the repo root (`LLM/HiLS-Attention-Lite/`) with `requirements.txt`
installed. Training needs the A100 pod (single 80GB device); everything else
runs on CPU.

## 0. Environment check

`uv run --no-project --with-requirements requirements.txt` builds a cached
ephemeral env from `requirements.txt` — the portable form; the system
`python3` with the deps installed is equivalent. Verified end-to-end:

```bash
uv run --no-project --with-requirements requirements.txt python \
    -c "import torch, sys; print(sys.version.split()[0], torch.__version__)"
# expect: 3.14.0 2.13.0 (or the system interpreter's versions)
python3 -m pytest -m "not gpu and not slow"
```

Expected: the interpreter version, the torch version, and `66 passed`
(CPU-only gate; `pytest.ini` marks `gpu`/`slow` tests out). A doc-integrity
pass rides the same gate:

```bash
python3 scripts/check_docs.py --coverage --links   # anchors resolve, symbols cited, links valid
python3 scripts/build_docs_html.py                 # rebuild the docs_html/ portal
```

Expected: `[doc-refs] coverage: PASS` and the portal path printed.

## 1. A 30-second model smoke (CPU)

```bash
uv run python -c "
import torch
from models.transformer import HiLSConfig, HiLSAttentionLM
m = HiLSAttentionLM(HiLSConfig(vocab_size=512, d_model=64, n_layers=2, n_heads=2,
                               n_kv_heads=1, head_dim=32, ffn_dim=128,
                               max_seq_len=256, chunk_len=64, n_selected=2))
print('logits', m(torch.randint(0, 512, (1, 128))).shape)   # torch.Size([1, 128, 512])
"
```

Every model is `models/transformer.py:HiLSConfig` +
`models/transformer.py:HiLSAttentionLM`; the defaults are the 341M production
shape (`configs/pretrain_a100_341m.yaml` is its YAML form).

## 2. Data

`data/prepare_data.py:main` delegates to the workspace `shared_data` pipeline
and pins `LLM_DATA_ROOT` to `data/pretrain_chinchilla` — the pack stage runs
as a subprocess that honors only that env var:

```bash
python3 data/prepare_data.py                     # full pipeline (download → tokenize → pack)
python3 data/prepare_data.py --skip-download     # corpus already on disk
python3 data/prepare_data.py --skip-download --skip-clean --skip-tokenize --skip-pack
                                                 # config-materialization smoke (no I/O heavy stages)
```

Shards land in `data/pretrain_chinchilla/shards/` — 50M tokens each, uint32,
GPT-2 BPE (vocab 50,257), same corpus as the other Lites.

## 3. Training (A100 pod)

```bash
bash scripts/launch_a100.sh
```

One config, one schedule: 53,407 steps @ seq 4096 (7.0B tokens) then 7,630
steps @ seq 16384 (1.0B tokens) with a 500-step LR re-warm at the switch —
~40–48 h at 35–40% MFU. The loop auto-resumes from the latest complete
checkpoint in `checkpoints/pretrain_a100`
(`utils/checkpoint.py:CheckpointManager` — three files per step, or the step
is not resumable). Before the long run, run the pod-side boundary checks:

```bash
python scripts/microbench_a100.py --phase A   # gate: < 15 GB
python scripts/microbench_a100.py --phase B   # gate: < 20 GB
python scripts/step_time_a100.py              # gate: MFU ≥ 33%
python scripts/e2e_gpu_smoke.py               # 200 A-steps + 20 B-steps + resume
```

## 4. Sample from a checkpoint

Training produces `checkpoints/pretrain_a100/model_step_<n>.safetensors`;
`inference/generate.py:step_decode` decodes chunk-synchronously (one chunk of
C tokens per step, sparse — see [long-context phases](../concepts/long-context-phases.md)):

```bash
uv run python -c "
import torch
from models.transformer import HiLSConfig, HiLSAttentionLM
from inference.generate import new_cache, prefill, step_decode
m = HiLSAttentionLM(HiLSConfig(vocab_size=512, d_model=64, n_layers=2, n_heads=2,
                               n_kv_heads=1, head_dim=32, ffn_dim=128,
                               max_seq_len=256, chunk_len=64, n_selected=2)).eval()
cache = new_cache(m, batch=1, max_seq=256)
logits = prefill(m, cache, torch.randint(0, 512, (1, 128)))       # chunks 0–1
logits = step_decode(m, cache, torch.randint(0, 512, (1, 64)), 128)  # chunk 2
print('decoded chunk logits', logits.shape)                        # torch.Size([1, 64, 512])
"
```

## 5. Evaluate (headline gates)

```bash
python scripts/longctx_eval.py --tiny --ctx 512 1024          # CPU self-check
python scripts/longctx_eval.py --ours-ckpt checkpoints/pretrain_a100 \
    --baseline-ckpt ../LLaMA-3-Lite/checkpoints/<model>.pt --ctx 16384   # A100: B1 + B2

python scripts/retrieval_eval.py --tiny                        # CPU self-check
python scripts/retrieval_eval.py --ours-ckpt checkpoints/pretrain_a100 \
    --lengths 16384 32768 65536                                # A100: B3

python scripts/loss_parity_eval.py --tiny                      # CPU plumbing check
python scripts/loss_parity_eval.py --ours-ckpt checkpoints/pretrain_a100 \
    --texts-file heldout.txt --baseline-ckpt ../LLaMA-3-Lite/checkpoints/<model>.pt  # A100: B4
```

Every script prints measured numbers with explicit `[PASS …]`/`[DISCLOSED …]`
markers — never a bare claim. Until the A100 runs happen, the four headline
numbers are *targets*, and the README says so.

## 6. Where to go next

- [HILS.md](../../HILS.md) — why the architecture is shaped this way
- [Debugging playbook](debugging-playbook.md) — when a run misbehaves
- [Config reference](../references/config.md) — every YAML field explained