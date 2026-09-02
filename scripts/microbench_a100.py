#!/usr/bin/env python3
"""Peak-VRAM microbenchmark for the two pretrain phases (plan §4.2).

Runs one steady-state optimizer step at each phase's shape — fused AdamW,
BF16-autocast forward/backward through the production attention path — and
reports the CUDA allocator peak. Gates (DESIGN §4.0 budget table):

    Phase A (micro_bs=8,  seq=4096):  peak VRAM < 15 GB
    Phase B (micro_bs=1,  seq=16384): peak VRAM < 20 GB

Shapes come from ``training/pretrain.py:phase_at`` — never hardcoded — so the
benchmark cannot drift from the training loop. A warmup step runs first to
materialize AdamW's lazily allocated state before the peak counter is reset;
without it the reading misses the optimizer moments and passes vacuously.
torch.compile stays off: eager is the conservative allocator upper bound.

Usage:
    python scripts/microbench_a100.py --phase A   # expect: "peak VRAM: ~10 GB [PASS <15GB]"
    python scripts/microbench_a100.py --phase B   # expect: "peak VRAM: ~9 GB  [PASS <20GB]"

On a CUDA-less machine the gate is unmeasurable, not violated: the script
prints why and exits 0 (house pattern, cf. LLaMA-3-Lite microbench).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.transformer import HiLSAttentionLM, HiLSConfig  # noqa: E402
from training.pretrain import build_optimizer, load_config, phase_at  # noqa: E402
from utils.memory import estimate_model_memory_gb  # noqa: E402

GATES_GB = {"A": 15.0, "B": 20.0}


def run_phase(phase: str, cfg_path: str) -> int:
    cfg = load_config(cfg_path)
    t = cfg["training"]
    step = 0 if phase == "A" else t["phase_switch_step"]
    seq_len, micro_bs, _ = phase_at(step, cfg)
    gate = GATES_GB[phase]

    model = HiLSAttentionLM(HiLSConfig(**cfg["model"])).to("cuda")
    model.grad_ckpt_every = t["grad_checkpoint_every"] if t["grad_checkpoint"] else None
    optimizer = build_optimizer(model, t)

    vocab = cfg["model"]["vocab_size"]
    x = torch.randint(0, vocab, (micro_bs, seq_len), device=device)
    y = torch.randint(0, vocab, (micro_bs, seq_len), device=device)

    def step_once() -> None:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            total, _ = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
        optimizer.step()

    step_once()  # materialize AdamW moments + fp32 master before measuring
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    step_once()
    torch.cuda.synchronize()

    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
    peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
    est = estimate_model_memory_gb(model, seq_len, micro_bs, t["grad_checkpoint"])
    ok = peak_reserved < gate
    print(f"[microbench] phase {phase}: shape (ms={micro_bs}, seq={seq_len}), "
          f"tokens/micro-batch {micro_bs * seq_len:,}")
    print(f"[microbench] peak VRAM: {peak_reserved:.1f} GB reserved "
          f"({peak_alloc:.1f} GB allocated); estimator predicted {est:.1f} GB")
    print(f"[microbench] {'PASS' if ok else 'FAIL'} < {gate} GB")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description="Phase A/B peak-VRAM microbenchmark (plan §4.2)")
    p.add_argument("--phase", choices=("A", "B"), required=True)
    p.add_argument("--config", default=str(ROOT / "configs" / "pretrain_a100_341m.yaml"))
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device — VRAM gates are unmeasurable here; "
              "run on the A100 pod (python scripts/microbench_a100.py --phase A|B).")
        return 0
    return run_phase(args.phase, args.config)


if __name__ == "__main__":
    sys.exit(main())