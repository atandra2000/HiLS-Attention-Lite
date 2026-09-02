#!/usr/bin/env python3
"""Step-time / MFU benchmark at the Phase A optimizer-step shape (plan §4.2).

One timed unit = one optimizer step of the production loop: ``grad_accum``
micro-batches (BF16-autocast forward/backward at micro_bs×seq) + grad clip +
fused AdamW. Shapes come from ``training/pretrain.py:phase_at``.

FLOPs convention: 6·N per token (2·N fwd + 4·N bwd) over unique (untied)
parameters, ×(1 + 1/(3·E)) recompute factor for every-Eth-block gradient
checkpointing; attention score FLOPs are excluded (sublinear under HiLS
chunk selection — they are the point of the architecture). Gate: MFU ≥ 33%
on an A100 80GB (312 TFLOPS BF16 peak); expect ~33–37%. If below the gate,
profile the per-chunk gather before touching the router.

Usage:
    python scripts/step_time_a100.py [--compile]   # --compile matches the
                                                   # production loop's block compile
On CPU the script prints a tokens/sec proxy and exits 0 (MFU undefined).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.transformer import HiLSAttentionLM, HiLSConfig  # noqa: E402
from training.pretrain import build_optimizer, load_config, phase_at  # noqa: E402

MFU_GATE = 33.0
A100_BF16_TFLOPS = 312.0


def n_unique_params(model: torch.nn.Module) -> int:
    """Parameter count without double-counting tied weights (embed <-> head)."""
    seen: set[int] = set()
    total = 0
    for p in model.parameters():
        ptr = p.data_ptr()
        if ptr not in seen:
            seen.add(ptr)
            total += p.numel()
    return total


def main() -> int:
    p = argparse.ArgumentParser(description="Phase A step-time / MFU benchmark (plan §4.2)")
    p.add_argument("--config", default=str(ROOT / "configs" / "pretrain_a100_341m.yaml"))
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--compile", action="store_true",
                   help="compile blocks like the production loop (fair MFU, slow warmup)")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device — MFU is undefined here; run on the A100 pod "
              "(python scripts/step_time_a100.py --compile).")
        return 0

    cfg = load_config(args.config)
    t = cfg["training"]
    seq_len, micro_bs, accum = phase_at(0, cfg)
    tokens_per_step = seq_len * micro_bs * accum

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = HiLSAttentionLM(HiLSConfig(**cfg["model"])).to("cuda")
    model.grad_ckpt_every = t["grad_checkpoint_every"] if t["grad_checkpoint"] else None
    if args.compile:
        model._fast_blocks = [torch.compile(b, mode=t.get("compile_mode", "max-autotune"))
                              for b in model.blocks]
    optimizer = build_optimizer(model, t)

    vocab = cfg["model"]["vocab_size"]
    x = torch.randint(0, vocab, (micro_bs, seq_len), device="cuda")
    y = torch.randint(0, vocab, (micro_bs, seq_len), device="cuda")

    def optimizer_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accum):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                total, _ = model(x, y)
            total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
        optimizer.step()

    for _ in range(args.warmup):
        optimizer_step()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.steps):
        optimizer_step()
    end.record()
    torch.cuda.synchronize()

    sec_per_step = start.elapsed_time(end) / 1000.0 / args.steps
    tps = tokens_per_step / sec_per_step
    n = n_unique_params(model)
    recompute = 1.0 + (1.0 / (3.0 * t["grad_checkpoint_every"])
                       if t["grad_checkpoint"] else 0.0)
    tflops = 6.0 * n * tps * recompute / 1e12
    mfu = tflops / A100_BF16_TFLOPS * 100
    print(f"[step_time] shape (ms={micro_bs}, seq={seq_len}, accum={accum}) "
          f"= {tokens_per_step:,} tokens/optimizer-step")
    print(f"[step_time] {sec_per_step:.2f} s/step, {tps:,.0f} tokens/sec, "
          f"{tflops:.1f} TFLOPS (6·N·recompute convention)")
    print(f"[step_time] MFU ~{mfu:.1f}% [{'PASS' if mfu >= MFU_GATE else 'FAIL'} >= {MFU_GATE:.0f}%]")
    return 0 if mfu >= MFU_GATE else 1


if __name__ == "__main__":
    sys.exit(main())