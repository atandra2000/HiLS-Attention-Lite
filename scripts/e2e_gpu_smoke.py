#!/usr/bin/env python3
"""End-to-end GPU smoke through the production loop (plan §4.2).

Runs ``training/pretrain.py:train`` twice over fixed synthetic batches — the
same seam the unit tests use (``batches=`` / ``max_steps=``) — so the smoke
exercises the real loop machinery: BF16 autocast, phase shapes, NaN guard,
selection watchdog, VRAM estimator, checkpoint save + resume.

    Phase A: --steps-a optimizer steps at (ms=8, seq=4096)
    Phase B: --steps-b optimizer steps at (ms=1, seq=16384)

Checks per phase: loop reaches the step cap, loss decreases, no guard fired
(nan-guard / watchdog / rollback in the logs), and the Phase A checkpoint is
written and resumable. ``--tiny`` swaps in a 2-layer config so the whole
harness self-checks on CPU in seconds:

    python scripts/e2e_gpu_smoke.py --tiny                 # CPU self-check
    python scripts/e2e_gpu_smoke.py                        # A100: 200 A-steps + 20 B-steps

Fixed (repeated) batches are deliberate: random iid tokens make a loss-
decrease assertion vacuous near the ln(vocab) floor; a memorizable stream
makes it a real signal.
"""
from __future__ import annotations

import argparse
import copy
import logging
import sys
import tempfile
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.pretrain import load_config, phase_at, train  # noqa: E402
from utils.checkpoint import CheckpointManager  # noqa: E402

GUARD_MARKERS = ("watchdog", "rolled back", "nan-guard")


class _GuardCapture(logging.Handler):
    """Collect training.pretrain log records so the smoke can assert quietness."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def fixed_batches(n: int, micro_bs: int, seq_len: int, vocab: int, seed: int) -> list[dict]:
    """n dict copies of one seeded random batch — memorizable, so loss can fall."""
    g = torch.Generator().manual_seed(seed)
    toks = torch.randint(0, vocab, (micro_bs, seq_len + 1), generator=g)
    batch = {"input": toks[:, :-1].clone(), "target": toks[:, 1:].clone()}
    return [dict(batch) for _ in range(n)]


def _write_cfg(cfg: dict, workdir: Path, name: str) -> str:
    path = workdir / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(path)


def run_phase_a(cfg: dict, workdir: Path, steps: int):
    """Phase-A-only train() run: switch pushed past the cap, warmup shortened."""
    t_a = copy.deepcopy(cfg)
    t_a["training"].update(
        warmup_steps=max(1, steps // 2),          # production warmup dwarfs a smoke run
        total_steps=steps + 1,
        phase_switch_step=steps + 1,              # stay in Phase A for the whole smoke
        save_interval=steps,                      # checkpoint at the cap for the resume check
        save_dir=str(workdir / "phaseA"),
    )
    path = _write_cfg(t_a, workdir, "smoke_phaseA")
    seq, ms, accum = phase_at(0, t_a)
    batches = fixed_batches(steps * accum + 2, ms, seq, t_a["model"]["vocab_size"], seed=11)
    return train(path, batches=batches, max_steps=steps)


def run_phase_b(cfg: dict, workdir: Path, steps: int):
    """Phase-B-only train() run at the 16K window (switch pinned to 0)."""
    t_b = copy.deepcopy(cfg)
    t_b["training"].update(
        warmup_steps=max(1, steps // 2),
        total_steps=steps,
        phase_switch_step=0,                      # start directly in Phase B
        save_interval=steps,
        save_dir=str(workdir / "phaseB"),
    )
    path = _write_cfg(t_b, workdir, "smoke_phaseB")
    seq, ms, accum = phase_at(0, t_b)
    batches = fixed_batches(steps * accum + 2, ms, seq, t_b["model"]["vocab_size"], seed=23)
    return train(path, batches=batches, max_steps=steps)


TINY_TRAINING = dict(micro_batch_size=2, gradient_accumulation_steps=1,
                     lr=1.0e-3, min_lr_ratio=0.05, weight_decay=0.1, beta1=0.9,
                     beta2=0.95, grad_clip=1.0, grad_checkpoint=False,
                     grad_checkpoint_every=3, compile=False, compile_mode="default",
                     nan_guard=True, nan_guard_max_consecutive=1,
                     selection_watchdog=True, phase_a_seq_len=128,
                     log_interval=1000, phase_b_warmup_steps=2)
TINY_MODEL = dict(vocab_size=50257, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2,
                  head_dim=32, ffn_dim=256, weight_tying=True, rms_norm_eps=1e-5,
                  init_std=0.02, rope_theta=500000.0, max_seq_len=256,
                  attn_impl="sdpa", chunk_len=32, n_selected=4, aux_balance_weight=0.01)


def tiny_cfg() -> dict:
    """Structurally faithful 2-layer config (mirrors tests/test_training.py)."""
    return {"model": dict(TINY_MODEL), "training": dict(TINY_TRAINING),
            "data": {"train_data_path": "unused-by-smoke"}}


def assert_quiet(capture: _GuardCapture, phase: str) -> None:
    fired = [m for m in capture.messages
             if any(marker in m.lower() for marker in GUARD_MARKERS)]
    assert not fired, f"{phase}: stability guards fired: {fired[:3]}"


def main() -> int:
    p = argparse.ArgumentParser(description="Two-phase e2e smoke through the production loop")
    p.add_argument("--config", default=str(ROOT / "configs" / "pretrain_a100_341m.yaml"))
    p.add_argument("--steps-a", type=int, default=200)
    p.add_argument("--steps-b", type=int, default=20)
    p.add_argument("--workdir", default=None,
                   help="checkpoint/log dir (default: fresh temp dir)")
    p.add_argument("--tiny", action="store_true",
                   help="2-layer CPU self-check config instead of --config")
    args = p.parse_args()

    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="hils_smoke_"))
    workdir.mkdir(parents=True, exist_ok=True)
    cfg = tiny_cfg() if args.tiny else load_config(args.config)

    capture = _GuardCapture()
    logging.getLogger("training.pretrain").addHandler(capture)

    print(f"[smoke] workdir: {workdir}")
    # --- Phase A: train, checkpoint, resume --------------------------------
    state_a = run_phase_a(cfg, workdir, args.steps_a)
    assert state_a.step == args.steps_a, f"phase A stopped at step {state_a.step}"
    assert state_a.losses[-1] < state_a.losses[0], (
        f"phase A loss did not decrease: {state_a.losses[0]:.4f} -> {state_a.losses[-1]:.4f}")
    assert CheckpointManager(t_a_dir(workdir)).latest_step() == args.steps_a, \
        "phase A checkpoint missing at the step cap"
    # resume: one more step continues from the checkpoint, not from zero
    resumed = _resume_one(cfg, workdir, args.steps_a)
    assert resumed.step == args.steps_a + 1, \
        f"resume restarted the loop (step {resumed.step}, expected {args.steps_a + 1})"
    assert_quiet(capture, "phase A")

    # --- Phase B: 16K window runs clean -------------------------------------
    state_b = run_phase_b(cfg, workdir, args.steps_b)
    assert state_b.step == args.steps_b, f"phase B stopped at step {state_b.step}"
    assert state_b.losses[-1] < state_b.losses[0], (
        f"phase B loss did not decrease: {state_b.losses[0]:.4f} -> {state_b.losses[-1]:.4f}")
    assert_quiet(capture, "phase B")

    print(f"[smoke] PASS: phase A {args.steps_a} steps "
          f"({state_a.losses[0]:.3f}→{state_a.losses[-1]:.3f}), resume ok; "
          f"phase B {args.steps_b} steps @16K "
          f"({state_b.losses[0]:.3f}→{state_b.losses[-1]:.3f}); guards quiet")
    return 0


def t_a_dir(workdir: Path) -> str:
    """Checkpoint dir of the Phase A smoke run."""
    return str(workdir / "phaseA")


def _resume_one(cfg: dict, workdir: Path, steps_a: int):
    """train() one step past the Phase A cap against the same save_dir."""
    t_a = copy.deepcopy(cfg)
    t_a["training"].update(
        warmup_steps=max(1, steps_a // 2),
        total_steps=steps_a + 2,
        phase_switch_step=steps_a + 2,            # resume run stays in Phase A too
        save_interval=steps_a + 5,
        save_dir=str(workdir / "phaseA"),
    )
    path = _write_cfg(t_a, workdir, "smoke_phaseA_resume")
    seq, ms, accum = phase_at(0, t_a)
    batches = fixed_batches(accum + 2, ms, seq, t_a["model"]["vocab_size"], seed=11)
    return train(path, batches=batches, max_steps=steps_a + 1)


if __name__ == "__main__":
    sys.exit(main())