"""Two-phase pretrain loop for HiLS-Attention-Lite (plan §4.1).

Standard teacher-forced loop over ``shared_data`` shards — the only novelty vs.
the house loop lives in the config: the aux balance loss is wired inside
``models/transformer.py:HiLSAttentionLM.forward``, and the Phase A→B switch
(seq 4096→16384, micro-bs 8→1, grad-accum 4→8) keeps tokens/optimizer-step
constant at 131,072 with a 500-step LR re-warm tent at the switch. Stability
machinery: fused AdamW (fp32 master weights), BF16 autocast (GPU only),
grad-ckpt every 3rd layer, NaN guard with checkpoint rollback, selection
watchdog (max-share > 50% for 500 steps → rollback).
"""
from __future__ import annotations

import contextlib
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LLM_ROOT = _PROJECT_ROOT.parent
for _p in (str(_PROJECT_ROOT), str(_LLM_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.router import selection_stats
from models.transformer import HiLSAttentionLM, HiLSConfig
from utils.checkpoint import CheckpointManager
from utils.logging import TrainingLogger
from utils.memory import assert_fits_in_available_gpu, estimate_model_memory_gb

logger = logging.getLogger(__name__)

EOS_TOKEN_ID = 50256  # GPT-2 EOS/PAD — house parity with data/prepare_data.py


@dataclass
class TrainState:
    step: int
    tokens_seen: int
    model: HiLSAttentionLM
    optimizer: torch.optim.Optimizer
    losses: list[float] = field(default_factory=list)  # per optimizer step


def load_config(cfg_path: str) -> dict:
    """Parse the pretrain YAML; validates the model section eagerly."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    HiLSConfig(**cfg["model"])
    return cfg


def _as_cfg(cfg):
    return load_config(cfg) if isinstance(cfg, str) else cfg


def build_optimizer(model: HiLSAttentionLM, cfg: dict) -> torch.optim.Optimizer:
    """Fused AdamW; params stay fp32 (the master copy) — bf16 lives in autocast."""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]  # norms/biases
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]),
        fused=torch.cuda.is_available())


def lr_at(step: int, cfg) -> float:
    """Warmup → one cosine across both phases; a 500-step re-warm tent rides on
    the cosine at phase_switch_step (continuous at both ends, peak at the
    midpoint). ``cfg`` is a config dict or a YAML path."""
    cfg = _as_cfg(cfg)
    t = cfg["training"]
    lr, floor = t["lr"], t["lr"] * t["min_lr_ratio"]
    warm, total = t["warmup_steps"], t["total_steps"]
    if step < warm:
        return lr * (step + 1) / warm  # (step+1): the first optimizer step trains
    g = floor + (lr - floor) * 0.5 * (1.0 + math.cos(math.pi * (step - warm) / (total - warm)))
    s, w = t["phase_switch_step"], t["phase_b_warmup_steps"]
    if s <= step < s + w:
        return g + (lr - g) * math.sin(math.pi * (step - s) / w)
    return g


def phase_at(step: int, cfg) -> tuple[int, int, int]:
    """(seq_len, micro_bs, grad_accum): (4096, 8, 4) → (16384, 1, 8) at
    phase_switch_step. Phase B keeps one 16K window per micro-batch (attention
    memory scales ~seq²) and restores the tokens/step budget via accumulation."""
    cfg = _as_cfg(cfg)
    t = cfg["training"]
    seq_a = t.get("phase_a_seq_len", 4096)
    bs_a, acc_a = t["micro_batch_size"], t["gradient_accumulation_steps"]
    if step < t["phase_switch_step"]:
        return seq_a, bs_a, acc_a
    seq_b = cfg["model"]["max_seq_len"]
    budget = seq_a * bs_a * acc_a
    assert budget % seq_b == 0, f"tokens/step {budget} not divisible by phase-B seq {seq_b}"
    return seq_b, 1, budget // seq_b


def _build_loader(cfg: dict, seq_len: int, micro_bs: int):
    """Canonical shared_data loader for the current phase."""
    from shared_data.loader import build_training_data

    d = cfg["data"]
    data_dir = Path(d["train_data_path"])
    if not data_dir.is_absolute():
        data_dir = _PROJECT_ROOT / data_dir
    workers = d.get("num_workers", 4)
    loader_cfg = {
        "seq_len": seq_len,
        "batch_size": micro_bs,
        "eos_token_id": EOS_TOKEN_ID,
        "tokenizer_name": d.get("tokenizer", "gpt2"),
        "val_split": d.get("val_split", 0.05),
        "num_workers": workers,
        "prefetch_factor": d.get("prefetch_factor", 4) if workers > 0 else None,
        "pin_memory": d.get("pin_memory", True),
        "shuffle_seed": d.get("shuffle_seed", 42),
    }
    train_loader, _, _ = build_training_data(loader_cfg, data_dir=data_dir)
    return train_loader


def train(cfg_path: str, *, batches=None, max_steps: int | None = None,
          device: str | None = None) -> TrainState:
    """Run the two-phase pretrain loop. ``batches`` (batch dicts) and
    ``max_steps`` are test/smoke seams overriding data loading and the step cap."""
    cfg = load_config(cfg_path)
    t = cfg["training"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = HiLSAttentionLM(HiLSConfig(**cfg["model"])).to(device)
    model.grad_ckpt_every = t["grad_checkpoint_every"] if t["grad_checkpoint"] else None
    if t.get("compile") and device == "cuda":
        # compiled handles live beside the raw ModuleList: checkpoints keep clean
        # keys (no torch.compile `_orig_mod.` prefix) and eval loads stay trivial
        model._fast_blocks = [torch.compile(b, mode=t.get("compile_mode", "max-autotune"))
                              for b in model.blocks]
    optimizer = build_optimizer(model, t)
    ckpt = CheckpointManager(t["save_dir"])
    state = TrainState(step=0, tokens_seen=0, model=model, optimizer=optimizer)

    latest = ckpt.latest_step()
    if latest is not None:
        meta = ckpt.load(model, latest, device=device, optimizer=optimizer)
        state.step = int(meta.get("step", latest))
        state.tokens_seen = int(meta.get("tokens_seen", 0))
        print(f"[train] resumed at step {state.step} ({state.tokens_seen:,} tokens seen)")

    def _restore(step_target: int) -> None:
        nonlocal rolled_back_to
        meta = ckpt.load(model, step_target, device=device, optimizer=optimizer)
        state.step = int(meta.get("step", step_target))
        state.tokens_seen = int(meta.get("tokens_seen", 0))
        rolled_back_to = step_target
        logger.warning("[guard] rolled back to last good checkpoint (step %d)", step_target)

    limit = t["total_steps"] if max_steps is None else max_steps
    nan_streak = watchdog_streak = 0
    rolled_back_to = None
    loader, cur, tlog, micro = None, None, None, 0
    while state.step < limit:
        seq_len, micro_bs, accum = phase_at(state.step, cfg)
        if cur != (seq_len, micro_bs):
            loader = batches if batches is not None else _build_loader(cfg, seq_len, micro_bs)
            cur = (seq_len, micro_bs)
            tlog = TrainingLogger(log_every=t["log_interval"], seq_len=seq_len,
                                  batch_size=micro_bs * accum)
            if device == "cuda":
                est = estimate_model_memory_gb(model, seq_len, micro_bs,
                                               t["grad_checkpoint"])
                assert_fits_in_available_gpu(est)
        for batch in loader:
            if state.step >= limit or phase_at(state.step, cfg)[:2] != cur:
                break  # phase switch (or cap): rebuild the loader
            tokens = batch["input"].to(device)
            targets = batch["target"].to(device)
            autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                        if device == "cuda" else contextlib.nullcontext())
            with autocast:
                total, aux = model(tokens, targets)
            loss = float(total.detach())
            if not math.isfinite(loss):
                if not t["nan_guard"]:
                    raise FloatingPointError(f"non-finite loss {loss} at step {state.step}")
                nan_streak += 1
                optimizer.zero_grad(set_to_none=True)
                micro = 0
                logger.warning("[nan-guard] non-finite loss at step %d (streak %d)",
                               state.step, nan_streak)
                if nan_streak >= t["nan_guard_max_consecutive"]:
                    last = ckpt.latest_step()
                    if last is None:
                        raise RuntimeError(
                            "nan-guard: non-finite loss and no checkpoint to roll back to")
                    if rolled_back_to == last:
                        raise RuntimeError(
                            f"nan-guard: loss still non-finite after rollback to step {last}")
                    _restore(last)
                    break
                continue
            (total / accum).backward()
            micro += 1
            if micro % accum:
                continue
            lr = lr_at(state.step, cfg)
            for gp in optimizer.param_groups:
                gp["lr"] = lr
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            micro = 0
            state.step += 1
            state.tokens_seen += seq_len * micro_bs * accum
            state.losses.append(loss)
            nan_streak = 0
            tlog.log(state.step, loss, lr)
            if state.step % t["save_interval"] == 0:
                ckpt.save(model, optimizer, state.step,
                          extra_meta={"tokens_seen": state.tokens_seen})
            if t["selection_watchdog"]:
                n_chunks = tokens.size(1) // model.cfg.chunk_len
                max_share = max(selection_stats(b.attn.last_selected, n_chunks)["max_share"]
                                for b in model.blocks)
                watchdog_streak = watchdog_streak + 1 if max_share > 0.5 else 0
                if watchdog_streak >= 500:
                    logger.warning("[watchdog] max-share > 50%% for %d steps", watchdog_streak)
                    last = ckpt.latest_step()
                    if last is None or rolled_back_to == last:
                        raise RuntimeError(
                            "watchdog: selection collapsed with no fresh checkpoint to roll back to")
                    _restore(last)
                    break
        if micro:  # partial accumulation window dropped at epoch end / phase switch
            optimizer.zero_grad(set_to_none=True)
            micro = 0
    return state


if __name__ == "__main__":
    train(sys.argv[1])