"""Training-loop tests: LR schedule, phase switch, two-step overfit, checkpoint
roundtrip, NaN-guard rollback, aux-loss wiring (Phase 4.1, plan §4.1).

Test list reconciled against DESIGN §7 (the authoritative source): the plan's
test_training.py bullets `test_two_step_overfit` and `test_grad_flow_all_params`
are DESIGN §7.7 model-wiring tests already covered by tests/test_models.py —
not duplicated here. The plan's '10 passed' verify line spans this file plus
tests/test_data.py plus tests/test_utils.py.
"""
import logging
import math
from pathlib import Path

import pytest
import torch
import yaml

import training.pretrain as pretrain
from models.transformer import HiLSAttentionLM, HiLSConfig
from training.pretrain import (
    TrainState,
    build_optimizer,
    lr_at,
    load_config,
    phase_at,
    train,
)
from utils.checkpoint import CheckpointManager

ROOT = Path(__file__).resolve().parents[1]
REAL_CFG = str(ROOT / "configs" / "pretrain_a100_341m.yaml")


def model_cfg(**kw) -> HiLSConfig:
    d = dict(vocab_size=50257, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2,
             head_dim=32, ffn_dim=256, weight_tying=True, rms_norm_eps=1e-5,
             init_std=0.02, rope_theta=500000.0, max_seq_len=256, attn_impl="sdpa",
             chunk_len=32, n_selected=4, aux_balance_weight=0.01)
    d.update(kw)
    return HiLSConfig(**d)


def tiny_yaml(tmp_path, **over) -> str:
    """Tiny but structurally faithful training config: accum=1 so micro==opt step."""
    training = dict(micro_batch_size=2, gradient_accumulation_steps=1, total_steps=6,
                    warmup_steps=2, phase_switch_step=4, phase_b_warmup_steps=2,
                    lr=1.0e-3, min_lr_ratio=0.05, weight_decay=0.1, beta1=0.9,
                    beta2=0.95, grad_clip=1.0, grad_checkpoint=False,
                    grad_checkpoint_every=3, compile=False, compile_mode="default",
                    save_interval=1, log_interval=1000, nan_guard=True,
                    nan_guard_max_consecutive=1, selection_watchdog=False,
                    save_dir=str(tmp_path / "ckpt"), phase_a_seq_len=128)
    training.update(over.get("training", {}))
    model = dict(vocab_size=50257, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2,
                 head_dim=32, ffn_dim=256, weight_tying=True, rms_norm_eps=1e-5,
                 init_std=0.02, rope_theta=500000.0, max_seq_len=256,
                 attn_impl="sdpa", chunk_len=32, n_selected=4, aux_balance_weight=0.01)
    model.update(over.get("model", {}))
    cfg = {"model": model, "training": training,
           "data": {"train_data_path": "unused-by-tests"}}
    p = tmp_path / "cfg.yaml"
    import yaml as _yaml
    p.write_text(_yaml.safe_dump(cfg))
    return str(p)


def make_batches(n, bs=2, seq=128, seed=7):
    """n copies of one batch — overfit tests need the same batch every step."""
    g = torch.Generator().manual_seed(seed)
    toks = torch.randint(0, 50257, (bs, seq + 1), generator=g)
    batch = {"input": toks[:, :-1].clone(), "target": toks[:, 1:].clone()}
    return [dict(batch) for _ in range(n)]


# ---------------------------------------------------------------- schedule ---

def test_lr_schedule_shape():
    """Warmup slope, cosine decay, 500-step re-warm tent at the switch, tail to
    min_lr_ratio (plan §4.1); deep-tail value pinned by the plan's verify line."""
    cfg = load_config(REAL_CFG)
    t = cfg["training"]
    lr, warm, total = t["lr"], t["warmup_steps"], t["total_steps"]
    S, W = t["phase_switch_step"], t["phase_b_warmup_steps"]

    assert lr_at(0, cfg) == pytest.approx(lr / warm)          # first step trains (no dead step)
    assert lr_at(warm // 2, cfg) == pytest.approx(lr * (warm // 2 + 1) / warm)  # linear slope
    assert lr_at(warm, cfg) == pytest.approx(lr)              # cosine takes over at the peak
    assert lr_at(30000, cfg) < lr_at(warm, cfg)               # cosine decay
    # re-warm tent: continuous at both ends, full peak at the midpoint
    assert lr_at(S, cfg) < lr_at(S + 100, cfg)                # rising
    assert lr_at(S + W // 2, cfg) == pytest.approx(lr)        # tent peaks at lr
    assert lr_at(S + W // 2, cfg) > lr_at(S + 400, cfg) > lr_at(S + W, cfg)  # falls back
    assert lr_at(total, cfg) == pytest.approx(lr * t["min_lr_ratio"])
    # plan verify: deep in the tail the schedule sits at min_lr_ratio (tol 1e-6)
    assert abs(lr_at(60000, REAL_CFG) - lr * 0.05) < 1e-6      # cfg arg may be a path


def test_phase_switch_config():
    cfg = load_config(REAL_CFG)
    assert phase_at(0, cfg) == (4096, 8, 4)
    assert phase_at(53407, cfg) == (16384, 1, 8)
    a, b = phase_at(0, cfg), phase_at(53407, cfg)
    assert a[0] * a[1] * a[2] == b[0] * b[1] * b[2] == 131072  # tokens/step preserved


# -------------------------------------------------------------------- loop ---

def test_two_step_overfit(tmp_path):
    path = tiny_yaml(tmp_path)
    state = train(path, batches=make_batches(4), max_steps=2)
    assert isinstance(state, TrainState) and state.step == 2
    assert len(state.losses) == 2 and state.losses[1] < state.losses[0]
    assert state.tokens_seen == 2 * 2 * 128  # steps · micro_bs · seq · accum


def test_final_step_saved(tmp_path):
    """The run's final state persists even off the save_interval grid — the
    headline evals must never score a checkpoint up to save_interval−1 steps
    stale (production: 61,037 steps with interval 4000 ⇒ last interval save 60,000)."""
    path = tiny_yaml(tmp_path, training={"save_interval": 100})
    state = train(path, batches=make_batches(4), max_steps=2)
    assert CheckpointManager(load_config(path)["training"]["save_dir"]).latest_step() \
        == state.step == 2


def test_checkpoint_roundtrip(tmp_path):
    path = tiny_yaml(tmp_path)
    batches = make_batches(3)
    state1 = train(path, batches=batches, max_steps=2)

    cfg = load_config(path)
    man = CheckpointManager(cfg["training"]["save_dir"])
    assert man.latest_step() == 2  # save_interval=1 → steps 1 and 2 saved
    fresh = HiLSAttentionLM(HiLSConfig(**cfg["model"]))
    man.load(fresh, 2, device="cpu")
    with torch.no_grad():
        x = batches[0]["input"]
        assert torch.equal(state1.model(x), fresh(x))

    state2 = train(path, batches=batches, max_steps=3)
    assert state2.step == 3  # resumed from step 2, not restarted at 0


def test_nan_guard_rolls_back(tmp_path, monkeypatch, caplog):
    class _NaNOnce(HiLSAttentionLM):
        calls = 0

        def forward(self, tokens, targets=None):
            _NaNOnce.calls += 1
            total, aux = super().forward(tokens, targets)
            if _NaNOnce.calls == 2:  # NaN exactly once: step 1 is good, step 2 fails
                return total * float("nan"), aux
            return total, aux

    monkeypatch.setattr(pretrain, "HiLSAttentionLM", _NaNOnce)
    path = tiny_yaml(tmp_path)  # nan_guard_max_consecutive=1
    with caplog.at_level(logging.WARNING, logger="training.pretrain"):
        state = train(path, batches=make_batches(4), max_steps=3)
    assert state.step == 3  # recovered: rollback to step 1, then steps 2-3 re-run
    assert any("rolled back" in r.getMessage().lower() for r in caplog.records)


def test_nan_guard_raises_without_checkpoint(tmp_path, monkeypatch):
    class _NaNAlways(HiLSAttentionLM):
        def forward(self, tokens, targets=None):
            total, aux = super().forward(tokens, targets)
            return total * float("nan"), aux

    monkeypatch.setattr(pretrain, "HiLSAttentionLM", _NaNAlways)
    path = tiny_yaml(tmp_path)
    with pytest.raises(RuntimeError, match="no checkpoint"):
        train(path, batches=make_batches(2), max_steps=1)


# ------------------------------------------------------------------ aux ------

def test_balance_loss_wired():
    """λ from config scales the aux term; λ=0 yields pure CE (plan §4.1)."""
    x = torch.randint(0, 50257, (1, 128))
    y = torch.randint(0, 50257, (1, 128))
    outs = []
    for lam in (0.0, 0.02, 0.04):
        torch.manual_seed(11)  # identical weights ⇒ identical CE and selection
        m = HiLSAttentionLM(model_cfg(aux_balance_weight=lam))
        with torch.no_grad():
            outs.append(m(x, y))
    (t0, a0), (t1, a1), (t2, a2) = outs
    assert a0.item() == 0.0                       # pure CE at λ=0
    assert a1.item() > 0
    assert torch.allclose(t1, t0 + a1, atol=1e-6)  # total = ce + aux
    assert torch.allclose(a2, 2 * a1, rtol=1e-4)   # aux linear in λ


def test_build_optimizer_param_groups(tmp_path):
    cfg = load_config(tiny_yaml(tmp_path))["training"]
    model = HiLSAttentionLM(HiLSConfig(**load_config(tiny_yaml(tmp_path))["model"]))
    opt = build_optimizer(model, cfg)
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["weight_decay"] == cfg["weight_decay"]
    assert opt.param_groups[1]["weight_decay"] == 0.0
    assert all(p.dim() < 2 for g in opt.param_groups[1:] for p in g["params"])