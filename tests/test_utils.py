"""utils/ tests: CheckpointManager completeness/atomicity, memory estimator,
training logger (Phase 4.1, plan §4.1 'copy + adapt tests/test_utils.py')."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.transformer import HiLSAttentionLM, HiLSConfig
from utils.checkpoint import CheckpointManager
from utils.logging import TrainingLogger
from utils.memory import assert_fits_in_available_gpu, estimate_model_memory_gb


def tiny_model():
    cfg = HiLSConfig(vocab_size=512, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
                     head_dim=16, ffn_dim=128, chunk_len=32, n_selected=4,
                     max_seq_len=256)
    return HiLSAttentionLM(cfg)


def test_checkpoint_latest_skips_incomplete(tmp_ckpt_dir):
    """Atomicity: a step is resumable only when all three files exist."""
    man = CheckpointManager(str(tmp_ckpt_dir))
    model, opt = tiny_model(), None
    man.save(model, torch.optim.AdamW(model.parameters()), 1)
    # simulate a crash mid-write of step 2: weights file only, no optim/meta
    from safetensors.torch import save_file
    save_file({"w": torch.zeros(2)}, tmp_ckpt_dir / "model_step_2.safetensors")
    assert man.latest_step() == 1


def test_checkpoint_empty_and_missing(tmp_ckpt_dir):
    man = CheckpointManager(str(tmp_ckpt_dir))
    assert man.latest_step() is None
    with pytest.raises(FileNotFoundError):
        man.load(tiny_model(), 5, device="cpu")


def test_checkpoint_meta_roundtrip(tmp_ckpt_dir):
    man = CheckpointManager(str(tmp_ckpt_dir))
    model = tiny_model()
    opt = torch.optim.AdamW(model.parameters())
    man.save(model, opt, 3, extra_meta={"tokens_seen": 123})
    meta = man.load(model, 3, device="cpu", optimizer=opt)
    assert meta["step"] == 3 and meta["tokens_seen"] == 123


def test_estimate_model_memory_ranking():
    model = tiny_model()
    full = estimate_model_memory_gb(model, seq_len=256, batch_size=2,
                                    grad_checkpoint=False)
    ckpt = estimate_model_memory_gb(model, seq_len=256, batch_size=2,
                                    grad_checkpoint=True)
    assert full > ckpt > 0  # checkpointing retains only block boundaries


def test_assert_fits_in_available_gpu_runs_on_cpu():
    assert_fits_in_available_gpu(1.0)  # no CUDA → guard skips silently


def test_training_logger_smoke(capsys):
    lg = TrainingLogger(log_every=2, seq_len=128, batch_size=2)
    lg.log(1, 2.0)  # inside the interval: no output
    lg.log(2, 1.5, lr=1.0e-3)
    out = capsys.readouterr().out
    assert "loss=" in out and "ppl=" in out and "lr=" in out