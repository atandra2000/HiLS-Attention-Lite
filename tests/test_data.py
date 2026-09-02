"""Data-path tests: shard contract, Phase-B 16K windowing, manifest dedup
metadata, prepare_data shim tokenizer contract (Phase 4.1, plan §4.1).

The shared pipeline (shared_data/) is the canonical source; these tests pin the
contract HiLS-Attention-Lite's training loop consumes.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # …/LLM/ → shared_data

from shared_data.dataset import ShardDataset
from shared_data.loader import PackedDataset
from shared_data.manifest import Manifest

ROOT = Path(__file__).resolve().parents[1]
VOCAB, EOS = 50257, 50256


def write_shard(tmp_path, n_tokens=195):
    """Minimal manifest + shard in the shared pipeline's on-disk format."""
    shards = tmp_path / "shards"
    shards.mkdir()
    toks = ((np.arange(n_tokens) * 7919 + 13) % VOCAB).astype(np.uint32)
    toks.tofile(shards / "shard_00000.bin")
    sha = hashlib.sha256((shards / "shard_00000.bin").read_bytes()).hexdigest()
    manifest = {
        "version": "1.0.0", "vocab_size": VOCAB, "eos_token_id": EOS,
        "dtype": "uint32", "total_tokens": n_tokens, "shard_count": 1,
        "shards_dir": "shards",
        "shards": [{"index": 0, "path": "shard_00000.bin", "n_tokens": n_tokens,
                    "sha256": sha, "n_eos": 0}],
        "sources": {"fineweb-edu": {"target_tokens": 100, "actual_tokens": n_tokens,
                                    "n_docs": 3, "n_dedup_dropped": 0}},
    }
    (shards / "manifest.json").write_text(json.dumps(manifest))
    return shards


def test_shard_contract_uint32_and_id_range(tmp_path):
    ds = ShardDataset(write_shard(tmp_path))
    assert len(ds.shards) == 1
    mm = ds.shards[0]
    assert mm.dtype == "uint32"
    assert mm.shape[0] == 195
    assert int(mm.min()) >= 0 and int(mm.max()) < VOCAB  # ids ⊆ [0, 50257)


def test_phase_b_16k_windowing():
    """Phase-B loader contract: contiguous (seq_len+1) windows at seq 16384,
    next-token shift (input = chunk[:-1], target = chunk[1:])."""
    seq = 16384
    buf = (np.arange(3 * (seq + 1)) % VOCAB).astype(np.uint32)
    ds = PackedDataset(buf, seq_len=seq, eos_id=EOS)
    assert len(ds) == 3
    item = ds[0]
    assert item["input"].shape == (seq,) and item["target"].shape == (seq,)
    assert torch_equal(item["input"], buf[:seq])
    assert torch_equal(item["target"], buf[1:seq + 1])


def torch_equal(t, np_arr):
    import torch
    return torch.equal(t, torch.from_numpy(np_arr.astype(np.int64)))


def test_manifest_dedup_metadata(tmp_path):
    m = Manifest.load(write_shard(tmp_path) / "manifest.json")
    assert m.dtype == "uint32"
    assert m.vocab_size == VOCAB and m.eos_token_id == EOS
    src = m.sources["fineweb-edu"]
    assert src.n_dedup_dropped == 0  # dedup metadata carried per source
    assert m.validate() == []


def test_prepare_data_shim_tokenizer_contract(tmp_path):
    """The shim pins the project's GPT-2 tokenizer contract in data_config.yaml."""
    spec = importlib.util.spec_from_file_location(
        "hils_prepare_data", ROOT / "data" / "prepare_data.py")
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    out = shim._ensure_hils_data_config(tmp_path)
    import yaml
    cfg = yaml.safe_load(out.read_text())
    tok = cfg["pipeline"]["tokenizer"]
    assert tok["name"] == "gpt2"
    assert tok["vocab_size"] == VOCAB and tok["eos_token_id"] == EOS
    assert tok["pad_token_id"] == EOS
    assert cfg["_generator"] == "HiLS-Attention-Lite/data/prepare_data.py"