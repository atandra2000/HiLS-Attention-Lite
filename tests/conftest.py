import pytest
import torch


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def tmp_ckpt_dir(tmp_path):
    d = tmp_path / "ckpt"
    d.mkdir()
    return d


@pytest.fixture
def tmp_data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    return d


# tiny_cfg / tiny_model fixtures land with models/transformer.py (Phase 2):
# they need HiLSAttentionLM + HiLSConfig to exist (house conftest pattern).