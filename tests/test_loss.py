"""Chunked-CE pipeline path (Phase 2.2): training/losses.py:chunked_lm_ce vs
eager F.cross_entropy (the pipeline's two-path pair, house test scale).

fp32 parity at atol=1e-6 is calibrated at the house test's scale (small V,
multi-chunk). At the production vocab (50257 = 7 × 8192 chunks) the two-level
fp32 logsumexp differs from the fused reduction by a few ulps of ~10.8 — the
test asserts that honestly at atol=1e-4 rather than hiding it.
"""
import torch
import torch.nn.functional as F

from training.losses import chunked_lm_ce


def _eager(hidden, weight, targets):
    return F.cross_entropy((hidden @ weight.t()).view(-1, weight.size(0)),
                           targets.view(-1))


def test_chunked_ce_matches_eager():
    torch.manual_seed(53)
    # house scale: V=256, chunk 64 → 4 chunks, incl. exact division
    hidden = torch.randn(2, 64, 128)
    weight = torch.randn(256, 128)
    targets = torch.randint(0, 256, (2, 64))
    assert torch.allclose(chunked_lm_ce(hidden, weight, targets, vocab_chunk=64),
                          _eager(hidden, weight, targets), atol=1e-6)
    # partial last chunk: V=100, chunk 64 → 36-token tail, plus full-vocab chunk
    hidden = torch.randn(2, 32, 64)
    weight = torch.randn(100, 64)
    targets = torch.randint(0, 100, (2, 32))
    for chunk in (64, None):
        assert torch.allclose(chunked_lm_ce(hidden, weight, targets, vocab_chunk=chunk),
                              _eager(hidden, weight, targets), atol=1e-6)
    # production shape: V=50257 with the default 8192-token chunks (7 chunks)
    hidden = torch.randn(2, 64, 32)
    weight = torch.randn(50257, 32)
    targets = torch.randint(0, 50257, (2, 64))
    got = chunked_lm_ce(hidden, weight, targets)
    ref = _eager(hidden, weight, targets)
    assert abs(got.item() - ref.item()) < 1e-4  # fp32 two-level reduction ulps