"""LongContextEvaluator tests: the Phase 5.1 measurement harness (plan §5.1).

Pins the protocol the headline scripts inherit: instrument zeroing before the
timed decode window (steady-state KV-access fraction), length snapping to
exact chunk multiples, NLL ≡ the model's own loss path (total − aux) and ≡
eager CE, deterministic needle-in-a-haystack rows, positional readout equal to
the model's forward, and baseline flag restoration. CPU-only at tiny scale —
the A100 gates (B1–B4) stay pod-side (AGENTS.md rule 4).
"""
import pytest
import torch
import torch.nn.functional as F

from inference.evaluate import LongContextEvaluator, Needle, _read_hidden
from models.transformer import HiLSAttentionLM, HiLSConfig


def tiny_cfg(**kw) -> HiLSConfig:
    d = dict(vocab_size=50257, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2,
             head_dim=32, ffn_dim=256, weight_tying=True, rms_norm_eps=1e-5,
             init_std=0.02, rope_theta=500000.0, max_seq_len=512,
             attn_impl="sdpa", chunk_len=128, n_selected=8, aux_balance_weight=0.01)
    d.update(kw)
    return HiLSConfig(**d)


class _StubTok:
    """Minimal .encode() tokenizer — the only contract evaluate() relies on."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 50257 for c in text]


def test_decode_throughput_and_kv_fraction_exact():
    torch.manual_seed(91)
    model = HiLSAttentionLM(tiny_cfg())
    res = LongContextEvaluator(model).evaluate([2048], n_steps=1, warmup=1)
    assert res["throughput"][2048] > 0
    # steady-state window: one chunk step at position 1920 reads k·C keys and
    # N=16 landmarks per layer; denominator = full-attention bytes at T=2048
    L, k, C, N = model.cfg.n_layers, model.cfg.n_selected, model.cfg.chunk_len, 16
    expected = (2 * L * k * C + L * N) / (2 * L * 1 * 2048)
    assert abs(res["kv_fraction"][2048] - expected) < 1e-9


def test_nll_matches_eager_ce():
    torch.manual_seed(93)
    model = HiLSAttentionLM(tiny_cfg(n_layers=1))
    ev = LongContextEvaluator(model)
    g = torch.Generator().manual_seed(5)
    stream = torch.randint(0, 50257, (1, 385), generator=g)
    res = ev.evaluate([384], token_stream=stream, n_steps=1, warmup=1)
    with torch.no_grad():
        logits = model(stream[:, :384])  # tileable input; position p predicts stream[p+1]
        eager = F.cross_entropy(logits.reshape(-1, 50257),
                                stream[:, 1:].reshape(-1)).item()
    assert abs(res["nll"][384] - eager) < 1e-4
    assert 9.0 < res["nll"][384] < 12.0  # random init sits near ln(50257)


def test_texts_need_a_tokenizer():
    model = HiLSAttentionLM(tiny_cfg(n_layers=1))
    with pytest.raises(ValueError):
        LongContextEvaluator(model).evaluate([512], texts=["hello"], n_steps=1, warmup=1)


def test_texts_path_matches_token_stream():
    torch.manual_seed(95)
    model = HiLSAttentionLM(tiny_cfg(n_layers=1))
    ev = LongContextEvaluator(model, tokenizer=_StubTok())
    text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo " * 4
    ids = torch.tensor([ord(c) % 50257 for c in text], dtype=torch.long)
    res_text = ev.evaluate([384], texts=[text], n_steps=1, warmup=1)
    res_toks = ev.evaluate([384], token_stream=ids, n_steps=1, warmup=1)
    assert res_text["nll"][384] == res_toks["nll"][384]


def test_retrieval_accuracy_deterministic():
    torch.manual_seed(97)
    model = HiLSAttentionLM(tiny_cfg(n_layers=1, d_model=64, n_heads=2,
                                     n_kv_heads=1, head_dim=32, ffn_dim=128))
    ev = LongContextEvaluator(model)
    needles = [Needle(key_ids=[101, 102, 103, 104], value_ids=[5000 + i])
               for i in range(4)]
    res = ev.evaluate([512], needles=needles, seed=11, n_steps=1, warmup=1)
    acc = res["retrieval"][512]
    assert 0.0 <= acc <= 1.0
    assert len(res["retrieval_detail"]) == 4
    again = ev.evaluate([512], needles=needles, seed=11, n_steps=1, warmup=1)
    assert again["retrieval"][512] == acc  # seeded haystacks → identical rows


def test_retrieval_multi_token_value_teacher_forced():
    torch.manual_seed(99)
    model = HiLSAttentionLM(tiny_cfg(n_layers=1))
    ev = LongContextEvaluator(model)
    needles = [Needle(key_ids=[201, 202, 203], value_ids=[300, 301, 302])]
    res = ev.evaluate([512], needles=needles, seed=3, n_steps=1, warmup=1)
    assert len(res["retrieval_detail"]) == 1
    assert 0.0 <= res["retrieval"][512] <= 1.0


def test_retrieval_rejects_needle_larger_than_ctx():
    model = HiLSAttentionLM(tiny_cfg(n_layers=1))
    ev = LongContextEvaluator(model)
    big = Needle(key_ids=list(range(200)), value_ids=list(range(200, 400)))
    with pytest.raises(ValueError):
        ev.evaluate([512], needles=[big], n_steps=1, warmup=1)


def test_all_chunk_baseline_restores_selection():
    torch.manual_seed(101)
    model = HiLSAttentionLM(tiny_cfg(n_layers=1))
    ev = LongContextEvaluator(model)
    res = ev.evaluate([512], baselines=["all-chunk"], n_steps=1, warmup=1)
    assert res["baselines"]["all-chunk"]["throughput"][512] > 0
    assert model.blocks[0].attn.n_selected == 8  # flipped for the baseline, restored
    with pytest.raises(ValueError):
        ev.evaluate([512], baselines=["no-such-baseline"], n_steps=1, warmup=1)


def test_lengths_snap_to_chunk_multiples():
    model = HiLSAttentionLM(tiny_cfg(n_layers=1))
    ev = LongContextEvaluator(model)
    res = ev.evaluate([500], n_steps=1, warmup=1)
    assert set(res["throughput"]) == {384}
    with pytest.raises(ValueError):
        ev.evaluate([384], n_steps=3, warmup=1)  # needs C·(1 + 1 + 3) = 640


def test_positional_readout_matches_model_forward():
    torch.manual_seed(103)
    model = HiLSAttentionLM(tiny_cfg())
    toks = torch.randint(0, 50257, (1, 256))
    with torch.no_grad():
        ref = model(toks)
        ours = model.head(_read_hidden(model, toks))
    assert torch.allclose(ref, ours, atol=1e-6)