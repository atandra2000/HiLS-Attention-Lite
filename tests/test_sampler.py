"""Sparse decode tests: cached KV + landmark cache, chunk-synchronous decode
steps, KV access counter (Phase 3.1, DESIGN §7.6).

Decode is chunk-synchronous: one step consumes a full chunk of C tokens and
attends its own chunk only once complete, so chunk-step state equals the
teacher-forced forward exactly at every layer — the load-bearing §7.6 gate
compares the whole decoded chunk's logits, fp64, atol=1e-5.
"""
import torch

from inference.generate import kv_access_fraction, new_cache, prefill, step_decode
from models.attention import apply_rope
from models.chunking import chunk_index_bounds
from models.transformer import HiLSAttentionLM, HiLSConfig


def tiny_cfg(**kw) -> HiLSConfig:
    d = dict(vocab_size=50257, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2,
             head_dim=32, ffn_dim=256, weight_tying=True, rms_norm_eps=1e-5,
             init_std=0.02, rope_theta=500000.0, max_seq_len=512,
             attn_impl="sdpa", chunk_len=128, n_selected=8, aux_balance_weight=0.01)
    d.update(kw)
    return HiLSConfig(**d)


def test_sparse_decode_matches_teacher_forced():
    torch.manual_seed(61)
    model = HiLSAttentionLM(tiny_cfg()).double()
    T, C = 384, 128  # prefill 2 chunks, decode the 3rd chunk in one chunk step
    tokens = torch.randint(0, 50257, (1, T))
    cache = new_cache(model, batch=1, max_seq=T)
    prompt_logits = prefill(model, cache, tokens[:, :256])
    with torch.no_grad():
        ref = model(tokens)
    # cache-fill sanity: the prefilled chunks' logits equal teacher-forced
    assert torch.allclose(prompt_logits, ref[:, 256 - C:256], atol=1e-5)
    # incremental chunk decode ≡ teacher-forced over the same tokens (whole chunk)
    logits = step_decode(model, cache, tokens[:, 256:], 256)
    assert torch.allclose(logits, ref[:, 256:], atol=1e-5)
    assert cache.seq_len == T


def test_kv_access_fraction_counter():
    torch.manual_seed(67)
    model = HiLSAttentionLM(tiny_cfg())  # 2 layers
    T = 1024  # N = 8 = k → n_sel = k exactly, no clamping
    tokens = torch.randint(0, 50257, (1, T + 128))
    cache = new_cache(model, batch=1, max_seq=T + 128)
    prefill(model, cache, tokens[:, :T])
    # prefill shares the step code path — zero the instrument to measure one step
    cache.kv_keys_read = 0
    cache.landmark_reads = 0
    cache.steps = 0
    step_decode(model, cache, tokens[:, T:], T)
    L, k, C = model.cfg.n_layers, model.cfg.n_selected, model.cfg.chunk_len
    assert cache.steps == 1
    # counter reads exactly k·C keys per decode step per layer (+ landmark read)
    assert cache.kv_keys_read == L * k * C
    assert cache.landmark_reads == L * (T // C + 1)

    # fraction ≈ k·C/T at the 16K context: 8·128/16384 = 0.0625 + landmark read
    big = HiLSAttentionLM(tiny_cfg(n_layers=1))
    tk = torch.randint(0, 50257, (1, 16384 + 128))
    cache16 = new_cache(big, batch=1, max_seq=16384 + 128)
    prefill(big, cache16, tk[:, :16384])
    cache16.kv_keys_read = 0
    cache16.landmark_reads = 0
    cache16.steps = 0
    step_decode(big, cache16, tk[:, 16384:], 16384)
    frac = kv_access_fraction(cache16)
    expected = (2 * k * C + 129) / (2 * 16512)  # selected K+V keys + landmark read
    assert abs(frac - expected) < 1e-9
    assert abs(frac - k * C / 16384) < 0.01
    assert frac <= 0.08  # the ~6% effective-KV-access headline gate


def test_length_extrapolation_forward():
    torch.manual_seed(71)
    model = HiLSAttentionLM(tiny_cfg(max_seq_len=256, n_layers=1))
    logits = model(torch.randint(0, 50257, (1, 1024)))  # 4× beyond trained max
    assert logits.shape == (1, 1024, 50257)
    assert torch.isfinite(logits).all()
    # landmark count scales linearly with T: N = T/C
    _, n1 = chunk_index_bounds(256, 128)
    _, n4 = chunk_index_bounds(1024, 128)
    assert n4 == 4 * n1
    # decode also runs at the extrapolated length (cache sized for it)
    cache = new_cache(model, batch=1, max_seq=1024 + 128)
    prefill(model, cache, torch.randint(0, 50257, (1, 1024)))
    out = step_decode(model, cache, torch.randint(0, 50257, (1, 128)), 1024)
    assert out.shape == (1, 128, 50257)


def test_landmark_cache_freezes_finalized_chunks():
    torch.manual_seed(73)
    model = HiLSAttentionLM(tiny_cfg(n_layers=1))
    tokens = torch.randint(0, 50257, (1, 512))
    cache = new_cache(model, batch=1, max_seq=512)
    prefill(model, cache, tokens[:, :256])
    assert cache.landmarks[0][:, :, 2].abs().sum() == 0  # unreached chunk
    frozen0 = cache.landmarks[0][:, :, 0].clone()
    frozen1 = cache.landmarks[0][:, :, 1].clone()
    # decode chunks 2 and 3; finalized landmarks never move
    step_decode(model, cache, tokens[:, 256:384], 256)
    step_decode(model, cache, torch.randint(0, 50257, (1, 128)), 384)
    assert torch.equal(cache.landmarks[0][:, :, 0], frozen0)
    assert torch.equal(cache.landmarks[0][:, :, 1], frozen1)
    # and the frozen landmark equals the teacher-forced chunk landmark
    with torch.no_grad():
        blk = model.blocks[0]
        xn = blk.attn_norm(model.embed(tokens))
        k = blk.attn.k_proj(xn).view(1, 512, 2, 32).transpose(1, 2)
        k = apply_rope(k, model._freqs_cis(512, tokens.device))
        lm_ref = blk.attn.landmarks(k, 128)  # (1, KV, 4, D)
    assert torch.allclose(lm_ref[0, :, 2, :], cache.landmarks[0][0, :, 2, :], atol=1e-9)