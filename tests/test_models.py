"""Model wiring tests: HiLSBlock + HiLSAttentionLM (Phase 2.2).

Covers plan §2.2 bullets plus the DESIGN §7.3/§7.6 forward checks
(first-chunk degenerate case, length extrapolation) and the class-contract
greedy generate. Full-size param budget: 320–365M (plan §2.2).
"""
import pytest
import torch

from models.attention import precompute_freqs_cis
from models.block import HiLSBlock
from models.transformer import HiLSAttentionLM, HiLSConfig


def tiny_cfg(**kw) -> HiLSConfig:
    d = dict(vocab_size=50257, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2,
             head_dim=32, ffn_dim=256, weight_tying=True, rms_norm_eps=1e-5,
             init_std=0.02, rope_theta=500000.0, max_seq_len=512,
             attn_impl="sdpa", chunk_len=128, n_selected=8, aux_balance_weight=0.01)
    d.update(kw)
    return HiLSConfig(**d)


def test_forward_shapes():
    torch.manual_seed(21)
    model = HiLSAttentionLM(tiny_cfg())
    logits = model(torch.randint(0, 50257, (1, 256)))
    assert logits.shape == (1, 256, 50257)
    assert torch.isfinite(logits).all()


def test_param_count():
    with torch.device("meta"):  # construction-only budget check, zero RAM
        model = HiLSAttentionLM(HiLSConfig())
    n = sum(p.numel() for p in model.parameters())
    assert 3.2e8 < n < 3.65e8  # ~341M ±4%


def test_weight_tying_shared():
    model = HiLSAttentionLM(tiny_cfg())
    assert model.head.weight is model.embed.weight


def test_two_step_overfit():
    torch.manual_seed(23)
    model = HiLSAttentionLM(tiny_cfg(n_layers=2))
    tokens = torch.randint(0, 50257, (2, 256))
    targets = torch.randint(0, 50257, (2, 256))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(2):
        opt.zero_grad()
        loss, _ = model(tokens, targets)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[1] < losses[0]


def test_grad_flow_all_params():
    torch.manual_seed(29)
    model = HiLSAttentionLM(tiny_cfg(n_layers=2))
    tokens = torch.randint(0, 50257, (2, 256))
    targets = torch.randint(0, 50257, (2, 256))
    loss, _ = model(tokens, targets)
    loss.backward()
    missing = [n for n, p in model.named_parameters()
               if p.grad is None or p.grad.abs().sum() == 0]
    assert not missing, f"params without gradient: {missing}"  # incl. W_ℓ


def test_compile_block_runs():
    torch.manual_seed(31)
    cfg = tiny_cfg(d_model=64, n_heads=2, n_kv_heads=1, head_dim=32,
                   ffn_dim=128, chunk_len=128, n_selected=2)
    block = HiLSBlock(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim,
                      cfg.chunk_len, cfg.n_selected, cfg.aux_balance_weight,
                      cfg.ffn_dim, cfg.rms_norm_eps, cfg.attn_impl,
                      cfg.rope_theta, cfg.max_seq_len).eval()
    freqs = precompute_freqs_cis(cfg.head_dim, 256, cfg.rope_theta)
    x = torch.randn(1, 256, 64)
    with torch.no_grad():
        out_ref, aux_ref = block(x, freqs)
        out_c, aux_c = torch.compile(block)(x, freqs)  # selection stays eager
    assert torch.allclose(out_c, out_ref, atol=1e-4)
    assert torch.allclose(aux_c, aux_ref, atol=1e-4)


def test_first_chunk_degenerate_case():
    # T = C: a single chunk that can only select itself; forward well-defined
    torch.manual_seed(37)
    model = HiLSAttentionLM(tiny_cfg(n_layers=1))
    loss, _ = model(torch.randint(0, 50257, (1, 128)),
                    torch.randint(0, 50257, (1, 128)))
    assert torch.isfinite(loss)


def test_length_extrapolation_forward():
    # 4× beyond max_seq_len: freqs/landmarks scale linearly, no position cache
    torch.manual_seed(41)
    model = HiLSAttentionLM(tiny_cfg(n_layers=1))
    logits = model(torch.randint(0, 50257, (1, 2048)))
    assert logits.shape == (1, 2048, 50257)
    assert torch.isfinite(logits).all()


def test_generate_greedy():
    torch.manual_seed(43)
    model = HiLSAttentionLM(tiny_cfg(n_layers=1)).eval()
    prompt = torch.randint(0, 50257, (1, 8))
    with torch.no_grad():
        out = model.generate(prompt, max_new_tokens=4)
    assert out.shape == (1, 12)
    assert torch.equal(out[:, :8], prompt)  # prompt preserved
    with torch.no_grad():
        assert torch.equal(out, model.generate(prompt, max_new_tokens=4))  # greedy det


def test_grad_checkpoint_block_runs():
    torch.manual_seed(47)
    model = HiLSAttentionLM(tiny_cfg(n_layers=2))
    model.grad_ckpt_every = 1  # checkpoint every block
    tokens = torch.randint(0, 50257, (2, 256))
    targets = torch.randint(0, 50257, (2, 256))
    loss, aux = model(tokens, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(aux)
    assert all(p.grad is not None for p in model.parameters())