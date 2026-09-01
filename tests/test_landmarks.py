"""Landmark projector + chunk-query pooling (Phase 1.2).

Contract (plan §1.2, DESIGN §2.2): ℓ_j = W_ℓ · meanpool(K_j) with identity
init, so at init the landmark is exactly the per-chunk key mean; q̄_i is the
per-chunk query mean. fp64 everywhere — these are exactness tests.
"""
import torch

from models.landmarks import LandmarkProjector, pooled_query


def test_landmark_identity_init_is_meanpool():
    torch.manual_seed(0)
    B, KV, T, D, C = 2, 2, 512, 8, 128
    k = torch.randn(B, KV, T, D, dtype=torch.float64)
    proj = LandmarkProjector(n_kv_heads=KV, head_dim=D).double()
    # identity init: W_ℓ = I exactly, so forward ≡ meanpool with no distortion
    assert torch.equal(proj.proj.weight, torch.eye(D, dtype=torch.float64))
    landmarks = proj(k, C)
    ref = k.view(B, KV, T // C, C, D).mean(dim=3)
    assert torch.equal(landmarks, ref)  # fp64 exact


def test_landmark_grad_flow():
    torch.manual_seed(1)
    B, KV, T, D, C = 2, 2, 256, 8, 128
    k = torch.randn(B, KV, T, D, dtype=torch.float64, requires_grad=True)
    proj = LandmarkProjector(n_kv_heads=KV, head_dim=D).double()
    landmarks = proj(k, C)
    landmarks.pow(2).sum().backward()
    assert k.grad is not None and k.grad.abs().sum() > 0  # ∂‖ℓ‖/∂K ≠ 0
    assert proj.proj.weight.grad is not None
    assert proj.proj.weight.grad.abs().sum() > 0  # ∂ℓ/∂W_ℓ ≠ 0


def test_landmark_shapes():
    # N = T/C for both pretrain phases: 4096 (N=32) and 16384-length shape math
    for T, N in ((512, 4), (4096, 32)):
        k = torch.randn(2, 4, T, 8)
        q = torch.randn(2, 16, T, 8)
        assert LandmarkProjector(n_kv_heads=4, head_dim=8)(k, 128).shape == (2, 4, N, 8)
        assert pooled_query(q, 128).shape == (2, 16, N, 8)