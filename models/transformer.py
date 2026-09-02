"""HiLSAttentionLM: Embedding → 24×HiLSBlock (grad-ckpt) → final RMSNorm → tied head."""
from dataclasses import dataclass

import torch
import torch.nn as nn

from models.attention import precompute_freqs_cis
from models.block import HiLSBlock, RMSNorm
from models.landmarks import LandmarkProjector
from training.losses import chunked_lm_ce


@dataclass
class HiLSConfig:
    """Every model: key from configs/pretrain_a100_341m.yaml (defaults = 341M)."""

    vocab_size: int = 50257
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 4
    head_dim: int = 64
    ffn_dim: int = 3072
    weight_tying: bool = True
    rms_norm_eps: float = 1e-5
    init_std: float = 0.02
    rope_theta: float = 500000.0
    max_seq_len: int = 16384
    attn_impl: str = "sdpa"
    # --- the HiLS core ---
    chunk_len: int = 128
    n_selected: int = 8
    landmark_init: str = "identity"
    fusion: str = "score_softmax"
    selection_scope: str = "per_query_chunk"
    aux_balance_weight: float = 0.01
    straight_through_selection: bool = False  # documented ablation hook (off in v1)

    @classmethod
    def from_yaml(cls, path):
        """Load the model: sub-dict of a config YAML; training:/data: sections are Phase 4."""
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw["model"])

    def __post_init__(self):
        """Fail fast on knobs with exactly one implementation each (load_config
        constructs HiLSConfig eagerly, so a bad value dies before the GPU)."""
        assert self.attn_impl in ("sdpa", "eager"), f"unknown attn_impl: {self.attn_impl!r}"
        assert self.landmark_init == "identity", (
            f"unknown landmark_init: {self.landmark_init!r} (only 'identity' exists in v1)")
        assert self.fusion == "score_softmax", (
            f"unknown fusion: {self.fusion!r} (only 'score_softmax' exists in v1)")
        assert self.selection_scope == "per_query_chunk", (
            f"unknown selection_scope: {self.selection_scope!r} "
            "(only 'per_query_chunk' exists in v1)")
        assert self.straight_through_selection is False, (
            "straight_through_selection is a documented ablation, not implemented "
            "in v1 — gradients flow through the fusion weights only")


class HiLSAttentionLM(nn.Module):
    """Dense backbone, HiLS attention. forward(tokens) → logits; with targets
    → (CE + mean-per-layer λ·L_bal, aux-for-logging). freqs_cis is computed for
    the actual sequence length on every forward — no position cache, so forward
    beyond max_seq_len extrapolates (DESIGN §7.6)."""

    def __init__(self, cfg: HiLSConfig):
        super().__init__()
        assert cfg.attn_impl in ("sdpa", "eager"), f"unknown attn_impl: {cfg.attn_impl!r}"
        assert cfg.head_dim * cfg.n_heads == cfg.d_model, "d_model must factor as heads × head_dim"
        self.cfg = cfg
        self.grad_ckpt_every = None  # runtime knob: training loop sets from config training:
        self._fast_blocks = None     # runtime knob: compiled block handles (training loop)
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([
            HiLSBlock(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim,
                      cfg.chunk_len, cfg.n_selected, cfg.aux_balance_weight,
                      cfg.ffn_dim, cfg.rms_norm_eps, cfg.attn_impl,
                      cfg.rope_theta, cfg.max_seq_len)
            for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)  # h @ E.T
        if cfg.weight_tying:
            self.head.weight = self.embed.weight
        half = cfg.head_dim // 2
        inv_freq = cfg.rope_theta ** (-2.0 * torch.arange(half).float() / cfg.head_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._init_weights()
        # the normal-init pass above clobbers W_ℓ — restore the identity prior
        # (ℓ = meanpooled K at init; DESIGN §2.2)
        for m in self.modules():
            if isinstance(m, LandmarkProjector) and m.proj.weight.device.type != "meta":
                nn.init.eye_(m.proj.weight)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Embedding, nn.Linear)):
                if m.weight.device.type != "meta":
                    nn.init.normal_(m.weight, std=self.cfg.init_std)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _freqs_cis(self, T: int, device) -> torch.Tensor:
        return precompute_freqs_cis(self.cfg.head_dim, T, self.cfg.rope_theta,
                                    dtype=self.inv_freq.dtype).to(device)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None):
        """tokens (B, T) → logits (B, T, V); with targets → (total_loss, aux)."""
        freqs_cis = self._freqs_cis(tokens.size(1), tokens.device)
        h = self.embed(tokens)
        auxes = []
        blocks = self._fast_blocks if self._fast_blocks is not None else self.blocks
        for i, block in enumerate(blocks):
            if self.grad_ckpt_every and i % self.grad_ckpt_every == 0 \
                    and self.training and torch.is_grad_enabled():
                h, aux = torch.utils.checkpoint.checkpoint(
                    block, h, freqs_cis, use_reentrant=False)
            else:
                h, aux = block(h, freqs_cis)
            auxes.append(aux)
        h = self.final_norm(h)
        if targets is None:
            return self.head(h)
        ce = chunked_lm_ce(h, self.head.weight, targets)
        aux = torch.stack(auxes).mean()  # λ-weighted per layer, averaged across layers
        return ce + aux, aux

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Greedy sampling, recompute-per-step. The working window is left-padded
        with token 0 to a chunk multiple (exact tiling invariant); the last logits
        row is always the last real token. Phase 3 replaces this with the cached
        sparse decode (inference/generate.py)."""
        ids = prompt_ids
        for _ in range(max_new_tokens):
            window = ids[:, -self.cfg.max_seq_len:]
            pad = (-window.size(1)) % self.cfg.chunk_len
            work = torch.cat([window.new_zeros(window.size(0), pad), window], dim=1)
            logits = self.forward(work)
            ids = torch.cat([ids, logits[:, -1].argmax(dim=-1, keepdim=True)], dim=1)
        return ids