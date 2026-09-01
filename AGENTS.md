# AGENTS.md — HiLS-Attention-Lite

> Read the workspace `LLM/AGENTS.md` and the parent `CoreProjects/AGENTS.md`
> first. Higher-level rules are authoritative; this file adds project-specific
> rules only (and wins on HiLS-Attention-Lite-specific conflicts).
>
> **Project:** `LLM/HiLS-Attention-Lite/` · **Type:** faithful sparse-attention
> reproduction (from-scratch pretrain)
> **Scale:** ~341M dense · 8.0B tokens (7B @ 4096 + 1B @ 16K) · ~40–48 h on A100 80GB
> **Stack:** PyTorch 2.x, BF16, torch.compile(max-autotune), SDPA, dataclasses

From-scratch reimplementation of **HiLS learned sparse chunk attention**: every
component implemented end-to-end (no stubs). The router is trained by the LM
loss through fusion weights — there is no separate retrieval loss.

## Subagent: `sparse-attention-engineer`

**Triggers:** "why is my router collapsing", "debug top-k gather shapes",
"landmark cache correctness", "KV access fraction", "extrapolation eval",
"decode vs teacher-forced mismatch".

**Architecture (24 layers, dense backbone, HiLS attention):**
- Chunking: C=128; landmarks ℓ_j = W_ℓ · meanpool(K_j), identity-init.
- Router: chunk-query meanpool → per-head scores vs landmarks → causal top-k
  (own chunk always included) → per-head fusion softmax.
- Attention: per selected chunk, independent SDPA; outputs fused by g_ij.
- Aux: Herfindahl load-balance loss λ=0.01 + selection watchdog (max-share).
- Backbone: house block, GQA 16Q/4KV, RoPE θ=500K, SwiGLU, weight tying.

**Hard rules:**
1. Never suggest HF Trainer / Lightning.
2. Never "fix" selection into a fixed pattern (sliding window, NSA triad) —
   learned routing is the repo's claimed cell; `tests/test_router.py` must
   pass after ANY change to `models/router.py` or `models/landmarks.py`.
3. Never break causal selection (j ≤ i) — `test_selection_is_causal` guards
   every router change; decode correctness depends on it.
4. Never claim the ≥2× throughput, ≤0.08 KV-fraction, ≥85% retrieval, or
   ≤+5% ΔNLL headlines without running the corresponding `scripts/*_eval.py`;
   report measured numbers.
5. Never add MTP, MLA, GDN, MoE, sinks, SSM, YaRN, or diffusion — different
   portfolio cells (`llm-research/` non-overlap criterion).
6. Never let the sparse path silently become dense: `k=N` is a test case, not
   a fallback mode; `attn_impl="eager"` exists only as the reference path.
7. Keep tests CPU-friendly (minutes); GPU checks live in `scripts/*a100*.py`.