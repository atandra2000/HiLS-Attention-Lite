# HiLS-Attention-Lite

A from-scratch PyTorch implementation of **HiLS learned sparse chunk
attention** (Tencent, arXiv:2607.02980): a ~341M-param GQA transformer whose
attention selects *which* chunks each query chunk reads — via landmark
scoring and causal top-k — and is **trained end-to-end by the LM loss**
through score-fused attention. The portfolio's first learnable (natively
trained) sparse-attention model.

> **Status:** scaffolded 2026-09-02 per
> [`llm-research/EXECUTION-PLAN-hils-attention-lite.md`](../../llm-research/EXECUTION-PLAN-hils-attention-lite.md)
> (Phases 0–3 complete; Phase 4.1 two-phase training loop and the Phase 4.2
> A100 boundary-check scripts (`scripts/microbench_a100.py`,
> `scripts/step_time_a100.py`, `scripts/e2e_gpu_smoke.py`) landed — A100 pod
> runs and Phase 5 evaluation pending).
> Design spec: [`llm-research/DESIGN-hils-attention-lite.md`](../../llm-research/DESIGN-hils-attention-lite.md).