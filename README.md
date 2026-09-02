# HiLS-Attention-Lite

A from-scratch PyTorch implementation of **HiLS learned sparse chunk
attention** (Tencent, arXiv:2607.02980): a ~341M-param GQA transformer whose
attention selects *which* chunks each query chunk reads — via landmark
scoring and causal top-k — and is **trained end-to-end by the LM loss**
through score-fused attention. The portfolio's first learnable (natively
trained) sparse-attention model.

> **Status:** scaffolded 2026-09-02 per
> [`llm-research/EXECUTION-PLAN-hils-attention-lite.md`](../../llm-research/EXECUTION-PLAN-hils-attention-lite.md)
> (Phases 0–3 complete; Phase 4.1 two-phase training loop landed — Phase 4.2
> A100 boundary checks and Phase 5 evaluation pending).
> Design spec: [`llm-research/DESIGN-hils-attention-lite.md`](../../llm-research/DESIGN-hils-attention-lite.md).