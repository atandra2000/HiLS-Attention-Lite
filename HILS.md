# HILS.md — HiLS learned sparse chunk attention, end to end

The authoritative technical doc for HiLS-Attention-Lite: what the operator
is, why each piece is shaped the way it is, what is inherited from upstream
HiLS and what is a deliberate Lite decision, and how the whole thing is
verified. Companion documents: [`llm-research/DESIGN-hils-attention-lite.md`](../../llm-research/DESIGN-hils-attention-lite.md)
(justification) and [`llm-research/EXECUTION-PLAN-hils-attention-lite.md`](../../llm-research/EXECUTION-PLAN-hils-attention-lite.md)
(the mechanical contract this repo implements). Component deep-dives live in
[docs/](docs/README.md); this file is the spine.

## 0. The one-paragraph summary

HiLS-Attention-Lite is a ~341M-parameter GQA transformer whose attention
**selects which 128-token chunks each query chunk reads**. Every chunk j of
the sequence carries a learned *landmark* ℓ_j (a linear compression of its
keys); every query chunk i scores its chunk-mass query q̄_i against all
admissible landmarks, keeps its own chunk plus the top-(k−1) by score
(k = 8, causal: j ≤ i), attends each selected chunk with an **independent
per-chunk softmax**, and fuses the k outputs by a softmax over the
retrieval scores. Because the selected *scores* re-enter the forward pass as
fusion weights g_ij, the LM loss trains the router end-to-end — no retrieval
loss, no RL, no fixed sparsity pattern. At 16K context the model reads
8 of 128 chunks per query: ~6% effective KV access, the property the whole
evaluation harness measures. Backbone: house LLaMA-style block (RoPE
θ=500K, SwiGLU, weight tying), 8.0B Chinchilla-class tokens on one A100 —
7.0B at seq 4096, then 1.0B at seq 16384 with a 500-step LR re-warm.

## 1. Why hierarchical score-fused attention

Dense attention at 16K costs O(T²) compute and an O(T) KV read **per token
decoded** — the dominant cost of long-context serving. Fixed sparsity
(sliding windows, dilated bands, block-sparse masks) buys the FLOPs back but
throws away content-based recall: a fact at depth 0.7 is simply unreachable
from the current position unless it falls inside the band.

Learned chunk routing is the middle path: the model itself decides, per
query chunk, which past chunks matter — *and the decision is trainable by
the ordinary LM loss*, because the selection's scores flow back through the
fusion weights. That last clause is the reason this repo exists: sparse
attention schemes with fixed patterns are trivial to implement and cannot
learn what to skip; sparse schemes with separate retrieval objectives train
two models and hope they agree. HiLS's hierarchical score-fused attention
makes the retrieval signal and the language-modeling signal the same signal.

Three consequences define the engineering:

1. **Selection is chunk-granular** — static shapes, batched SDPA, compile-
   friendly (per-token selection would re-enter data-dependent shapes).
2. **Per-chunk softmax, not joint** — k independent softmaxes over C keys
   each, fused by g. This is the operator (§2); a single softmax over k·C
   keys is a *different function*, and the repo never silently degrades into
   it (`k = N` is a tested degenerate case, not a dense mode).
3. **The KV cache stays physically dense in v1** — the ~6% claim is
   *effective access*, measured by instrumentation, not cache bytes
   (§8); selective materialization is a documented v2 idea.

## 2. The operator: hierarchical score-fused attention

### 2.1 Annotated forward pass

```
tokens (B, T) ──► Embedding (d_model 1024, tied with head)
                    │
              24 × HiLSBlock:
                    │  RMSNorm (models/block.py:RMSNorm, fused F.rms_norm)
                    │   │
                    │   ▼  models/attention.py:HiLSAttention (the distinctive module)
                    │   1. Q (16 heads), K,V (4 KV heads) — RoPE θ=500K applied to Q,K
                    │   2. chunk into C=128 tiles: N = T/C           (exact tiling)
                    │   3. ℓ_j = W_ℓ · meanpool(K_j)   per KV head    (landmarks)
                    │   4. q̄_i = meanpool(Q_i)         per query chunk
                    │   5. s_ijh = ⟨q̄_ih, ℓ_j·kv(h)⟩ / √d_head        (scores)
                    │   6. S(i) = {i} ∪ top-(k−1) by mean_h s_ijh, j ≤ i  (causal top-k)
                    │   7. g_ijh = softmax over k selected scores     (fusion weights)
                    │   8. o_ij = SDPA(Q_i, K_j, V_j)  per selected chunk — independent
                    │   9. o_i = Σ_j g_ijh · o_ijh     (score-weighted fusion)
                    │   ▼  out_proj + residual
                    │  RMSNorm → SwiGLU (ffn 3072) + residual
                    │
              Final RMSNorm ──► head h @ Eᵀ ──► chunked cross-entropy
                    loss = CE + λ · L_bal          (λ = 0.01, per layer, averaged)
```

The block is `models/block.py:HiLSBlock` (attention sublayer + SwiGLU
sublayer, both pre-norm residual); the model is
`models/transformer.py:HiLSAttentionLM` — embedding, 24 blocks (gradient
checkpointing every 3rd in training), final RMSNorm, tied linear head.
`models/transformer.py:HiLSConfig` is the single knob surface; its defaults
are the 341M production shape and `configs/pretrain_a100_341m.yaml` writes
exactly those keys.

### 2.2 The math, precisely

For query chunk i (C query tokens), selected chunks S(i) = {j_0 = i, j_1, …,
j_{k−1}} (causal, ties toward higher index):

```
q_i   = RoPE(Q_i)                    # (H, C, d_head) after GQA grouping of K,V
ℓ_j   = W_ℓ · meanpool(K_j)          # (KV, d_head) per KV head
q̄_i   = meanpool(Q_i)                # (H, d_head)
s_ijh = ⟨q̄_ih, ℓ_j,kv(h)⟩ / √d_head  # score of chunk j for head h
S(i)  = {i} ∪ top-(k−1) mean_h s_ijh over {j : j ≤ i, j ≠ i}
g_ijh = softmax_j({s_ijh : j ∈ S(i)})           # per head over k slots
o_ih  = Σ_{m=0}^{k−1} g_ij_m h · softmax(Q_ih K_j_m^T / √d) V_j_m
```

The per-chunk softmax inside o_ij is the hierarchical factorization; g rides
on top. Rows with fewer than k admissible chunks pad by repeating the own
index with −inf fused scores — exactly zero fusion weight, still causal.

`models/attention.py:hils_attention_core` implements steps 7–9 in the
static-shape form: gather selected KV to `(B, N, k, KV, C, D)`, run k
batched SDPA calls over `(B·N, ·)`, fuse. The eager core
`models/attention.py:eager_hils_attention_core` implements the identical
math as an O(T²)
per-query-chunk loop over a full score matrix — the ground truth the
production path is proven against (fp64, atol 1e-5, `k = N` included). The
two cores are the repo's two-path convention: `attn_impl` selects the
kernel, never the math (`models/attention.py:EagerHiLSAttention` subclasses
`models/attention.py:HiLSAttention` and swaps only the core).

### 2.3 Chunking and landmarks

`models/chunking.py:validate_chunkable` enforces exact tiling (no ragged
tail — the router, the landmark meanpool, and the KV gather all index by
N = T/C, and `models/chunking.py:chunk_index_bounds` is their shared bound
helper). `models/landmarks.py:LandmarkProjector` is one 64×64 linear map per
KV head (~0.39M params total), **identity-initialized**: at step 0 a
landmark is exactly the mean of its chunk's keys — a sane compression prior
the LM loss improves from there. The router's query side,
`models/landmarks.py:pooled_query`, is the same meanpool on Q. Both are
unfold-free reshapes (chunks are contiguous tiles ⇒ meanpool is a view +
mean), and both run per forward — landmarks are cheap (one meanpool + matmul
over N·C·d) and always in sync with the keys that produced them.

RoPE is absolute and unstretched: `models/attention.py:apply_rope` rotates
Q and K at absolute positions, with the rotation table from
`models/attention.py:precompute_freqs_cis` computed for the *actual*
sequence length on every
forward — no position cache, so evaluation beyond `max_seq_len` (16K) is
just forward at 32K/64K with θ = 500K and nothing else (§9, gate B3).

## 3. The landmark router

### 3.1 Scoring and causal top-k

`models/router.py:retrieval_scores` produces the `(B, H, N, N)` score
tensor; selection is `models/router.py:select_chunks`:

- **Head-shared.** Chunks rank by the mean score over query heads; all heads
  share one S(i). The discrete indices are computed eagerly — data-dependent
  gathers break `torch.compile` (house precedent; blocks compile, routing
  does not).
- **Own chunk always** at slot 0: local continuity is never risked to the
  router. The remaining k−1 slots are best-scoring admissible chunks.
- **Ties break toward the higher chunk index** (flip + stable argsort, no
  epsilon), so selection matches the score-free reference sets of
  `models/chunking.py:causal_chunk_candidates` as exact set equality — the
  semantics `tests/test_router.py` pins.
- **Right-aligned padding.** A row with r < k−1 other admissible chunks pads
  by repeating the own index with −inf fused scores: zero fusion weight,
  still causal. This is what lets decode score **one** query chunk against
  all N cached landmarks with the same function training uses (§8).

### 3.2 Native trainability — gradients through fusion

Selection is discrete; its indices carry no gradient. The *scores* of the
selected chunks do: `models/router.py:fusion_weights` softmaxes them into
per-head fusion weights, and the block output is Σ_j g_ij · o_ij. The LM
loss therefore reaches q̄ and W_ℓ through g — the router is trained by the
LM loss through the fusion weights, which is the native-trainability claim
reproduced end-to-end. `test_retrieval_score_gradient_flow` asserts
non-None gradients on the landmark projection and the query path.

### 3.3 Load balancing and the watchdog

The failure mode is routing collapse — every query chunk reading the same
few chunks. Defense in two layers:

- **Aux loss.** `models/router.py:balance_loss` is the Herfindahl index
  N·Σ p_j² of per-chunk selection mass (= 1 at uniform, unbounded at
  collapse), computed on the *soft* fusion-weight mass (hard counts would
  block gradient), λ-weighted per layer inside the attention module and
  averaged across layers by the model forward.
- **Watchdog.** `models/router.py:selection_stats` reports entropy, max
  share, and per-layer histograms; `training/pretrain.py:train` rolls back
  when max share exceeds 50% for 500 consecutive steps. The aux term touches
  only the score path — `aux_balance_weight: 0.0` is a clean ablation flag.

## 4. Training pipeline: two phases, one schedule

One config, one cosine, one loop. `training/pretrain.py:load_config` parses
the YAML (and constructs `models/transformer.py:HiLSConfig` eagerly, so bad
model keys fail before the GPU is touched);
`training/pretrain.py:train` runs it:

| | Phase A | Phase B |
|---|---|---|
| window | 4096 (N = 32 chunks, 25% selected) | 16384 (N = 128, 6.25%) |
| micro-bs / grad-accum | 8 / 4 | 1 / 8 |
| tokens | 7.0B | 1.0B |
| steps | 53,407 | 7,630 |

Tokens per optimizer step stay at 131,072 across the switch —
`training/pretrain.py:phase_at` derives both shapes from the config and
asserts the budget divides Phase B's window. The LR is one cosine across
both phases with a 500-step re-warm tent at the switch
(`training/pretrain.py:lr_at`), because the sequence-length change is a
distribution shift for the optimizer state even though the router's
candidate space only grows. Optimizer: fused AdamW, fp32 master weights
(`training/pretrain.py:build_optimizer`), BF16 autocast on GPU, grad-clip
1.0, grad-ckpt every 3rd layer. Loop state is
`training/pretrain.py:TrainState` (step, tokens_seen, model, optimizer,
losses); checkpoints are three-file sets managed by
`utils/checkpoint.py:CheckpointManager`, and the loop auto-resumes from the
latest *complete* set.

Stability machinery (all tested, all observable in the log): NaN guard with
checkpoint rollback, the selection watchdog of §3.3, and the VRAM estimator
pair `utils/memory.py:estimate_model_memory_gb` /
`utils/memory.py:assert_fits_in_available_gpu` that fails a run before step
1 when the budget table is violated. The pod-side boundary scripts
(`scripts/microbench_a100.py`, `scripts/step_time_a100.py`,
`scripts/e2e_gpu_smoke.py`) verify the estimates and the loop mechanics
before the 40–48 h run is spent.

## 5. The loss: never materialize (B, T, V)

At micro_bs 8 / seq 4096 / vocab 50,257, a naive fp32 cross-entropy
materializes ~6.6 GB of logits — the largest single line item in the VRAM
budget and the reason `training/losses.py:chunked_lm_ce` exists. It walks
the vocab in 8192-token slices: per slice, one head GEMM, an fp32 logsumexp,
and the target logit; a custom autograd Function retains each chunk's bf16
logits so backward derives the softmax from them instead of recomputing the
GEMM. Global normalization composes the chunk logsumexps exactly, and the
pipeline is proven against plain `F.cross_entropy` over full logits
(`tests/test_loss.py`, fp32 atol 1e-6). The model's `forward(tokens,
targets)` returns `(CE + aux, aux)` — the aux term rides alongside, which is
also how the evaluator recovers pure NLL (total − aux) without a second
loss path.

## 6. Data

`data/prepare_data.py:main` wraps the workspace `shared_data` pipeline with
this repo's tokenizer contract (GPT-2 BPE, vocab 50,257, EOS/PAD 50256 —
house parity with the other Lites) and pins the data root so the pack
subprocess writes to `data/pretrain_chinchilla/shards/`. The corpus is the
same Chinchilla mixture as DiffusionGemma-Lite / Mamba-3-Lite: 8.0B tokens,
no cross-document boundaries inside a window, 50M-token uint32 shards,
shuffle seed 42. Phase B consumes the same shards at seq 16384 — windows
are longer, not different data.

## 7. Verification: the load-bearing tests

Every claim above has a test that would fail loudly if it stopped being
true. The four that guard the whole correctness story:

| test | pins | tolerance |
|---|---|---|
| `test_hils_matches_eager_reference` | sparse gather path ≡ O(T²) per-chunk-loop reference, incl. `k = N` | fp64, atol 1e-5 |
| `test_sparse_decode_matches_teacher_forced` | cached chunk-synchronous decode ≡ teacher-forced forward, every layer | fp64, atol 1e-5 |
| `test_retrieval_score_gradient_flow` | LM-loss gradients reach W_ℓ and q̄ (native trainability) | non-None |
| `test_selection_is_causal` / `test_topk_exact_semantics` | no future chunks; own-chunk-always; reference set equality | exact |

Around them: chunk tiling invariants, landmark identity-init, fusion
normalization, balance-loss collapse penalty, chunked-CE ≡ eager CE,
checkpoint round-trip + resume bit-equality, phase-switch config, NaN-guard
rollback, KV-access counter exactness, and the Phase-5 evaluator protocol
tests. The full matrix is `python3 -m pytest -m "not gpu and not slow"` —
66 tests, CPU, minutes. House rule: any change to `models/router.py` or
`models/landmarks.py` reruns the router tests *first*; any change to
attention reruns sparse ≡ eager *first*.

## 8. Sparse decode: chunk-synchronous, verbatim training math

Decode consumes **one full chunk of C tokens per step** — a chunk-synchronous
loop, not token-by-token. The reason is correctness, not convenience: a
chunk's landmark is finalized by the same step that consumes it, so every
layer's step state is *identical* to a teacher-forced forward over the same
tokens — fp64-exact, the §7.6 gate. There is no second decode math to drift:
`inference/generate.py:step_decode` reuses `models/landmarks.py:pooled_query`,
the router trio (`models/router.py:retrieval_scores`,
`models/router.py:select_chunks`, `models/router.py:fusion_weights`), and
the per-chunk SDPA factorization verbatim.

State lives in `inference/generate.py:HiLSCache`: chunk-major per-layer K/V
`(L, B, KV, N_max, C, D)`, the landmark table, and two instruments
(`kv_keys_read`, `landmark_reads`). `inference/generate.py:new_cache` sizes
it; `inference/generate.py:prefill` is literally a loop of chunk steps, so
prompt fill and generation share one code path. Per step per layer: rope the
chunk, finalize its landmark, score q̄ against the N = i+1 cached landmarks,
causal top-k (all admissible), gather k·C keys, k per-chunk SDPA calls, fuse.

Selection-level causality (j ≤ i) means **no attention mask exists anywhere
in decode** — a chunk attends its own C keys only once they are all present.

**Effective KV access.** `inference/generate.py:kv_access_fraction` is the
v1 headline instrument: bytes of K/V read per decode step ÷ full-attention
bytes at the same context — per step per layer, 2·k·C key units plus the N
landmarks, over 2·T. At 16K with k=8, C=128: 0.0625 + the landmark read ≈
0.066. The claim is *effective access*: the v1 cache physically stores all
K/V, and dropping un-read bytes is a v2 idea that would not change the
compute story (§1, point 3).

## 9. Evaluation: the four headline gates, measured

One harness — `inference/evaluate.py:LongContextEvaluator` — produces every
headline number, and tests pin its protocol on CPU before the pod runs it:
decode throughput@ctx (warmup step untimed, instruments zeroed for the
steady-state window), the KV-access fraction over that same window,
NLL-vs-length through the model's own loss path (total − aux), and
needle-in-a-haystack retrieval
(`inference/evaluate.py:Needle` rows: haystack, planted key+value, key
repeated at the row end, every value token argmax-checked;
`inference/evaluate.py:register_baseline` adds in-repo baselines — the
built-in `"all-chunk"` re-times the same weights with n_selected → N).

| gate | script | measures | criterion |
|---|---|---|---|
| B1 | `scripts/longctx_eval.py` | decode throughput @ 16K vs LLaMA-3-Lite | ≥ 1.8× (claim ~2×) |
| B2 | `scripts/longctx_eval.py` | effective KV-access fraction @ 16K | ≤ 0.08 (claim ~6%) |
| B3 | `scripts/retrieval_eval.py` | NIAH retrieval @ 64K (4× extrapolation, no RoPE stretching) | ≥ 85%, else disclose |
| B4 | `scripts/loss_parity_eval.py` | held-out ΔNLL @ 4096 vs LLaMA-3-Lite | ≤ +5%, else disclose |

The honesty contract is mechanical: every script prints its measured number
with an explicit `[PASS …]` / `[DISCLOSED …]` marker; a missing checkpoint,
baseline, tokenizer, or shard is disclosed with the reason, never silently
dropped; a measured miss is disclosed next to its gate; the scripts exit 0
either way. Flag-by-flag reference for every script:
[docs/references/eval-scripts.md](docs/references/eval-scripts.md); the
operational order for the pod work:
[docs/guides/a100-runbook.md](docs/guides/a100-runbook.md). Until the A100
pod runs them, the four numbers are **targets, not results** — the README
says exactly that, and this document claims no measured value the scripts
have not printed.

The in-repo diagnostic worth watching even before the baseline exists: the
all-chunk ratio in the `scripts/longctx_eval.py` output — what learned top-k
selection saves *with identical weights*. All-chunk selection reads N chunks
per query against our k, so the selection-work ratio N/k grows from 4× at
the 4K window to 16× at 64K; wall-clock tracks that gather+SDPA count,
bounded below by the fixed backbone cost.

## 10. Recipe deltas vs upstream

How this recipe differs from upstream HiLS (arXiv:2607.02980,
`tencent/HiLS-Attention-7B`) and from the portfolio's other Lites. Each delta
is a *choice*, and the alternative is named. The operator itself — landmarks,
causal chunk top-k, per-chunk softmax, score fusion — is upstream-faithful;
the deltas below are the Lite shell around it.

### 10.1 House block vs the OLMo3 backbone

Upstream builds on OLMo3 (QK-norm, non-parametric layer norms, MHA). This
repo uses the house LLaMA-style block — `models/block.py:HiLSBlock` with
GQA 16Q/4KV, `models/block.py:RMSNorm` (fused `F.rms_norm`, learned scale),
SwiGLU, weight tying, **no QK-norm**. Why: the claimed cell is the attention
primitive, and the house block keeps chunked-CE, gradient checkpointing, and
the eval harness drop-in across the portfolio while isolating the sparse
variable (DESIGN §1). The cost is disclosed: no QK-norm ablation is
available here, and the parity anchor absorbs the difference empirically
(gate B4), not structurally.

### 10.2 From-scratch pretrain vs converted checkpoints

Upstream's released model continues pretraining from a converted
full-attention checkpoint. This repo **pretrains from scratch** — 8.0B
tokens, init_std 0.02, `models/transformer.py:HiLSAttentionLM._init_weights`
followed by the identity-init restore for W_ℓ. Why: a converted model
inherits full-attention routing behavior, and the claim under test ("chunk
selection is learnable by the LM loss alone, from nothing") would be
unfalsifiable. Wall-clock is the honest price: ~40–48 h on one A100
(`scripts/step_time_a100.py` measures MFU before the long run is spent).

### 10.3 Scale and the extrapolation protocol

Upstream evaluates at 7B with YaRN-extended 4× baselines and an
"infinite-context" serving stack (paged/offloaded KV). Here: 341M, and
extrapolation **without any stretching** — house RoPE θ=500K,
`models/transformer.py:HiLSAttentionLM.forward` computes the rotation table
for the actual sequence length on every call, so 64K evaluation is a plain
forward (gate B3). The KV story is *effective access* measured by
instrumentation, not a serving stack; selective materialization is a
documented v2 idea, not a v1 claim.

### 10.4 The four Lite decisions inside the operator

Choices the paper does not pin down at Lite scale — each documented, each
pinned by a test in `tests/test_router.py`:

| choice | here | plausible alternative |
|---|---|---|
| selection granularity | **per query chunk** — all C queries of chunk i share S(i); static shapes, batched SDPA | per-query-token sets (dynamic shapes, no batching) |
| landmark init | **identity** — W_ℓ = I ⇒ ℓ is meanpooled K at step 0 | random-init compression |
| balance form | **Herfindahl** N·Σ p_j² on the *soft* fusion mass | entropy bonus, hard-count MoE-style load loss |
| own chunk | **always selected** at slot 0, competes in the fusion softmax like any other | pure top-k (local continuity at the router's mercy) |
| tie-breaking | **higher chunk index wins** (flip + stable argsort, no epsilon) | ε-jitter or first-index-wins |
| gradient path | **through fusion weights only**; `straight_through_selection: false` is validated fail-fast — no STE hook implemented in v1 | straight-through selection, Gumbel tricks |

### 10.5 What is deliberately NOT carried over

| upstream component | Lite decision | reason |
|---|---|---|
| 7B scale, ~50B continued-pretrain tokens | 341M from-scratch, 8.0B tokens | single-A100 Chinchilla-class budget |
| YaRN-extended 4× baseline comparisons | no stretching; extrapolation is the claim | YaRN is GPT-OSS-Lite's story |
| "infinite context" serving (paged/offloaded KV) | dense physical cache v1; effective access measured | v1 stays simple and correct; the compute win does not depend on cache bytes |
| converted-from-full-attention checkpoints | from-scratch pretraining only | house convention; conversion is upstream's deployment story |
| custom CUDA retrieval kernels | pure PyTorch gather + SDPA | house rule: opt-in kernels only; none needed at 341M |

## 11. Repository map

```
models/       chunking → landmarks → router → attention (sdpa + eager) → block → transformer
training/     losses.py (chunked CE) · pretrain.py (two-phase loop + guards)
inference/    generate.py (cache + chunk-synchronous decode) · evaluate.py (the harness)
data/         prepare_data.py — shared_data shim, GPT-2 tokenizer contract
utils/        checkpoint · logging · memory (budget table + guard)
scripts/      launch_a100.sh · microbench_a100 · step_time_a100 · e2e_gpu_smoke
              longctx_eval · retrieval_eval · loss_parity_eval · check_docs · build_docs_html
tests/        21-test core matrix + evaluator protocol + doc gate (66 total, CPU)
configs/      pretrain_a100_341m.yaml — the one production config
docs/         concepts · guides · references (symbol-anchored, gate-checked)
```

Reading order for a new contributor: this file →
[docs/concepts/hils-routing.md](docs/concepts/hils-routing.md) →
[docs/concepts/hierarchical-attention.md](docs/concepts/hierarchical-attention.md)
→ [docs/concepts/long-context-phases.md](docs/concepts/long-context-phases.md)
→ [docs/references/api.md](docs/references/api.md) as the index into source.
When a run misbehaves: [docs/guides/debugging-playbook.md](docs/guides/debugging-playbook.md)
symptom-first; when a config knob is unclear:
[docs/references/config.md](docs/references/config.md).

## 12. References

- HiLS: learned sparse chunk attention — arXiv:2607.02980 (Tencent, Jul 2026);
  released checkpoint `tencent/HiLS-Attention-7B`
- [`llm-research/DESIGN-hils-attention-lite.md`](../../llm-research/DESIGN-hils-attention-lite.md)
  — architecture + rationale (§2 operator, §3 config, §4 pipeline, §6
  trade-offs, §12 the Lite-decision ledger)
- [`llm-research/EXECUTION-PLAN-hils-attention-lite.md`](../../llm-research/EXECUTION-PLAN-hils-attention-lite.md)
  — the mechanical build contract; §8 the final verification matrix
- `LLM/LLaMA-3-Lite/` — the 515M full-attention GQA baseline (matched
  harness; budget delta disclosed in every comparison)
- `LLM/DiffusionGemma-Lite/` — house plan/doc/test conventions, chunked-CE
  lineage, watchdog pattern
- `shared_data/` — 50M-token uint32 shards, GPT-2 tokenizer, seed 42
- Portfolio non-overlap: MTP, MLA, GDN, MoE, sinks, SSM, YaRN, and diffusion
  belong to other cells — this repo's cell is learned sparse routing only
  (AGENTS.md rule 5).