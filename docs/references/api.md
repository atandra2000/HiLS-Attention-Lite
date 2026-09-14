# Reference: API — the public surface, symbol-anchored

Every public symbol of the importable packages, one anchor per row. Anchors
are machine-checked by `scripts/check_docs.py --coverage` (CI): each
`file.py:Symbol` must resolve against the working tree, and every public
symbol must appear at least once across the docs.

## models/ — chunking, landmarks, router, attention, block, model

| anchor | what it is |
|---|---|
| `models/chunking.py:validate_chunkable` | exact-tile assertion: positive lengths, `seq_len % chunk_len == 0` |
| `models/chunking.py:chunk_index_bounds` | half-open valid chunk range `(0, N)` for a tileable sequence |
| `models/chunking.py:causal_chunk_candidates` | score-free reference candidate sets `[i, i-1, …][:k]` — what top-k must reproduce |
| `models/landmarks.py:LandmarkProjector` | per-layer, per-KV-head linear map (identity init): ℓ_j = W_ℓ · meanpool(K_j) |
| `models/landmarks.py:pooled_query` | chunk-query estimate q̄_i = meanpool(Q_i) for the router |
| `models/router.py:retrieval_scores` | s_ijh = ⟨q̄_ih, ℓ_j·kv(h)⟩/√d_head → (B, H, N, N) |
| `models/router.py:select_chunks` | causal head-shared top-k: own chunk at slot 0, ties → higher index, right-aligned padding |
| `models/router.py:fusion_weights` | per-head softmax over the k selected scores; −inf pads → exactly 0 |
| `models/router.py:balance_loss` | Herfindahl load balance L_bal = N·Σ p_j² (1 at uniform, unbounded at collapse) |
| `models/router.py:selection_stats` | {entropy, max_share, per_layer} for the collapse watchdog |
| `models/attention.py:apply_rope` | absolute-position RoPE; trig in ≥ fp32, cast back |
| `models/attention.py:precompute_freqs_cis` | Llama-style complex rotation table (max_seq_len, D/2) |
| `models/attention.py:hils_attention_core` | production core: gather selected KV → k SDPA calls → fuse by g |
| `models/attention.py:eager_hils_attention_core` | O(T²) ground truth: full score matrix, per-chunk softmax by masking |
| `models/attention.py:HiLSAttention` | the module: rope → landmarks → scores → top-k → core; stashes `last_selected` |
| `models/attention.py:EagerHiLSAttention` | identical math via the eager core (`attn_impl="eager"`) |
| `models/block.py:RMSNorm` | fused `F.rms_norm`, learned scale, no bias |
| `models/block.py:HiLSBlock` | RMSNorm → attention (+residual) → RMSNorm → SwiGLU (+residual); reports λ·L_bal |
| `models/transformer.py:HiLSConfig` | every model knob; `from_yaml` reads the `model:` section |
| `models/transformer.py:HiLSAttentionLM` | embed → 24×block → final norm → tied head; forward(tokens) → logits, forward(tokens, targets) → (loss, aux) |

## training/ — objective + loop

| anchor | what it is |
|---|---|
| `training/losses.py:chunked_lm_ce` | vocab-chunked LM cross-entropy; bounds the fp32 CE peak (retained bf16 logits still total O(B·T·V)) |
| `training/pretrain.py:load_config` | parse the YAML; validate the model section eagerly |
| `training/pretrain.py:build_optimizer` | fused AdamW, fp32 master weights, decay ≥2-D params |
| `training/pretrain.py:lr_at` | warmup → cosine + 500-step re-warm tent at the switch |
| `training/pretrain.py:phase_at` | (seq_len, micro_bs, grad_accum) per step; token budget constant |
| `training/pretrain.py:TrainState` | step / tokens_seen / model / optimizer / losses |
| `training/pretrain.py:train` | the loop: autocast, guards, watchdog, checkpoints; test seams `batches=` / `max_steps=` |

## data/, inference/ — corpus, decode, evaluation

| anchor | what it is |
|---|---|
| `data/prepare_data.py:main` | shared_data pipeline entry with the GPT-2 tokenizer contract |
| `inference/generate.py:HiLSCache` | chunk-major K/V + landmark cache + KV-access counters |
| `inference/generate.py:new_cache` | empty cache sized for max_seq tokens |
| `inference/generate.py:prefill` | chunk-aligned prompt fill — a loop of chunk steps |
| `inference/generate.py:step_decode` | one chunk-synchronous decode step; selection reuses the training router |
| `inference/generate.py:kv_access_fraction` | effective KV bytes/step ÷ full-attention bytes (the ~6% @ 16K instrument) |
| `inference/evaluate.py:LongContextEvaluator` | the one harness: throughput@ctx, KV fraction, NLL-vs-length, retrieval |
| `inference/evaluate.py:Needle` | id-level (key, value, filler pool, depth) for needle-in-a-haystack rows |
| `inference/evaluate.py:register_baseline` | registry hook for in-repo baselines (built-in: `"all-chunk"`) |

## utils/ — checkpointing, logging, memory

| anchor | what it is |
|---|---|
| `utils/checkpoint.py:CheckpointManager` | three-file checkpoint sets; `latest_step` only counts complete sets |
| `utils/logging.py:TrainingLogger` | interval loss/ppl/tokens-per-sec; optional WandB via `WANDB_PROJECT` |
| `utils/memory.py:estimate_model_memory_gb` | peak-GB estimate: params + AdamW master + activations + CE chain |
| `utils/memory.py:assert_fits_in_available_gpu` | fail-fast VRAM guard (2 GB margin) |

## Scripts (pod-side, all PASS/DISCLOSED)

`scripts/longctx_eval.py` (B1+B2), `scripts/retrieval_eval.py` (B3),
`scripts/loss_parity_eval.py` (B4), `scripts/microbench_a100.py` (B5),
`scripts/step_time_a100.py` (B6), `scripts/e2e_gpu_smoke.py` (production-loop
smoke), `scripts/launch_a100.sh` (the one-command pretrain). Doc tooling:
`scripts/check_docs.py` (this gate) and `scripts/build_docs_html.py`
(the portal).