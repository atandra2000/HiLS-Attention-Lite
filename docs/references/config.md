# Reference: config — the production YAML

`configs/pretrain_a100_341m.yaml` is the one config: the model section
constructs `models/transformer.py:HiLSConfig` verbatim, the training section
drives `training/pretrain.py:train`, parsed by
`training/pretrain.py:load_config` (which validates the model keys eagerly —
an unknown or missing key fails before any GPU is touched).

## `model:` — architecture (models/transformer.py:HiLSConfig fields)

| field | value | note |
|---|---|---|
| `vocab_size` | 50257 | GPT-2 BPE, house parity with `shared_data` |
| `d_model` / `n_layers` | 1024 / 24 | house backbone; all novelty lives in attention |
| `n_heads` / `n_kv_heads` | 16 / 4 | GQA; `head_dim 64` ⇒ `d_model = heads × head_dim` |
| `ffn_dim` | 3072 | SwiGLU intermediate (3× d_model) |
| `weight_tying` | true | embed ↔ head (head is `h @ Eᵀ`) |
| `rms_norm_eps` / `init_std` | 1e-5 / 0.02 | house block norms; N(0, 0.02²) init |
| `rope_theta` | 500000 | absolute RoPE, **no stretching** — extrapolation is the claim |
| `max_seq_len` | 16384 | Phase B ceiling; forward beyond it extrapolates |
| `attn_impl` | "sdpa" | production gather path; `"eager"` = O(T²) reference |
| `chunk_len` | 128 | C: 32 chunks @ 4096, 128 @ 16K — exact tiling enforced |
| `n_selected` | 8 | k: own chunk + top-7 by landmark score |
| `landmark_init` | "identity" | W_ℓ = I ⇒ ℓ starts as the meanpooled K |
| `fusion` | "score_softmax" | per-head softmax over the k selected scores |
| `selection_scope` | "per_query_chunk" | all C queries of a chunk share S(i) |
| `aux_balance_weight` | 0.01 | λ for the Herfindahl load-balance term |
| `straight_through_selection` | false | documented ablation hook, off in v1 |

Invariants the constructor enforces: `attn_impl ∈ {sdpa, eager}`,
`n_heads · head_dim == d_model`, and identity-init is *restored* after the
normal-init pass so W_ℓ = I at step 0 regardless of `init_std`.

## `training:` — the loop

| field | value | consumer |
|---|---|---|
| `micro_batch_size` / `gradient_accumulation_steps` | 8 / 4 | Phase A shape via `training/pretrain.py:phase_at` |
| `total_steps` | 61037 | 53,407 (A) + 7,630 (B) |
| `warmup_steps` | 2000 | `training/pretrain.py:lr_at` (first step trains: `(step+1)/warmup`) |
| `phase_switch_step` | 53407 | the A→B flip; one cosine + 500-step re-warm tent |
| `phase_a_seq_len` | 4096 | Phase B uses `model.max_seq_len` |
| `phase_b_warmup_steps` | 500 | re-warm tent width |
| `lr` / `min_lr_ratio` | 3e-4 / 0.05 | cosine floor = 1.5e-5 |
| `weight_decay` / `beta1` / `beta2` | 0.1 / 0.9 / 0.95 | fused AdamW (`training/pretrain.py:build_optimizer`; decay ≥2-D params only) |
| `grad_clip` | 1.0 | global norm, every optimizer step |
| `grad_checkpoint` / `grad_checkpoint_every` | true / 3 | every 3rd layer; `1 + 1/(3E)` recompute factor in MFU |
| `compile` / `compile_mode` | true / max-autotune | blocks only — selection stays eager |
| `save_interval` / `log_interval` | 4000 / 50 | `utils/checkpoint.py:CheckpointManager` / `utils/logging.py:TrainingLogger` |
| `nan_guard` / `nan_guard_max_consecutive` | true / 5 | rollback after 5 bad micro-batches |
| `selection_watchdog` | true | max-share > 50% for 500 steps → rollback |
| `save_dir` | checkpoints/pretrain_a100 | three-file checkpoint sets |

Tokens per optimizer step: `seq_a · ms_a · accum_a = 131,072`; Phase B derives
`(16384, 1, 8)` from the same budget (`training/pretrain.py:phase_at` asserts
divisibility).

## `data:` — corpus + tokenizer contract

| field | value | note |
|---|---|---|
| `train_data_path` | data/pretrain_chinchilla | packed shards from `data/prepare_data.py:main` |
| `tokenizer` | gpt2 | 50,257 vocab; EOS/PAD 50256 |
| `shard_size_tokens` | 50,000,000 | uint32 shards |
| `max_tokens` / `long_context_tokens` | 8.0B / 1.0B | Phase A 7.0B + Phase B 1.0B |
| `data_mix` | hils-default | same shared_data mixture as the other Lites |

## VRAM budget (why the shapes are what they are)

`utils/memory.py:estimate_model_memory_gb` encodes the budget: fp32 AdamW
master (12 bytes/param), boundary activations under every-3rd-layer
checkpointing, and the chunked-CE chain (retained bf16 chunk logits + one
transient fp32 slice). `utils/memory.py:assert_fits_in_available_gpu` fails
the run before step 1 when the estimate leaves < 2 GB margin; the measured
allocator peak is cross-checked by `scripts/microbench_a100.py` (gates:
< 15 GB Phase A, < 20 GB Phase B).