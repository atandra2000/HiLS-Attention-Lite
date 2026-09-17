# HiLS visual master guide

Open the [interactive implementation atlas](../hils_visual_guide.html) for the
complete architecture, routing laboratory, data pipeline, training schedule,
optimization ledger and cached-decode explanation.

- [Model and routing map](../hils_architecture.html)
- [Corpus and training-window map](../hils_dataflow.html)
- [Two-phase training workflow](../hils_workflow.html)
- [Validation and browser receipts](../hils_receipts.json)

## Source map

| Topic | Authoritative implementation |
|---|---|
| Dense backbone and training/inference outputs | `models/transformer.py:HiLSAttentionLM`, `models/transformer.py:HiLSConfig`, `models/block.py:HiLSBlock` |
| Position encoding and sparse operator | `models/attention.py:apply_rope`, `models/attention.py:precompute_freqs_cis`, `models/attention.py:hils_attention_core`, `models/attention.py:eager_hils_attention_core` |
| Summaries and routing | `models/landmarks.py:LandmarkProjector`, `models/landmarks.py:pooled_query`, `models/router.py:retrieval_scores`, `models/router.py:select_chunks`, `models/router.py:fusion_weights` |
| Balance and monitoring | `models/router.py:balance_loss`, `models/router.py:selection_stats` |
| Data preparation | `data/prepare_data.py:main`; workspace shared_data/loader.py owns the active consumer |
| Training, phases and LR | `training/pretrain.py:train`, `training/pretrain.py:load_config`, `training/pretrain.py:phase_at`, `training/pretrain.py:lr_at`, `training/pretrain.py:build_optimizer` |
| Vocabulary loss | `training/losses.py:chunked_lm_ce` |
| Persistence | `utils/checkpoint.py:CheckpointManager` |
| Cached evaluation | `inference/generate.py:HiLSCache`, `inference/generate.py:step_decode`, `inference/generate.py:prefill`, `inference/generate.py:kv_access_fraction`, `inference/evaluate.py:LongContextEvaluator` |

## Corrections to the unfinished draft

The visual guide describes current source; older design prose is not evidence of
implemented behavior.

1. **Chunk causality is not token causality.** Routing masks future chunks, but
   SDPA has no within-chunk causal mask. The query pool and own landmark also
   include the full current chunk. A tiny FP64 probe changes a later token and
   observes a change in earlier logits. Cached/full parity tests compare this
   same teacher-forced operator; they do not establish autoregressive correctness.
2. **Chunked CE retains all chunk logits for backward.** Each vocabulary slice
   saves its logits. This avoids one monolithic tensor and bounds temporary
   per-slice work, but retained logits still total O(BTV). The previous 6.6 GB
   elimination claim was unsupported.
3. **The active shared loader copies the corpus into RAM.** Shards are mapped,
   then copied to a flat uint32 array: about 32 GB for 8B tokens before other
   allocations. PackedDataset slices T+1 positions and does not enforce isolated
   document windows or construct a document attention mask.
4. **Checkpoints are three direct-write files.** Existence checks are not atomic
   completion or corruption validation. No RNG or sampler/data cursor is saved.
   Resume bit-equality in a controlled test does not prove arbitrary replay.
5. **Sparse reads do not shrink the physical cache.** The default cache stores
   the full K/V history. A full-prefix analytical read estimate is 6.640625% at
   16K including landmarks; measured throughput and retrieval remain open gates.
6. **Checkpointing covers every third block.** Indices 0,3,...,21 are recomputed,
   not all 24 blocks. Head-shared selection and per-head fusion are distinct.

## Reproduce the checks

From the project root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -m "not gpu and not slow" -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 python3 scripts/check_docs.py --coverage --links
node docs/check-hils-guide.mjs docs "$JCODE_SCRATCH_DIR"
python3 scripts/build_docs_html.py
```

The browser check uses an installed Chrome and Node's standard library. It opens
local files with the network disabled, measures desktop/mobile containment,
checks routing causality/padding/ties and head-specific weights, tests the LR
schedule and keyboard input, and records screenshots and an artifact hash.

Diagram specs are `hils_architecture.json`, `hils_dataflow.json` and
`hils_workflow.json`. Archify `deliver` records exact specification/HTML
hashes and nine showcase checks. `visual-check` separately measures the real
browser at four desktop sizes and captures both endpoint themes. Screenshot
inspection is recorded independently in the combined receipt.

The 68-test CPU baseline passes. A100 throughput, training duration, retrieval,
NLL parity and peak-memory gates are not measured in this documentation task.
The causal contract needs review before interpreting next-token quality results.
Model/training code and the pre-existing loss-parity-script edit are preserved.

## Current quality-review pass

See [pinned source and limits](../quality-review.md). All guides use Overview → Model → Data → Training → Distinctive mechanism → Evidence. Prior perceptual approval does not apply to refreshed artifacts. The proposed 11/12px target remains unmet.
