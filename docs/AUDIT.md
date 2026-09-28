# HiLS-Attention-Lite — Documentation & Codebase Audit

> **Scope.** The full documentation corpus (`docs/`, `README.md`, `HILS.md`,
> `AGENTS.md`, `SKILLS.md`) against the working tree. **Date: 2026-09-21.**
> Configuration pinned: `configs/pretrain_a100_341m.yaml`.

## Verification runs (2026-09-21, this machine)

| Command | Result |
|---|---|
| `python3 scripts/check_docs.py --coverage --links` | resolution **PASS**, coverage **PASS** (0 uncovered), line anchors **PASS**, links **PASS**, exit 0 |
| `python3 -m pytest tests/test_doc_refs.py -q` | all passed, exit 0 |
| `python3 -m pytest -m "not gpu and not slow" -q` | full CPU suite, exit 0 |

Anchor counts and corpus sizes below are measured (`wc -w`), not estimated.

**Corpus size** — measured 2026-09-21, 16 files / 11.3K words total
(quality-review.md and AUDIT.md are meta artifacts):

| Track | Files | ~Words |
|---|---|---|
| concepts/ | 3 | ~1,970 |
| guides/ (6) | 6 | ~3,870 |
| references/ | 3 | ~2,850 |
| training.md | 1 | ~1,150 |
| nav + meta | 3 | ~1,510 |

## 1. State of the docs — what is already excellent

- **Machine-enforced honesty.** Every `file.py:Symbol` anchor in the corpus is
  resolved by import + hasattr against the working tree; line-number citations
  are banned; coverage mode requires every public symbol in the 14
  production modules to be cited at least once. Stale docs fail the gate.
- **PASS/DISCLOSED discipline** in the [eval-scripts reference](references/eval-scripts.md):
  scripts either print a gate verdict or explicitly disclose that the A100
  form did not run. No invented headline numbers.
- **Four tracks are complete** (concepts, references, guides, applied
  training.md) with a nav map that lists every entry.
- **Equivalence-first verification culture**: fast vs eager attention parity,
  `k = N` selects every chunk, sparse decode vs teacher forcing — the
  load-bearing tests are documented, not folklore ([HILS.md](../HILS.md) §7).

## 2. Findings

| ID | Severity | Finding | Evidence | Status |
|---|---|---|---|---|
| F1 | medium | Nav map did not list learning-paths, glossary, training.md, AUDIT.md (the latter two did not exist) | docs/README.md before this pass | **Fixed 2026-09-21** — both docs written, nav updated |
| F2 | low | `docs/quality-review.md` sits outside the four tracks | it is a diagram-receipt artifact (specs + source hashes), not reader documentation | **Accepted** — meta artifact; recorded here so it is a decision, not an omission |
| F3 | low | A100-dependent numbers (wall-clock, MFU, peak VRAM) are targets or config-derived, not run receipts | config header comments; PASS/DISCLOSED semantics | **Open** — update from receipts after the pod run; keep tagged until then |
| F4 | low | Corpus depth is the smallest among the portfolio's deep-doc repos | measured: ~11.3K words vs 70K+ for the flagship | **Accepted** — depth-proportional standard (workspace DOC_UPGRADE_PLAN §1.4); grow with post-run measurements, do not pad |

No docs↔code misalignment survived the gate: all anchors resolve, all
coverage-list symbols are cited.

## 3. From-scratch codebase explanation

HiLS-Attention-Lite is a dense pre-LN transformer whose attention is **learned
sparse chunk attention**: the sequence is cut into chunks of 128 tokens; each
query chunk scores all earlier chunks by comparing a pooled query against a
per-chunk landmark (a compressed key summary, identity-initialized); it
attends to its own chunk plus the top-7 scored chunks, and the selected
chunks' softmax outputs are fused by the same scores. Because fusion weights
are differentiable, the router trains end-to-end without reinforcement-style
tricks; a Herfindahl balance term plus a watchdog defend against collapse.

| Layer | Symbols | Read |
|---|---|---|
| routing | `models/chunking.py:causal_chunk_candidates`, `models/landmarks.py:LandmarkProjector`, `models/router.py:select_chunks` | [HiLS routing](concepts/hils-routing.md) |
| attention | `models/attention.py:HiLSAttention`, `models/attention.py:hils_attention_core`, `models/attention.py:eager_hils_attention_core` | [Hierarchical attention](concepts/hierarchical-attention.md) |
| model | `models/block.py:HiLSBlock`, `models/transformer.py:HiLSAttentionLM`, `models/transformer.py:HiLSConfig` | [API reference](references/api.md) |
| objective | `training/losses.py:chunked_lm_ce` | [HILS.md](../HILS.md) §5 |
| loop | `training/pretrain.py:train`, `training/pretrain.py:phase_at`, `training/pretrain.py:lr_at` | [Training pipeline](training.md) |
| decode/eval | `inference/generate.py:step_decode`, `inference/evaluate.py:LongContextEvaluator` | [Long-context phases](concepts/long-context-phases.md) |

## 4. Modification plan (priority order)

1. **After the A100 run**: replace every target/DISCLOSED number with the
   measured receipt; update [config reference](references/config.md) VRAM
   section from the microbench output. Closes F3.
2. **Post-run measurement docs**: a measured step-time/MFU section in
   [training.md](training.md) (§1 table), sourced from
   `scripts/step_time_a100.py` receipts — not before.
3. **Keep coverage at 0 gaps**: any new public symbol lands together with its
   first citation; the gate enforces it in CI.
4. **Do not pad** toward peer corpus sizes; grow tracks only with
   code-keyed content that cites real symbols (F4 discipline).

## 5. Acceptance criteria for "audit complete"

- [x] All three verification runs pass on the audit date (see table above).
- [x] Coverage: 0 uncited public symbols in the 14 production modules.
- [x] Nav map lists every doc in all four tracks.
- [x] No untagged estimate in any doc: A100-dependent numbers carry
      target/DISCLOSED framing until receipts exist.
- [x] Findings table has an owner-status for every row (fixed / accepted / open).

Re-audit trigger: the A100 run completing, or any `COVERAGE_MODULES` change.
