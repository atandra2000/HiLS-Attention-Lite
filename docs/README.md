# HiLS-Attention-Lite — Documentation

Deep-dive and reference docs for the from-scratch HiLS learned sparse chunk
attention reproduction. Start with [HILS.md](../HILS.md) (the authoritative
technical doc), then the concept pages; guides are task-first; references pin
the config and the public API.

Conventions: every `file.py:Symbol` anchor in these pages is machine-checked
against the working tree by `scripts/check_docs.py` — the anchor names the
symbol `Symbol` in `file.py`, and line numbers are never cited (they rot).
`models/attention.py:HiLSAttention` therefore means "read that class."

## Visual atlas

Start with the [interactive visual guide](hils_visual_guide.html): three Archify
maps, a routing experiment, a learning-rate explorer and source-grounded
corrections to the unfinished draft. [Verification receipts](hils_receipts.json).

## Concepts

- [HiLS routing](concepts/hils-routing.md) — chunking, landmarks, causal
  top-k, score fusion, native trainability, load balancing
- [Hierarchical attention](concepts/hierarchical-attention.md) — the
  per-chunk softmax operator, the two implementation paths, RoPE
- [Long-context phases](concepts/long-context-phases.md) — 7B @ 4096 then
  1B @ 16K, chunk-synchronous decode, the evaluation harness

## Guides

- [Master guide](guides/master-guide.md) — comprehensive overview of architecture,
  data pipeline, two-phase pretraining, and systems optimizations
- [Quickstart](guides/quickstart.md) — data → train → sample → evaluate,
  every command verified end-to-end
- [A100 pod runbook](guides/a100-runbook.md) — boundary checks → the 40–48 h
  pretrain (resume semantics, what to watch) → the headline evals
- [Debugging playbook](guides/debugging-playbook.md) — symptom-first recipes
  for the known failure modes (router collapse, NaN, phase-switch spikes)

## References

- [Config reference](references/config.md) — the production YAML, field by field
- [Eval-scripts reference](references/eval-scripts.md) — every gate script:
  flags, gates, CPU vs A100 forms, PASS/DISCLOSED semantics
- [API reference](references/api.md) — the full public surface, symbol-anchored

## Tooling

- `scripts/check_docs.py` — anchor resolution + symbol coverage (CI gate)
- `scripts/build_docs_html.py` — builds the `docs_html/` portal (gitignored)
- `python3 -m pytest -m "not gpu and not slow"` — the CPU gate every change passes