# Learning paths — how to read the HiLS-Attention-Lite docs

Three reading paths. Each row is one step: the doc to read and what you will
know after reading it. Every cited symbol is machine-checked against the
working tree by `scripts/check_docs.py`.

## Beginner — what is this model?

| Step | Doc | What you will know after |
|---|---|---|
| 1 | [HILS.md](../../HILS.md) §0–§2 | The one-paragraph idea: attention restricted to a few chunks per query, selected by landmark scores, outputs fused by the same scores — and why that is trainable end-to-end. |
| 2 | [HiLS routing](../concepts/hils-routing.md) | How a sequence becomes chunks, what a landmark is, how causal top-k selection works, and what the router deliberately does not do. |
| 3 | [Hierarchical attention](../concepts/hierarchical-attention.md) | The per-chunk softmax operator, why the fast and eager paths are the same math, and how RoPE is applied. |
| 4 | [Quickstart](quickstart.md) | How to run the CPU smoke, build data, and drive every command end-to-end. |

## Intermediate — train it

| Step | Doc | What you will know after |
|---|---|---|
| 1 | [Long-context phases](../concepts/long-context-phases.md) | The two-phase plan (7B tokens @ seq 4096, then 1B @ seq 16384) and chunk-synchronous decode. |
| 2 | [Training pipeline](../training.md) | The applied loop as it exists in `training/pretrain.py:train` — schedule, guardrails, checkpointing, memory preflight, loss path. |
| 3 | [Config reference](../references/config.md) | Every field of the production YAML and why the shapes are what they are. |
| 4 | [A100 pod runbook](a100-runbook.md) | Boundary checks, the long run, resume semantics, and what to watch. |

## Expert — verify and optimize it

| Step | Doc | What you will know after |
|---|---|---|
| 1 | [Training pipeline](../training.md) — loss + guardrail sections | Where the aux balance loss is wired (`models/transformer.py:HiLSAttentionLM.forward`) and how the NaN guard and selection watchdog roll back via `utils/checkpoint.py:CheckpointManager`. |
| 2 | [HILS.md](../../HILS.md) §7 | The load-bearing equivalence tests: fast vs eager attention, `k = N` selecting every chunk, sparse decode matching teacher forcing. |
| 3 | [Eval-scripts reference](../references/eval-scripts.md) | Every gate script (longctx, retrieval, loss parity, microbench, step time), its flags, and PASS/DISCLOSED semantics. |
| 4 | [Debugging playbook](debugging-playbook.md) | Symptom-first recipes for router collapse, NaN streaks, and phase-switch spikes. |
| 5 | [API reference](../references/api.md) | The full public surface, symbol-anchored, module by module. |

After any path: [Glossary](glossary.md) resolves notation and config keys;
[AUDIT.md](../AUDIT.md) records the current verification state of the whole
corpus.
