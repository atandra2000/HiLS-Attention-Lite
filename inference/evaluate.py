"""Long-context evaluation harness (Phase 5.1, plan §5.1).

LongContextEvaluator is the single measurement harness behind the headline
scripts (scripts/longctx_eval.py, scripts/retrieval_eval.py,
scripts/loss_parity_eval.py): one protocol per quantity, so the A100 headline
runs (B1–B4) measure exactly what tests/test_evaluate.py pinned on CPU.

Quantities (all measured, never claimed — AGENTS.md rule 4):

- decode throughput @ ctx: chunk-synchronous decode through
  inference/generate.py:prefill + step_decode — tokens/sec over the timed
  window after an untimed warmup step. Cache instruments are zeroed before
  the window, so the KV fraction is steady-state decode, not prefill.
- effective KV-access fraction: inference/generate.py:kv_access_fraction
  over the same window. The ≤0.08 @ 16K claim is *effective access* — the v1
  cache physically stores all K/V (DESIGN §2.6).
- NLL vs length: teacher-forced over T+1 token ids through the model's own
  loss path — total − aux from models/transformer.py:HiLSAttentionLM.forward
  (the chunked-CE objective; a full-vocab logits tensor is never built).
  Ids come from `token_stream` directly, or from `texts` via the
  constructor's tokenizer (needs only `.encode(str) -> list[int]`).
- retrieval accuracy: batched needle-in-a-haystack. Each row is
  [haystack][key][value][haystack][key][value] at exactly T tokens: the
  needle's key is repeated at the row end and every value position must be
  argmax-correct — multi-token values are teacher-forced by construction
  (the full value span follows the repeated key, so each checked position
  predicts the next value token). Rows sharing a needle shape batch into one
  forward; the head applies only at the checked positions (a full (B, T, V)
  tensor at 64K would be ~6.6 GB/row).

Readout: _read_hidden mirrors the block loop of
models/transformer.py:HiLSAttentionLM.forward (no aux, and no checkpointing
under no_grad) and returns post-final-norm hidden states; tests pin the
mirror to the model's own forward.

Baselines: `baselines=["all-chunk"]` re-times the same weights with
n_selected → N (read-every-chunk) through the identical harness — the in-repo
diagnostic for what learned top-k selection saves. Cross-repo baselines
(LLaMA-3-Lite) belong to the scripts, which print measured numbers with
explicit PASS/DISCLOSED markers. New baselines register via register_baseline.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import Tensor

from inference.generate import kv_access_fraction, new_cache, prefill, step_decode


@dataclass
class Needle:
    """One planted (key, value) association for needle-in-a-haystack eval.

    `key_ids` both identifies the needle inside the haystack and is the query
    repeated at the row end; `value_ids` is the span the final positions must
    regenerate (checked token-by-token). `filler_ids` cycles as haystack text
    when given, else the haystack is seeded uniform token ids. `depth` places
    the needle (0 = top, 1 = bottom of the haystack)."""

    key_ids: list[int]
    value_ids: list[int]
    filler_ids: list[int] | None = None
    depth: float = 0.5


BASELINE_FACTORIES: dict = {}


def register_baseline(name: str, fn) -> None:
    """Register fn(evaluator, lengths, n_steps, warmup) → {"throughput": {T: tps}}."""
    BASELINE_FACTORIES[name] = fn


@torch.no_grad()
def _read_hidden(model, tokens: Tensor) -> Tensor:
    """Hidden states after final_norm, mirroring
    models/transformer.py:HiLSAttentionLM.forward's block loop without the
    head, so positional readout avoids the (B, T, V) logits tensor at 64K."""
    freqs = model._freqs_cis(tokens.size(1), tokens.device)
    h = model.embed(tokens)
    blocks = model._fast_blocks if model._fast_blocks is not None else model.blocks
    for block in blocks:
        h, _ = block(h, freqs)  # aux discarded — measurement, not training
    return model.final_norm(h)

class LongContextEvaluator:
    """Throughput@ctx, KV-access fraction, NLL-vs-length, retrieval accuracy.

    plan §5.1 contract: `__init__(model, tokenizer)` and
    `evaluate(ctx_lengths, baselines=None)`. The keyword-only options
    (`texts`, `token_stream`, `needles`, `n_steps`, `warmup`, `seed`,
    `retrieval_batch`) extend that contract with defaults, so the documented
    two-argument calls behave exactly as specified. Requested lengths snap
    DOWN to chunk multiples (HiLS tiles exactly —
    models/chunking.py:validate_chunkable) and must leave room inside the
    cache for prefill ≥ C plus the warmup + timed chunks."""

    def __init__(self, model, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

    def evaluate(self, ctx_lengths: list[int],
                 baselines: list[str] | None = None, *,
                 texts: list[str] | None = None,
                 token_stream: Tensor | None = None,
                 needles: list[Needle] | None = None,
                 n_steps: int = 3, warmup: int = 1,
                 retrieval_batch: int = 8, seed: int = 0) -> dict:
        """Run the requested measurements; every length key in the result is
        the snapped chunk multiple actually evaluated."""
        if n_steps < 1 or warmup < 0:
            raise ValueError("need n_steps >= 1 and warmup >= 0")
        lengths = self._snap_lengths(ctx_lengths, n_steps, warmup)
        result: dict = {"model": type(self.model).__name__,
                        "throughput": {}, "kv_fraction": {}}
        for T in lengths:
            tps, frac = self._measure_decode(T, n_steps=n_steps, warmup=warmup)
            result["throughput"][T] = tps
            result["kv_fraction"][T] = frac
        if baselines:
            result["baselines"] = {}
            for name in baselines:
                fn = BASELINE_FACTORIES.get(name)
                if fn is None:
                    raise ValueError(f"unknown baseline {name!r}; "
                                     f"registered: {sorted(BASELINE_FACTORIES)}")
                result["baselines"][name] = fn(self, lengths, n_steps, warmup)
        if texts is not None or token_stream is not None:
            result["nll"] = {T: self._nll_at(T, texts, token_stream) for T in lengths}
        if needles is not None:
            result["retrieval"], result["retrieval_detail"] = {}, []
            for T in lengths:
                acc, detail = self._retrieval(T, needles, retrieval_batch, seed)
                result["retrieval"][T] = acc
                result["retrieval_detail"] += detail
        return result

    def _snap_lengths(self, ctx_lengths, n_steps: int, warmup: int) -> list[int]:
        C = self.model.cfg.chunk_len
        min_T = C * (1 + warmup + n_steps)  # prefill ≥ C plus warmup + timed chunks
        out: list[int] = []
        for L in ctx_lengths:
            T = (int(L) // C) * C
            if T < min_T:
                raise ValueError(
                    f"ctx {L} (snapped {T}) is below the minimum {min_T} = "
                    f"chunk_len·(1 + warmup {warmup} + n_steps {n_steps})")
            if T not in out:
                out.append(T)
        return out

    @torch.no_grad()
    def _measure_decode(self, T: int, n_steps: int, warmup: int) -> tuple[float, float]:
        """Chunk-synchronous decode at ctx T → (tokens/sec, KV-access fraction).

        Prefills T − (warmup + n_steps)·C tokens, runs the warmup step, zeroes
        the cache instruments, then times n_steps one-chunk
        inference/generate.py:step_decode calls (C tokens each)."""
        model = self.model
        C = model.cfg.chunk_len
        device = next(model.parameters()).device
        g = torch.Generator().manual_seed(0)  # fixed token content; timing is the quantity
        toks = torch.randint(0, model.cfg.vocab_size, (1, T), generator=g).to(device)
        cache = new_cache(model, batch=1, max_seq=T)
        pos = T - C * (warmup + n_steps)
        prefill(model, cache, toks[:, :pos])
        for _ in range(warmup):
            step_decode(model, cache, toks[:, pos:pos + C], pos)
            pos += C
        cache.kv_keys_read = 0
        cache.landmark_reads = 0
        cache.steps = 0
        self._sync(device)
        t0 = time.perf_counter()
        for _ in range(n_steps):
            step_decode(model, cache, toks[:, pos:pos + C], pos)
            pos += C
        self._sync(device)
        dt = max(time.perf_counter() - t0, 1e-9)
        return C * n_steps / dt, kv_access_fraction(cache)

    @torch.no_grad()
    def _nll_at(self, T: int, texts: list[str] | None,
                token_stream: Tensor | None) -> float:
        """Teacher-forced NLL/token at ctx T: total − aux — the model's own
        loss path minus the aux term (both returned separately by
        models/transformer.py:HiLSAttentionLM.forward)."""
        if token_stream is not None:
            ids = token_stream.reshape(-1).long()
        else:
            if self.tokenizer is None:
                raise ValueError("texts require a tokenizer with .encode; "
                                 "pass token_stream for pre-tokenized input")
            ids = torch.tensor([i for text in texts for i in self.tokenizer.encode(text)],
                               dtype=torch.long)
        if ids.numel() == 0:
            raise ValueError("no tokens to evaluate")
        reps = -(-(T + 1) // ids.numel())
        ids = ids.repeat(reps)[:T + 1]
        total, aux = self.model(ids[:-1].unsqueeze(0), ids[1:].unsqueeze(0))
        return float(total) - float(aux)

    @torch.no_grad()
    def _retrieval(self, T: int, needles: list[Needle], batch: int,
                   seed: int) -> tuple[float, list[dict]]:
        """Needle-in-a-haystack accuracy at ctx T. Rows sharing a needle
        shape (len(key), len(value)) batch into one forward; the head is
        applied only at the checked positions."""
        model = self.model
        C = model.cfg.chunk_len
        V = model.cfg.vocab_size
        device = next(model.parameters()).device

        by_span: dict[tuple[int, int], list[int]] = {}
        for n_i, nd in enumerate(needles):
            by_span.setdefault((len(nd.key_ids), len(nd.value_ids)), []).append(n_i)

        hits, detail = 0, []
        for (k_len, v_len), idxs in by_span.items():
            fixed = 2 * (k_len + v_len)
            if T - fixed < C:
                raise ValueError(
                    f"ctx {T} leaves {T - fixed} tokens for the haystack; a needle "
                    f"(key {k_len} + value {v_len}) needs at least {fixed + C}")
            rows, starts = [], []
            for n_i in idxs:
                nd = needles[n_i]
                g = torch.Generator().manual_seed(seed + 7919 * n_i + 104729 * T)
                a = int(nd.depth * (T - fixed))  # haystack before the planted needle
                b = T - fixed - a                # haystack between needle and query
                key = torch.tensor(nd.key_ids, dtype=torch.long)
                value = torch.tensor(nd.value_ids, dtype=torch.long)
                rows.append(torch.cat([self._filler(a, nd.filler_ids, V, g), key, value,
                                       self._filler(b, nd.filler_ids, V, g), key, value]))
                starts.append(a + k_len + v_len + b + k_len - 1)
            step = max(1, batch)
            for i0 in range(0, len(rows), step):
                sel = idxs[i0:i0 + step]
                inp = torch.stack(rows[i0:i0 + step]).to(device)
                h = _read_hidden(model, inp)                     # (B, T, d)
                pos = torch.tensor([[starts[i0 + r] + j for j in range(v_len)]
                                    for r in range(len(sel))], device=device)
                pred = model.head(h[:, pos])                     # (B, v_len, V) argmax next
                for r, n_i in enumerate(sel):
                    want = torch.tensor(needles[n_i].value_ids, device=pred.device)
                    ok = bool((pred[r].argmax(-1) == want).all())
                    hits += ok
                    detail.append({"length": T, "needle": n_i, "hit": ok,
                                   "depth": needles[n_i].depth})
        return hits / len(needles), detail

    @staticmethod
    def _filler(n: int, pool: list[int] | None, vocab: int,
                g: torch.Generator) -> Tensor:
        """n haystack ids: cycled from `pool` when given, else seeded uniform."""
        if n <= 0:
            return torch.empty(0, dtype=torch.long)
        if pool:
            pick = torch.randint(0, len(pool), (n,), generator=g)
            return torch.tensor(pool, dtype=torch.long)[pick]
        return torch.randint(0, vocab, (n,), generator=g)

    @staticmethod
    def _sync(device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()


def _all_chunk_baseline(ev: LongContextEvaluator, lengths, n_steps, warmup) -> dict:
    """Same weights with n_selected → N (read-every-chunk), timed through the
    identical harness — the in-repo bound learned top-k selection is measured
    against. models/attention.py:HiLSAttention clamps k to n_chunks; the
    flag is restored on exit."""
    attns = [blk.attn for blk in ev.model.blocks]
    saved = [a.n_selected for a in attns]
    try:
        for a in attns:
            a.n_selected = 1 << 30
        return {"throughput": {T: ev._measure_decode(T, n_steps=n_steps, warmup=warmup)[0]
                               for T in lengths}}
    finally:
        for a, s in zip(attns, saved):
            a.n_selected = s


register_baseline("all-chunk", _all_chunk_baseline)
