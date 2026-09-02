#!/usr/bin/env python3
"""Needle-in-a-haystack retrieval + NLL-vs-length (plan §5.1, gate B3).

Protocol (inference/evaluate.py:LongContextEvaluator — the one harness every
headline number shares): template needles ("The special magic number for
<key>-<i> is <value>.") planted inside a seeded haystack at alternating depth,
the needle's key repeated at the row end, every value token checked by argmax
(multi-token values are teacher-forced by construction). NLL-vs-length runs
on the held-out tail of a token shard when one is available.

Lengths default to 16384 32768 65536 — up to 4× the 16K training length, no
RoPE stretching: models/transformer.py:HiLSAttentionLM computes frequencies
for the actual sequence length on every forward (DESIGN §7.6), and landmark
count scales linearly (N = T/C).

Gate, evaluated at the LARGEST requested length (production: 64K):
    retrieval accuracy ≥ 85% — else the measured curve is disclosed.

Honesty contract (AGENTS.md rule 4): [PASS …]/[DISCLOSED …] markers only.
A missing checkpoint, tokenizer, or shard is disclosed, never silently
dropped; the script exits 0 in every reported case. On --tiny (random init)
retrieval is plumbing-only and its number is meaningless — disclosed as such.

CPU self-check:
    python scripts/retrieval_eval.py --tiny
A100 headline:
    python scripts/retrieval_eval.py --ours-ckpt checkpoints/pretrain_a100 \
        --lengths 16384 32768 65536 --needles 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.evaluate import LongContextEvaluator, Needle  # noqa: E402
from models.transformer import HiLSAttentionLM, HiLSConfig  # noqa: E402
from training.pretrain import load_config  # noqa: E402
from utils.checkpoint import CheckpointManager  # noqa: E402

RETRIEVAL_GATE = 0.85  # B3: ≥ 85% at the largest length (4× extrapolation), else disclose

KEY_WORDS = ["aurora", "basilisk", "cinder", "dusk", "ember", "fathom", "gale",
             "harbor", "indigo", "jasper", "kelp", "lumen", "mosaic", "nectar",
             "onyx", "prism", "quartz", "ripple", "saffron", "thistle"]
FILLER_SENT = ("The lighthouse keeper counted the waves and recorded the weather in a "
               "worn leather notebook, as she had every evening for thirty years.\n")

TINY_MODEL = dict(vocab_size=50257, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2,
                  head_dim=32, ffn_dim=256, weight_tying=True, rms_norm_eps=1e-5,
                  init_std=0.02, rope_theta=500000.0, max_seq_len=256,
                  attn_impl="sdpa", chunk_len=32, n_selected=4, aux_balance_weight=0.01)


def get_tokenizer():
    """GPT-2 BPE (the repo's tokenizer contract) or None — disclosed either way."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
        return tok
    except Exception as exc:  # noqa: BLE001 — offline/missing hub is disclosed
        print(f"[retrieval] DISCLOSED: no tokenizer ({type(exc).__name__}: {exc}) — "
              "needles fall back to synthetic token ids")
        return None


def build_text_needles(n: int, tokenizer) -> list:
    """Template needles with exact token-span alignment via tokenizer offsets:
    key_ids = needle text up to the value span, value_ids = the span itself."""
    needles = []
    for i in range(n):
        word = KEY_WORDS[i % len(KEY_WORDS)]
        value = str(10000 + (i * 7919) % 89999)
        full = f"The special magic number for {word}-{i} is {value}."
        enc = tokenizer(full, return_offsets_mapping=True, add_special_tokens=False)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        v0 = full.index(value)  # the value appears exactly once (keys are words+i)
        span = [k for k, (s, e) in enumerate(offs)  # overlap: token i covers part of the value
                if s < v0 + len(value) and e > v0]
        if not span or offs[span[-1]][1] != v0 + len(value):
            raise RuntimeError(f"needle {i}: value span not aligned in tokenization")
        needles.append(Needle(key_ids=ids[:span[0]], value_ids=[ids[k] for k in span],
                              filler_ids=tokenizer(FILLER_SENT, add_special_tokens=False)["input_ids"],
                              depth=0.25 + 0.5 * (i % 2)))
    return needles


def build_synthetic_needles(n: int, vocab: int) -> list:
    """Deterministic id-level needles when no tokenizer is available."""
    return [Needle(key_ids=[11 + i, 12 + i, 13 + i, 14 + i],
                   value_ids=[5000 + i, 6000 + i], depth=0.25 + 0.5 * (i % 2))
            for i in range(n)]


def find_heldout_shard() -> Path | None:
    """The last shard on disk — its tail is the held-out NLL source."""
    for d in (ROOT / "data" / "pretrain_chinchilla" / "shards",
              ROOT.parent / "shared_data" / "pretrain_chinchilla",
              ROOT.parent / "shared_data" / "pretrain_chinchilla" / "shards"):
        if d.exists():
            bins = sorted(d.glob("*.bin"))
            if bins:
                return bins[-1]
    return None


def load_shard_tail(path: Path, n_tokens: int) -> object:
    """Last n_tokens of a uint32 shard as an int64 tensor (held-out tail)."""
    a = np.memmap(path, dtype=np.uint32, mode="r")
    take = min(n_tokens, a.shape[0])
    return torch.from_numpy(np.asarray(a[-take:], dtype=np.int64))

def main() -> int:
    p = argparse.ArgumentParser(description="Needle-in-a-haystack retrieval + NLL vs length (plan §5.1)")
    p.add_argument("--config", default=str(ROOT / "configs" / "pretrain_a100_341m.yaml"))
    p.add_argument("--tiny", action="store_true",
                   help="2-layer random-init CPU self-check (retrieval is plumbing-only)")
    p.add_argument("--ours-ckpt", default=None,
                   help="checkpoint dir for utils/checkpoint.py:CheckpointManager (optional)")
    p.add_argument("--lengths", type=int, nargs="+", default=None,
                   help="context lengths (default: 512 1024 tiny / 16384 32768 65536 production)")
    p.add_argument("--needles", type=int, default=None,
                   help="needles per length (default: 8 tiny / 20 production)")
    p.add_argument("--shard", default=None, help="uint32 shard for the NLL tail (default: auto)")
    p.add_argument("--nll-tokens", type=int, default=131072,
                   help="held-out tail size for NLL-vs-length")
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--batch", type=int, default=8, help="retrieval rows per forward")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    lengths = args.lengths or ([512, 1024] if args.tiny else [16384, 32768, 65536])
    n_needles = args.needles or (8 if args.tiny else 20)

    cfg = {"model": dict(TINY_MODEL)} if args.tiny else load_config(args.config)
    model = HiLSAttentionLM(HiLSConfig(**cfg["model"]))
    if args.ours_ckpt:
        mgr = CheckpointManager(args.ours_ckpt)
        step = mgr.latest_step()
        if step is None:
            print(f"[retrieval] DISCLOSED: no checkpoint in {args.ours_ckpt} — random init")
        else:
            mgr.load(model, step, device=device)
            print(f"[retrieval] loaded ours @ step {step} from {args.ours_ckpt}")
    model = model.to(device).eval()

    tokenizer = get_tokenizer()
    if tokenizer is not None:
        needles = build_text_needles(n_needles, tokenizer)
        print(f"[retrieval] needles: {n_needles} template needles "
              f"(GPT-2 BPE, depths 0.25/0.75 alternating)")
    else:
        needles = build_synthetic_needles(n_needles, model.cfg.vocab_size)
        print(f"[retrieval] needles: {n_needles} synthetic id needles")

    # NLL source: held-out tail of a shard (ours are GPT-2 tokens)
    stream = None
    shard = Path(args.shard) if args.shard else find_heldout_shard()
    if shard is not None:
        stream = load_shard_tail(shard, args.nll_tokens)
        print(f"[retrieval] NLL source: held-out tail ({stream.numel():,} tokens) "
              f"of {shard.name}")
    else:
        print("[retrieval] DISCLOSED: no token shard found — NLL-vs-length skipped "
              "(pass --shard <uint32 .bin>)")

    ev = LongContextEvaluator(model, tokenizer=tokenizer)
    res = ev.evaluate(lengths, needles=needles, token_stream=stream,
                      n_steps=args.steps, warmup=args.warmup,
                      retrieval_batch=args.batch, seed=args.seed)

    print("[retrieval]   ctx | retrieval | nll/token |   tok/s | kv fraction")
    for T in lengths:
        acc = res["retrieval"][T]
        nll = res.get("nll", {}).get(T)
        nll_s = f"{nll:9.4f}" if nll is not None else "    (skip)"
        print(f"[retrieval] {T:>6} | {acc:>9.4f} | {nll_s} | "
              f"{res['throughput'][T]:>7,.0f} | {res['kv_fraction'][T]:.4f}")

    T = max(lengths)
    acc = res["retrieval"][T]
    ok = acc >= RETRIEVAL_GATE
    if args.tiny:
        verdict = "DISCLOSED"
        print(f"[retrieval] B3 retrieval @ {T}: {acc:.4f} [DISCLOSED — random-init "
              "tiny run is plumbing-only, not a measurement]")
    else:
        verdict = "PASS" if ok else "DISCLOSED"
        print(f"[retrieval] B3 retrieval @ {T} ({T // 1024}K, "
              f"{T // 16384}× the 16K training length): {acc:.4f} "
              f"[{verdict} ≥ {RETRIEVAL_GATE}]")
        if not ok:
            curve = ", ".join(f"{t}: {res['retrieval'][t]:.3f}" for t in lengths)
            print(f"[retrieval] measured curve — {curve} (extrapolation miss "
                  "disclosed, not hidden)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
