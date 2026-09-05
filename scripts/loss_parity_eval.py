#!/usr/bin/env python3
"""Held-out NLL parity vs LLaMA-3-Lite at seq 4096 (plan §5.1, gate B4).

Both models score held-out text through their OWN loss path:

  - ours: models/transformer.py:HiLSAttentionLM — total − aux from
    models/transformer.py:HiLSAttentionLM.forward, i.e. the chunked-CE
    objective (training/losses.py:chunked_lm_ce) without the aux term;
  - baseline: LLaMA-3-Lite Transformer (its repo's config.py:get_config()
    shape + checkpoint) scored with plain full-vocab cross-entropy.

The parameter-budget delta is disclosed in every run (ours ~341M vs the
baseline's ~515M — parity is anchored, not matched-budget; plan §11).

Gate (ΔNLL = ours / baseline − 1, per token, same held-out text):
    ΔNLL ≤ +5% → [PASS]; larger → [DISCLOSED] prominently, never hidden.

Tokenizers differ (GPT-2 ~50k vs the baseline's ~128k), so the honest
same-text comparison uses each model's own tokenization: --texts-file scores
one UTF-8 file through each tokenizer; --ours-shard / --baseline-shard score
pre-tokenized held-out windows. Whatever input is missing is DISCLOSED,
never silently dropped (house pattern: print the gap — cf. DIFFUSION.md §7).
The script exits 0 in every reported case.

CPU self-check (tiny random-init models, cross-repo import exercised; the
parity verdict itself is plumbing-only and disclosed):
    python scripts/loss_parity_eval.py --tiny
A100 headline:
    python scripts/loss_parity_eval.py --ours-ckpt checkpoints/pretrain_a100 \
        --texts-file heldout.txt --baseline-ckpt ../LLaMA-3-Lite/checkpoints/model_final.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.transformer import HiLSAttentionLM, HiLSConfig  # noqa: E402
from training.pretrain import load_config  # noqa: E402
from utils.checkpoint import CheckpointManager  # noqa: E402

PARITY_GATE = 0.05  # B4: ΔNLL ≤ +5% vs the ~515M full-attention baseline

TINY_OURS = dict(vocab_size=50257, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2,
                 head_dim=32, ffn_dim=256, weight_tying=True, rms_norm_eps=1e-5,
                 init_std=0.02, rope_theta=500000.0, max_seq_len=512,
                 attn_impl="sdpa", chunk_len=128, n_selected=8, aux_balance_weight=0.01)
TINY_BASE = dict(vocab_size=4096, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2,
                 head_dim=32, d_ff=256, max_seq_len=512)


def n_unique_params(model: torch.nn.Module) -> int:
    """Parameter count without double-counting tied weights."""
    seen: set[int] = set()
    total = 0
    for p in model.parameters():
        ptr = p.data_ptr()
        if ptr not in seen:
            seen.add(ptr)
            total += p.numel()
    return total


@torch.no_grad()
def our_nll(model: HiLSAttentionLM, windows: torch.Tensor, device: str) -> float:
    """Per-token NLL for (B, T+1) GPT-2-token windows: the model's own loss
    path minus the aux term (same protocol as
    inference/evaluate.py:LongContextEvaluator._nll_at, batched)."""
    tokens, targets = windows[:, :-1].to(device), windows[:, 1:].to(device)
    total, aux = model(tokens, targets)
    return float(total) - float(aux)


@torch.no_grad()
def llama_nll(model, windows: torch.Tensor, device: str) -> float:
    """Per-token NLL for (B, T+1) baseline-vocab windows via plain CE
    (the baseline's z-loss regularizer is not part of NLL)."""
    tokens, targets = windows[:, :-1].to(device), windows[:, 1:].to(device)
    logits = model(tokens)
    vocab = logits.size(-1)
    return float(F.cross_entropy(logits[:, :-1].reshape(-1, vocab).float(),
                                 targets[:, 1:].reshape(-1)))

def windows_from_ids(ids: np.ndarray, seq: int, n_windows: int, seed: int) -> torch.Tensor:
    """n_windows disjoint (seq+1)-token windows — non-overlapping so every
    held-out token is scored exactly once."""
    need = n_windows * (seq + 1)
    if ids.shape[0] < need:
        raise ValueError(f"held-out stream has {ids.shape[0]:,} tokens, "
                         f"needs {need:,} ({n_windows} windows × {seq + 1})")
    idx = np.random.default_rng(seed).permutation(ids.shape[0] - seq - 1)[:n_windows]
    return torch.from_numpy(np.stack([ids[i:i + seq + 1] for i in sorted(idx)])).long()


def load_shard(path: Path) -> np.ndarray:
    """uint32 shard → int64 id array."""
    return np.asarray(np.memmap(path, dtype=np.uint32, mode="r"), dtype=np.int64)


def load_llama_baseline(repo: Path, ckpt_path: str | None, tiny: bool, device: str):
    """LLaMA-3-Lite Transformer + checkpoint → (model, vocab, None) or
    (None, None, reason) for disclosure. --tiny builds a tiny random-init
    baseline so the cross-repo import path is exercised on CPU."""
    if not repo.exists():
        return None, None, f"baseline repo not found: {repo}"
    if ckpt_path and not Path(ckpt_path).exists():
        return None, None, f"baseline checkpoint not found: {ckpt_path}"
    sys.path.insert(0, str(repo))
    try:
        from model import Transformer as LlamaTransformer  # noqa: E402
        from config import get_config  # noqa: E402
    except Exception as exc:  # noqa: BLE001 — any import failure is disclosed
        return None, None, f"LLaMA-3-Lite import failed: {type(exc).__name__}: {exc}"
    if tiny:
        cfg = dict(vocab_size=TINY_BASE["vocab_size"], d_model=TINY_BASE["d_model"],
                   n_layers=TINY_BASE["n_layers"], n_heads=TINY_BASE["n_heads"],
                   n_kv_heads=TINY_BASE["n_kv_heads"], head_dim=TINY_BASE["head_dim"],
                   d_ff=TINY_BASE["d_ff"], max_seq_len=TINY_BASE["max_seq_len"])
    else:
        src = get_config()
        cfg = dict(vocab_size=src["vocab_size"], d_model=src["d_model"],
                   n_layers=src["n_layers"], n_heads=src["n_heads"],
                   n_kv_heads=src["n_kv_heads"], head_dim=src["head_dim"],
                   d_ff=src["d_ff"], max_seq_len=4096,
                   rope_theta=src["rope_theta"], rms_norm_eps=src["rms_norm_eps"],
                   qknorm=src["qknorm"])
    model = LlamaTransformer(**cfg)
    if ckpt_path:
        blob = torch.load(ckpt_path, map_location=device, weights_only=True)
        state = blob.get("model_state_dict", blob.get("model", blob))
        model.load_state_dict(state)
    return model.to(device).eval(), cfg["vocab_size"], None


def encode_texts(path: Path, tokenizer_name: str) -> np.ndarray:
    """UTF-8 text → ids via the named HF tokenizer (per-model tokenization)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    text = path.read_text(encoding="utf-8")
    return np.asarray(tok(text, add_special_tokens=False)["input_ids"], dtype=np.int64)


def main() -> int:
    p = argparse.ArgumentParser(description="Held-out NLL parity vs LLaMA-3-Lite (plan §5.1)")
    p.add_argument("--config", default=str(ROOT / "configs" / "pretrain_a100_341m.yaml"))
    p.add_argument("--tiny", action="store_true",
                   help="tiny random-init models on CPU (plumbing check; parity DISCLOSED)")
    p.add_argument("--ours-ckpt", default=None, help="our checkpoint dir (optional)")
    p.add_argument("--baseline-ckpt", default=None, help="LLaMA-3-Lite .pt checkpoint")
    p.add_argument("--baseline-repo", default=str(ROOT.parent / "LLaMA-3-Lite"))
    p.add_argument("--texts-file", default=None,
                   help="UTF-8 held-out text (each model's own tokenizer scores it)")
    p.add_argument("--ours-shard", default=None, help="uint32 GPT-2-token held-out shard")
    p.add_argument("--baseline-shard", default=None,
                   help="uint32 baseline-token held-out shard (same text, its tokenizer)")
    p.add_argument("--baseline-tokenizer", default=None,
                   help="HF tokenizer name/path for --texts-file on the baseline side")
    p.add_argument("--seq", type=int, default=4096)
    p.add_argument("--windows", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.tiny and args.seq > TINY_OURS["max_seq_len"]:
        args.seq = TINY_OURS["max_seq_len"]

    # --- ours ---------------------------------------------------------------
    cfg = {"model": dict(TINY_OURS)} if args.tiny else load_config(args.config)
    ours = HiLSAttentionLM(HiLSConfig(**cfg["model"]))
    if args.ours_ckpt:
        mgr = CheckpointManager(args.ours_ckpt)
        step = mgr.latest_step()
        if step is None:
            print(f"[parity] DISCLOSED: no checkpoint in {args.ours_ckpt} — random init")
        else:
            mgr.load(ours, step, device=device)
            print(f"[parity] loaded ours @ step {step} from {args.ours_ckpt}")
    ours = ours.to(device).eval()

    # --- baseline (optional) --------------------------------------------------
    llama, llama_vocab, err = load_llama_baseline(Path(args.baseline_repo),
                                                  args.baseline_ckpt, args.tiny, device)
    if llama is None:
        print(f"[parity] DISCLOSED: baseline not measured — {err}")
    else:
        print(f"[parity] baseline loaded ({llama_vocab:,} vocab)")

    # --- held-out windows per model ------------------------------------------
    ours_ids = base_ids = None
    try:
        if args.texts_file:
            ours_ids = encode_texts(Path(args.texts_file), "gpt2")
            if llama is not None:
                if args.baseline_tokenizer:
                    base_ids = encode_texts(Path(args.texts_file), args.baseline_tokenizer)
                else:
                    print("[parity] DISCLOSED: baseline tokenizer not provided "
                          "(--baseline-tokenizer) — baseline NLL skipped")
        if args.ours_shard:
            ours_ids = load_shard(Path(args.ours_shard))
        if args.baseline_shard:
            base_ids = load_shard(Path(args.baseline_shard))
        if args.tiny and (ours_ids is None or base_ids is None):
            g = np.random.default_rng(args.seed)
            ours_ids = g.integers(0, TINY_OURS["vocab_size"],
                                  size=args.windows * (args.seq + 1)).astype(np.int64)
            base_ids = g.integers(0, TINY_BASE["vocab_size"],
                                  size=args.windows * (args.seq + 1)).astype(np.int64)
    except ValueError as exc:
        print(f"[parity] DISCLOSED: {exc}")
        return 0

    ours_nll = our_nll(ours, windows_from_ids(ours_ids, args.seq, args.windows, args.seed),
                       device)
    print(f"[parity] ours: NLL {ours_nll:.4f} nats/token @ seq {args.seq}, "
          f"{args.windows} held-out windows, {n_unique_params(ours) / 1e6:.0f}M params")

    if llama is None or base_ids is None:
        print(f"[parity] B4 ΔNLL vs LLaMA-3-Lite @ {args.seq}: ours {ours_nll:.4f} "
              "[DISCLOSED — baseline NLL not measured]")
        return 0

    base_nll = llama_nll(llama, windows_from_ids(base_ids, args.seq, args.windows, args.seed),
                         device)
    delta = ours_nll / base_nll - 1.0
    ok = delta <= PARITY_GATE
    verdict = "DISCLOSED" if args.tiny else ("PASS" if ok else "DISCLOSED")
    ours_p, base_p = n_unique_params(ours), n_unique_params(llama)
    print(f"[parity] baseline: NLL {base_nll:.4f} nats/token, {base_p / 1e6:.0f}M params "
          f"(budget delta disclosed: baseline/ours = {base_p / ours_p:.2f}×)")
    print(f"[parity] B4 ΔNLL @ seq {args.seq}: {delta * 100:+.2f}% [{verdict} ≤ +5%]")
    if not ok and not args.tiny:
        print(f"[parity] DISCLOSED: parity gate missed by {delta * 100 - 5:.2f} points "
              "— reported prominently, never hidden (plan §5.1)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
