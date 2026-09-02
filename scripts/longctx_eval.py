#!/usr/bin/env python3
"""Long-context decode throughput + effective KV-access fraction (plan §5.1, gates B1/B2).

Per context length, through the one harness every headline number shares
(inference/evaluate.py:LongContextEvaluator):

  - ours: chunk-synchronous decode tokens/sec (inference/generate.py:prefill +
    step_decode) and the instrumented KV-access fraction
    (inference/generate.py:kv_access_fraction) over the same timed window;
  - all-chunk diagnostic: the same weights re-timed with n_selected → N —
    what learned top-k selection saves, in-repo, no external model needed;
  - baseline (optional): LLaMA-3-Lite checkpoint decode tokens/sec at the
    same context (its model decodes with a full forward per token — no KV
    cache), with the parameter-budget delta disclosed in the output.

Gates, evaluated at the LARGEST requested length (production: 16384):
    B1  decode throughput ≥ 1.8× vs the LLaMA-3-Lite baseline  (claim ~2×)
    B2  KV-access fraction ≤ 0.08                              (claim ~6%)

Honesty contract (AGENTS.md rule 4): every gate prints its measured number
with an explicit [PASS …] / [DISCLOSED …] marker. A missing checkpoint or
baseline is DISCLOSED, never silently dropped; a measured miss is also
DISCLOSED next to its gate. The script exits 0 in every reported case —
a disclosed miss is a result, not a crash. The B2 gate is designed for the
16K context (8 of 128 chunks selected): tiny/short runs print their measured
fraction and DISCLOSE against the production gate.

CPU self-check (tiny 2-layer config, no checkpoints):
    python scripts/longctx_eval.py --tiny --ctx 512 1024
A100 headline:
    python scripts/longctx_eval.py --ours-ckpt checkpoints/pretrain_a100 \
        --baseline-ckpt ../LLaMA-3-Lite/checkpoints/model_final.pt --ctx 16384
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.evaluate import LongContextEvaluator  # noqa: E402
from models.transformer import HiLSAttentionLM, HiLSConfig  # noqa: E402
from training.pretrain import load_config  # noqa: E402
from utils.checkpoint import CheckpointManager  # noqa: E402

THROUGHPUT_GATE = 1.8     # B1: ours / LLaMA-3-Lite decode tok/s (claim ~2×)
KV_FRACTION_GATE = 0.08   # B2: effective KV access at the 16K context (claim ~6%)

TINY_MODEL = dict(vocab_size=50257, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2,
                  head_dim=32, ffn_dim=256, weight_tying=True, rms_norm_eps=1e-5,
                  init_std=0.02, rope_theta=500000.0, max_seq_len=256,
                  attn_impl="sdpa", chunk_len=32, n_selected=4, aux_balance_weight=0.01)


def n_unique_params(model: torch.nn.Module) -> int:
    """Parameter count without double-counting tied weights (embed ↔ head)."""
    seen: set[int] = set()
    total = 0
    for p in model.parameters():
        ptr = p.data_ptr()
        if ptr not in seen:
            seen.add(ptr)
            total += p.numel()
    return total


def build_ours(cfg: dict, ckpt_dir: str | None, device: str) -> HiLSAttentionLM:
    """HiLSAttentionLM from the config; latest checkpoint restored when given
    (utils/checkpoint.py:CheckpointManager — random init is disclosed)."""
    model = HiLSAttentionLM(HiLSConfig(**cfg["model"]))
    if ckpt_dir:
        mgr = CheckpointManager(ckpt_dir)
        step = mgr.latest_step()
        if step is None:
            print(f"[longctx] DISCLOSED: no checkpoint in {ckpt_dir} — random init")
        else:
            mgr.load(model, step, device=device)
            print(f"[longctx] loaded ours @ step {step} from {ckpt_dir}")
    return model.to(device).eval()

@torch.no_grad()
def llama_decode_tps(model, ctx: int, steps: int, vocab: int, device: str) -> float:
    """LLaMA-3-Lite decode tokens/sec at ctx: one full forward per generated
    token (its attention has no KV cache), sliding window kept at ctx tokens.
    Same greedy-generation protocol as our chunk-synchronous harness: fixed
    random context, tokens/sec = steps / wall-clock."""
    x = torch.randint(0, vocab, (1, ctx), device=device)
    for _ in range(2):  # untimed warmup
        logits = model(x)
        x = torch.cat([x[:, 1:], logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        logits = model(x)
        x = torch.cat([x[:, 1:], logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    if device == "cuda":
        torch.cuda.synchronize()
    return steps / max(time.perf_counter() - t0, 1e-9)


def load_llama_baseline(repo: Path, ckpt_path: str, ctx: int, device: str):
    """LLaMA-3-Lite Transformer from its repo + checkpoint. Returns
    (model, vocab_size) or (None, reason) — the caller discloses what's missing."""
    if not repo.exists():
        return None, f"baseline repo not found: {repo}"
    if not Path(ckpt_path).exists():
        return None, f"baseline checkpoint not found: {ckpt_path}"
    sys.path.insert(0, str(repo))
    try:
        from model import Transformer as LlamaTransformer  # noqa: E402
        from config import get_config  # noqa: E402
    except Exception as exc:  # noqa: BLE001 — any import failure is disclosed
        return None, f"LLaMA-3-Lite import failed: {type(exc).__name__}: {exc}"
    cfg = get_config()
    model = LlamaTransformer(
        vocab_size=cfg["vocab_size"], d_model=cfg["d_model"], n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"], n_kv_heads=cfg["n_kv_heads"], head_dim=cfg["head_dim"],
        d_ff=cfg["d_ff"], max_seq_len=max(cfg["seq_len"], ctx),
        rope_theta=cfg["rope_theta"], rms_norm_eps=cfg["rms_norm_eps"],
        qknorm=cfg["qknorm"])
    blob = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = blob.get("model_state_dict", blob.get("model", blob))
    model.load_state_dict(state)
    model.cfg_vocab = cfg["vocab_size"]
    return model.to(device).eval(), cfg["vocab_size"]


def main() -> int:
    p = argparse.ArgumentParser(description="Long-context decode throughput + KV fraction (plan §5.1)")
    p.add_argument("--config", default=str(ROOT / "configs" / "pretrain_a100_341m.yaml"))
    p.add_argument("--tiny", action="store_true",
                   help="2-layer CPU self-check config instead of --config")
    p.add_argument("--ours-ckpt", default=None,
                   help="checkpoint dir for utils/checkpoint.py:CheckpointManager (optional)")
    p.add_argument("--baseline-ckpt", default=None,
                   help="LLaMA-3-Lite .pt checkpoint (optional; B1 is DISCLOSED without it)")
    p.add_argument("--baseline-repo", default=str(ROOT.parent / "LLaMA-3-Lite"))
    p.add_argument("--ctx", type=int, nargs="+", default=None,
                   help="context lengths (default: 512 1024 tiny / 16384 production)")
    p.add_argument("--steps", type=int, default=3, help="timed decode chunks per length")
    p.add_argument("--warmup", type=int, default=1, help="untimed warmup chunks per length")
    p.add_argument("--device", default=None, help="default: cuda when available")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    lengths = args.ctx or ([512, 1024] if args.tiny else [16384])
    cfg = {"model": dict(TINY_MODEL)} if args.tiny else load_config(args.config)
    model = build_ours(cfg, args.ours_ckpt, device)
    ours_params = n_unique_params(model)

    ev = LongContextEvaluator(model)
    res = ev.evaluate(lengths, baselines=["all-chunk"], n_steps=args.steps, warmup=args.warmup)

    print(f"[longctx] ours: {ours_params / 1e6:.0f}M params, device {device}")
    print("[longctx]   ctx | ours tok/s | kv fraction | all-chunk tok/s | selection saves")
    for T in lengths:
        tps, frac = res["throughput"][T], res["kv_fraction"][T]
        allchunk = res["baselines"]["all-chunk"]["throughput"][T]
        saves = tps / allchunk if allchunk > 0 else float("nan")
        print(f"[longctx] {T:>6} | {tps:>10,.0f} | {frac:>11.4f} | "
              f"{allchunk:>15,.0f} | {saves:>7.1f}×")

    # --- baseline (optional): LLaMA-3-Lite decode throughput ----------------
    baseline_tps = {}
    llama = None
    llama_params = None
    llama_vocab = None
    if args.baseline_ckpt:
        llama, llama_vocab, err = load_llama_baseline(Path(args.baseline_repo),
                                                      args.baseline_ckpt, max(lengths), device)
        if llama is None:
            print(f"[longctx] DISCLOSED: baseline not measured — {err}")
        else:
            llama_params = n_unique_params(llama)
            print(f"[longctx] baseline: LLaMA-3-Lite {llama_params / 1e6:.0f}M params "
                  f"(budget delta disclosed: ours {ours_params / 1e6:.0f}M, "
                  f"{llama_params / ours_params:.2f}×)")
    else:
        print("[longctx] DISCLOSED: no --baseline-ckpt — B1 speedup unmeasurable here; "
              "pass ../LLaMA-3-Lite/checkpoints/<model>.pt on the pod")

    # --- gates (at the largest requested length) ----------------------------
    T = max(lengths)
    frac = res["kv_fraction"][T]
    ok_b2 = frac <= KV_FRACTION_GATE
    print(f"[longctx] B2 KV-access fraction @ {T}: {frac:.4f} "
          f"[{'PASS' if ok_b2 else 'DISCLOSED'} ≤ {KV_FRACTION_GATE}]")

    if llama is not None:
        base = llama_decode_tps(llama, T, max(args.steps * 4, 8), llama_vocab, device)
        speedup = res["throughput"][T] / base if base > 0 else float("nan")
        ok = speedup >= THROUGHPUT_GATE
        print(f"[longctx] baseline decode @ {T}: {base:,.0f} tok/s → "
              f"speedup {speedup:.2f}× [{'PASS' if ok else 'DISCLOSED'} ≥ {THROUGHPUT_GATE}×]")
    else:
        print(f"[longctx] B1 throughput @ {T}: {res['throughput'][T]:,.0f} tok/s "
              "[DISCLOSED — baseline not measured]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
