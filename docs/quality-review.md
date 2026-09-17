# Diagram quality review

Scope: three specs, their existing HTML outputs, guide and evidence only. No model code changes.

## Pinned source

Revision: `3dadb375e1e055ac9cbba10be3cdc6ff757f0b6b`. Configuration: `configs/pretrain_a100_341m.yaml`.

Config SHA-256: `85610edecbc65cc7b0102b50a0ef5b8720c62cb4e269d217a9dd26ae5e5f8d72`.

Shared data is not a Git repository. Its relevant on-disk files are pinned by SHA-256 below. Dtype is verified from the packing code, not a production manifest or corpus inventory.

## Source and semantic checks

- `models/transformer.py:HiLSAttentionLM.forward` defines output contracts.
- `training/pretrain.py:train` defines iteration, counters, recovery and stop transitions.
- `utils/checkpoint.py:CheckpointManager.save` writes directly, without RNG or data-cursor persistence.
- The bounded repeated unit excludes final norm, head and loss. Both residual additions are explicit.
- GPT-2 packing selects uint16; HiLS copies to a uint32 host buffer. Mamba expects serialized tensor shards, so the raw-shard bridge remains absent.
- HiLS watchdog requires max-share >0.5 for 500 consecutive optimizer updates. A <=0.5 update resets its streak. Bad-loss streak resets only on optimizer success. Saving precedes watchdog evaluation.

## Acceptance and limitations

All three specs must pass nine showcase checks before delivery. Browser checks run only after successful delivery. Exact hashes and fresh status are in the adjacent combined receipt. No stale visual approval is inherited.

Perceptual review: unavailable. The image reader returned a provider/model image-input limitation. No claim of visual polish is made. The proposed 12px essential / 11px secondary first-screen target is not met. Automated 6px floor is a different gate. GPU benchmarks, production corpus inventory and full pretraining are not run.

Unlabelled solid model edges denote computation whose endpoints imply the operation. Workflow labels state conditions. Dashed workflow edges are returns, not asynchronous work.

## Source hashes

| Path | SHA-256 |
|---|---|
| `models/transformer.py` | `724d33119797534d8be7b89820f2cb5b2905b1b7a3fbc5215425c8cd37c3bae0` |
| `training/pretrain.py` | `caa8893537342a3c1df7ce2c580805d08aeb01b7f724ace830f97952e61a1de0` |
| `utils/checkpoint.py` | `ca0f99b0dafa65ff8ecdd99c20de9ae61284a12e24f51b802a8a11860cadb8c8` |
| `data/prepare_data.py` | `12af294911d338e6a62d9a42151c901afe1879b4d1044bc5e7ac609b08b5ec0c` |
| `models/router.py` | `3b3e5f36d3fefc867c118e7d6f2fa3fc13b20ce54540e1252ee1ef2385288b6d` |
| `models/block.py` | `aa1d5beab8e41701bb8360338f9df325ba06805a1d49f22bcdd1230a3804dc17` |
| `models/attention.py` | `fefa9e203d1c24e7a5ea9170138cad29d6c77b1c268e2e689c5c2923926fef93` |
| `training/losses.py` | `1094f8741b64e3edaa32b656a562519b184a2f908cc3864389a550a30e67be2d` |
| `../shared_data/shard_writer.py` | `35a81ffc5d268af96009b68c6ea17cf82c6ea5a2b5a83c225399ee9e78c5b3a8` |
| `../shared_data/scripts/pack_shards.py` | `0f90101375437b5155260897ce7478f02f71bc0476f635465bb98e7fe21d8f31` |
| `../shared_data/loader.py` | `18fcdd882c84eafecb4d27c719d9c80df6fed7ec279e5199f61a8da485168347` |
| `../shared_data/dataset.py` | `74a3fe12b76138e78b8fc2115441d88b139bceb9fd4bcdcb132bec20d352ceab` |
| `../shared_data/config/mixture.yaml` | `fefd48c045af197cb18da1e0d710dee4aaf7aebf63af544633f89848364820e0` |
