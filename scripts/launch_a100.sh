#!/usr/bin/env bash
# Launch the two-phase pretrain on an A100 80GB pod (plan §4.1, SKILLS Skill 2).
#
# One config, one schedule: 53,407 steps @ seq 4096 (7.0B tokens) then
# 7,630 steps @ seq 16384 (1.0B tokens) — configs/pretrain_a100_341m.yaml.
# Resumes automatically from the latest complete checkpoint in
# checkpoints/pretrain_a100 (utils/checkpoint.py:CheckpointManager).
#
# Before launching: prepare shards (python3 data/prepare_data.py) and run the
# boundary checks (scripts/microbench_a100.py, scripts/step_time_a100.py,
# scripts/e2e_gpu_smoke.py).
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 training/pretrain.py configs/pretrain_a100_341m.yaml