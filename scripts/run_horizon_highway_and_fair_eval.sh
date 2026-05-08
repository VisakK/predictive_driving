#!/usr/bin/env bash
set -euo pipefail

# Disabled mode avoids W&B's local socket service in restricted environments.
# Videos are saved directly under experiments/048.../results/videos.
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DIR="${WANDB_DIR:-/tmp}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb-cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb-config}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export PYTHONPATH=src

PYTHON=.predictive_driving/bin/python
CONFIG=experiments/048_expected_horizon_highway/config.yaml

mkdir -p logs experiments/048_expected_horizon_highway

echo "=== Training 048 ExpectedInput-H10 highway ==="
"${PYTHON}" -m driving.train_adversarial \
  --config "${CONFIG}" \
  --run_name expected_horizon_highway_048 2>&1 \
  | tee experiments/048_expected_horizon_highway/train.log

echo "=== Running fair evaluation with ExpectedInput-H10 ==="
"${PYTHON}" scripts/fair_eval.py 2>&1 \
  | tee experiments/035_fair_eval/fair_eval_horizon.log
