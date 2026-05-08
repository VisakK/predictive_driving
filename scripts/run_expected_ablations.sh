#!/usr/bin/env bash
set -euo pipefail

export WANDB_MODE=disabled
export WANDB_DIR=/tmp
export WANDB_CACHE_DIR=/tmp/wandb-cache
export WANDB_CONFIG_DIR=/tmp/wandb-config
export MPLCONFIGDIR=/tmp/mpl
export PYTHONPATH=src

PYTHON=.predictive_driving/bin/python

runs=(
  "experiments/042_expected_input_highway/config.yaml expected_input_highway_042"
  "experiments/043_expected_input_roundabout/config.yaml expected_input_roundabout_043"
  "experiments/044_expected_input_reward_highway/config.yaml expected_input_reward_highway_044"
  "experiments/045_expected_input_reward_roundabout/config.yaml expected_input_reward_roundabout_045"
  "experiments/046_expected_online_highway/config.yaml expected_online_highway_046"
  "experiments/047_expected_online_roundabout/config.yaml expected_online_roundabout_047"
)

for run in "${runs[@]}"; do
  config=${run%% *}
  name=${run#* }
  echo "=== Starting ${name} ==="
  "${PYTHON}" -m driving.train_adversarial \
    --config "${config}" \
    --run_name "${name}" 2>&1 | tee "${config%/config.yaml}/train.log"
done

"${PYTHON}" scripts/fair_eval.py 2>&1 | tee experiments/035_fair_eval/fair_eval_expected.log
