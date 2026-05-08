#!/usr/bin/env bash
set -u
source .predictive_driving/bin/activate
export WANDB_MODE=offline

CONFIGS=(
  "experiments/002_ppo_merge/config.yaml smoke_002_b2a41"
  "experiments/003_ppo_roundabout/config.yaml smoke_003_b2a41"
  "experiments/004_ppo_intersection/config.yaml smoke_004_b2a41"
  "experiments/005_ppo_two_way/config.yaml smoke_005_b2a41"
  "experiments/006_ppo_u_turn/config.yaml smoke_006_b2a41"
  "experiments/007_ppo_exit/config.yaml smoke_007_b2a41"
  "experiments/008_ppo_racetrack/config.yaml smoke_008_b2a41"
  "experiments/009_ppo_highway_occgrid/config.yaml smoke_009_b2a41"
  "experiments/010_ppo_merge_occgrid/config.yaml smoke_010_b2a41"
  "experiments/011_ppo_roundabout_occgrid/config.yaml smoke_011_b2a41"
  "experiments/012_ppo_intersection_occgrid/config.yaml smoke_012_b2a41"
  "experiments/013_ppo_two_way_occgrid/config.yaml smoke_013_b2a41"
  "experiments/014_ppo_u_turn_occgrid/config.yaml smoke_014_b2a41"
  "experiments/015_ppo_exit_occgrid/config.yaml smoke_015_b2a41"
  "experiments/016_ppo_racetrack_occgrid/config.yaml smoke_016_b2a41"
)

mkdir -p logs/smokes
for entry in "${CONFIGS[@]}"; do
  read -r cfg name <<< "$entry"
  logf="logs/smokes/${name}.log"
  echo "=== $(date -Iseconds) START $name ==="
  if python -m driving.train --config "$cfg" --run_name "$name" --smoke > "$logf" 2>&1; then
    echo "=== PASS $name ==="
  else
    echo "=== FAIL $name (see $logf) ==="
    tail -20 "$logf"
  fi
done
echo "=== ALL SMOKES DONE ==="
