#!/usr/bin/env bash
set -u
source .predictive_driving/bin/activate
export WANDB_MODE=offline

CONFIGS=(
  "experiments/001_ppo_baseline/config.yaml ppo_highway_v0_kin_b2a41"
  "experiments/002_ppo_merge/config.yaml ppo_merge_v0_kin_b2a41"
  "experiments/003_ppo_roundabout/config.yaml ppo_roundabout_v0_kin_b2a41"
  "experiments/004_ppo_intersection/config.yaml ppo_intersection_v0_kin_b2a41"
  "experiments/005_ppo_two_way/config.yaml ppo_two_way_v0_kin_b2a41"
  "experiments/006_ppo_u_turn/config.yaml ppo_u_turn_v0_kin_b2a41"
  "experiments/007_ppo_exit/config.yaml ppo_exit_v0_kin_b2a41"
  "experiments/008_ppo_racetrack/config.yaml ppo_racetrack_v0_kin_b2a41"
  "experiments/009_ppo_highway_occgrid/config.yaml ppo_highway_v0_occ_b2a41"
  "experiments/010_ppo_merge_occgrid/config.yaml ppo_merge_v0_occ_b2a41"
  "experiments/011_ppo_roundabout_occgrid/config.yaml ppo_roundabout_v0_occ_b2a41"
  "experiments/012_ppo_intersection_occgrid/config.yaml ppo_intersection_v0_occ_b2a41"
  "experiments/013_ppo_two_way_occgrid/config.yaml ppo_two_way_v0_occ_b2a41"
  "experiments/014_ppo_u_turn_occgrid/config.yaml ppo_u_turn_v0_occ_b2a41"
  "experiments/015_ppo_exit_occgrid/config.yaml ppo_exit_v0_occ_b2a41"
  "experiments/016_ppo_racetrack_occgrid/config.yaml ppo_racetrack_v0_occ_b2a41"
)

mkdir -p logs/full
for entry in "${CONFIGS[@]}"; do
  read -r cfg name <<< "$entry"
  expdir=$(dirname "$cfg")
  resdir="$expdir/results"
  logf="$resdir/train.log"
  errf="$resdir/error.log"
  mkdir -p "$resdir"
  echo "=== $(date -Iseconds) START $name ==="
  if python -m driving.train --config "$cfg" --run_name "$name" > "$logf" 2>&1; then
    echo "=== $(date -Iseconds) PASS $name ==="
  else
    echo "=== $(date -Iseconds) FAIL $name ==="
    cp "$logf" "$errf"
    tail -40 "$logf"
  fi
done
echo "=== ALL FULL RUNS DONE ==="
