#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
source .predictive_driving/bin/activate

run_exp() {
    local num="$1"
    local name="$2"
    local run_name="$3"
    local cfg="experiments/${num}_${name}/config.yaml"
    local log="experiments/${num}_${name}/results/train.log"
    mkdir -p "experiments/${num}_${name}/results"
    echo "=== STARTING ${num}_${name} (${run_name}) ==="
    date
    python -m driving.train --config "$cfg" --run_name "$run_name" 2>&1 | tee "$log"
    echo "=== FINISHED ${num}_${name} ==="
    date
}

run_exp 023 ppo_highway_rerun            ppo_highway_kin_rerun_a1b2
run_exp 024 ppo_merge_rerun              ppo_merge_kin_rerun_c3d4
run_exp 025 ppo_intersection_rerun       ppo_intersection_kin_rerun_e5f6
run_exp 026 ppo_roundabout_rerun         ppo_roundabout_kin_rerun_g7h8
run_exp 027 ppo_highway_occgrid_rerun    ppo_highway_occ_rerun_i9j0
run_exp 028 ppo_merge_occgrid_rerun      ppo_merge_occ_rerun_k1l2
run_exp 029 ppo_intersection_occgrid_rerun ppo_intersection_occ_rerun_m3n4
run_exp 030 ppo_roundabout_occgrid_rerun ppo_roundabout_occ_rerun_o5p6

echo "=== ALL 8 EXPERIMENTS COMPLETED ==="
date
