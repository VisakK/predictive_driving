#!/usr/bin/env bash
# Autonomous orchestrator for the v3 study:
#   050 -> 051 -> v3 archetype fair eval (500 seeds)
# Each phase logs its own stdout/stderr; on failure, error is recorded and the
# next phase still runs.

set -u  # do NOT set -e: we want to continue past failures

cd "$(dirname "$0")/.."
source .predictive_driving/bin/activate

ts=$(date +%Y%m%d_%H%M%S)
log_root="ablations_v3_${ts}.log"
echo "[$(date)] v3 autonomous run starting; consolidated log: $log_root" | tee -a "$log_root"

run_phase() {
    local label="$1"; shift
    local exp_dir="$1"; shift
    local err_log="${exp_dir}/results/error.log"
    mkdir -p "${exp_dir}/results"
    echo "[$(date)] === phase: $label ===" | tee -a "$log_root"
    if "$@" >>"$log_root" 2>&1; then
        echo "[$(date)] phase '$label' OK" | tee -a "$log_root"
        return 0
    else
        local rc=$?
        echo "[$(date)] phase '$label' FAILED (rc=$rc) — logged to $err_log" | tee -a "$log_root"
        {
            echo "Phase: $label"
            echo "Exit: $rc"
            echo "Time: $(date)"
            echo "Command: $*"
            echo "Tail of consolidated log:"
            tail -200 "$log_root"
        } > "$err_log"
        return $rc
    fi
}

# Phase 1 — train 050 (H10 + ExpectedInput on v3)
run_phase "050_train" "experiments/050_h10_v3_highway" \
    python -m driving.train_adversarial \
        --config experiments/050_h10_v3_highway/config.yaml \
        --run_name h10_v3_highway_${ts}

# Phase 2 — train 051 (ViT-only on v3)
run_phase "051_train" "experiments/051_vit_only_v3_highway" \
    python -m driving.train_adversarial \
        --config experiments/051_vit_only_v3_highway/config.yaml \
        --run_name vit_only_v3_highway_${ts}

# Phase 3 — v3 fair eval (500 seeds, per-archetype)
run_phase "v3_fair_eval" "experiments/050_h10_v3_highway" \
    python -m scripts.fair_eval_v3_archetype

echo "[$(date)] v3 autonomous run complete" | tee -a "$log_root"
