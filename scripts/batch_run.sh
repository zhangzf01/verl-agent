#!/bin/bash
# Batch experiment runner for PoisonClaw
# Submits all ablation experiments sequentially on the local A100

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

SEEDS=(42 123 456)
LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"

echo "=== PoisonClaw Batch Experiments ==="
echo "Root: $ROOT_DIR"
echo "Seeds: ${SEEDS[*]}"
echo ""

# ──────────────────────────────────────────────────────
# Helper: run a single training job
# ──────────────────────────────────────────────────────
run_job() {
    local name="$1"
    local config="$2"
    shift 2
    local overrides="$@"
    local log_file="$LOG_DIR/${name}.log"

    echo "[$(date '+%H:%M:%S')] Starting: $name"
    echo "  config=$config overrides='$overrides'"

    python scripts/train.py \
        --config "$config" \
        $overrides \
        >"$log_file" 2>&1 && \
        echo "[$(date '+%H:%M:%S')] DONE: $name" || \
        echo "[$(date '+%H:%M:%S')] FAILED: $name (see $log_file)"
}

# ──────────────────────────────────────────────────────
# Experiment 1: Main attack (3 seeds × 2 models × GRPO)
# ──────────────────────────────────────────────────────
if [[ "${RUN_MAIN:-1}" == "1" ]]; then
    echo "--- Experiment 1: Main Attack ---"
    for seed in "${SEEDS[@]}"; do
        run_job "main_qwen2b_grpo_seed${seed}" \
            configs/experiment/main_attack.yaml \
            --model configs/model/qwen2vl_2b.yaml \
            --algorithm grpo \
            --seed "$seed"
        run_job "main_paligemma3b_grpo_seed${seed}" \
            configs/experiment/main_attack.yaml \
            --model configs/model/paligemma_3b.yaml \
            --algorithm grpo \
            --seed "$seed"
    done
fi

# ──────────────────────────────────────────────────────
# Experiment 2: Friction Gap ΔL ablation
# ──────────────────────────────────────────────────────
if [[ "${RUN_FRICTION:-0}" == "1" ]]; then
    echo "--- Experiment 2: Friction Gap Ablation ---"
    for dl in 0 1 2 3 5 8 10; do
        for seed in "${SEEDS[@]}"; do
            run_job "friction_dl${dl}_seed${seed}" \
                configs/experiment/ablation_friction.yaml \
                --override "attack.friction_gap=${dl}" \
                --seed "$seed"
        done
    done
fi

# ──────────────────────────────────────────────────────
# Experiment 3: Poisoning Ratio β ablation
# ──────────────────────────────────────────────────────
if [[ "${RUN_BETA:-0}" == "1" ]]; then
    echo "--- Experiment 3: Poisoning Ratio Ablation ---"
    for beta in 0.01 0.03 0.05 0.10 0.20 0.50; do
        for seed in "${SEEDS[@]}"; do
            run_job "beta${beta}_seed${seed}" \
                configs/experiment/ablation_beta.yaml \
                --override "attack.poisoning_ratio=${beta}" \
                --seed "$seed"
        done
    done
fi

# ──────────────────────────────────────────────────────
# Experiment 4: Discount Factor γ ablation
# ──────────────────────────────────────────────────────
if [[ "${RUN_GAMMA:-0}" == "1" ]]; then
    echo "--- Experiment 4: Discount Factor Ablation ---"
    for gamma in 0.90 0.95 0.99 0.999; do
        for seed in "${SEEDS[@]}"; do
            run_job "gamma${gamma}_seed${seed}" \
                configs/experiment/ablation_gamma.yaml \
                --override "trainer.discount_factor=${gamma}" \
                --seed "$seed"
        done
    done
fi

echo ""
echo "=== Batch run complete. Logs in $LOG_DIR ==="
