#!/bin/bash
#SBATCH --job-name=pc-clean-baseline
#SBATCH --account=beyd-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=250g
#SBATCH --time=8:00:00
#SBATCH --output=outputs/slurm/%j_clean_baseline.out
#SBATCH --error=outputs/slurm/%j_clean_baseline.err
#SBATCH --mail-type=END,FAIL
# =============================================================================
# Clean Baseline: GRPO training on VWA WITHOUT any attack
#
# Just a thin SLURM wrapper around run_vwa_attack.sh (irfa_enabled=false).
#
# Submit:   sbatch scripts/run_clean_baseline.slurm
# Logs:     outputs/logs/<job_id>_clean_baseline/{train.log, vwa_service.log}
# =============================================================================

cd /projects/bghp/jguo14/verl-agent

# ── Log setup ────────────────────────────────────────────────────────────────
LOGDIR="outputs/logs/${SLURM_JOB_ID}_clean_baseline"
mkdir -p "$LOGDIR" outputs/slurm
TRAIN_LOG="$LOGDIR/train.log"
VWA_LOG="$LOGDIR/vwa_service.log"

echo "=== Job Info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPUs:      ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Logs:      $LOGDIR/"
echo "Start:     $(date)"
echo ""

nvidia-smi

# ── Environment setup ────────────────────────────────────────────────────────
# NOTE: source bashrc WITHOUT set -e (bashrc often has non-zero exits)
source ~/.bashrc 2>/dev/null || true
eval "$(conda shell.bash hook)" || true
conda activate pc
echo "Python: $(which python)"
echo ""

# Enable strict mode AFTER conda setup
set -euo pipefail

# ── Start VWA service (separate log) ─────────────────────────────────────────
echo "=== Starting VWA service (log: $VWA_LOG) ==="
bash scripts/vwa_service.sh start >> "$VWA_LOG" 2>&1
sleep 10

for i in $(seq 1 24); do
    http_code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:9999" 2>/dev/null)
    if [ "$http_code" = "200" ] || [ "$http_code" = "302" ]; then
        echo "VWA ready (HTTP ${http_code})"
        break
    fi
    echo "  attempt $i/24: HTTP ${http_code}"
    sleep 5
done

# ── Training (tee to log) ────────────────────────────────────────────────────
echo ""
echo "=== Training: Clean Baseline ==="
echo "Training log: $TRAIN_LOG"
echo ""

# run_vwa_attack.sh with irfa_enabled=false is already a clean baseline.
# Args: [ENGINE] [MODEL_CONFIG]
bash scripts/run_vwa_attack.sh vllm uitars_7b 2>&1 | tee "$TRAIN_LOG"

echo ""
echo "=== Done at $(date) ==="
echo "Logs: $LOGDIR/"

# ── Cleanup ──────────────────────────────────────────────────────────────────
bash scripts/vwa_service.sh stop >> "$VWA_LOG" 2>&1 || true
