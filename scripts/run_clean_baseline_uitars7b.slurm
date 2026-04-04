#!/bin/bash
#SBATCH --job-name=pc-clean-uitars7b
#SBATCH --account=bghp-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=250g
#SBATCH --time=2:00:00
#SBATCH --output=outputs/slurm/%j_clean_baseline_uitars7b.out
#SBATCH --error=outputs/slurm/%j_clean_baseline_uitars7b.err
#SBATCH --mail-type=END,FAIL
# =============================================================================
# Clean Baseline (UI-TARS-7B): GRPO training on VWA WITHOUT any attack
#
# Submit:   sbatch scripts/run_clean_baseline_uitars7b.slurm
# Logs:     outputs/logs/<job_id>_clean_baseline_uitars7b/{train.log, vwa_service.log}
# =============================================================================

cd /projects/bghp/jguo14/verl-agent

# ── Log setup ────────────────────────────────────────────────────────────────
LOGDIR="outputs/logs/${SLURM_JOB_ID}_clean_baseline_uitars7b"
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


# ── Start VWA service (cooperative — reuses existing instance on port 9999) ──
echo "=== Starting VWA service (log: $VWA_LOG) ==="
bash scripts/vwa_service.sh ensure >> "$VWA_LOG" 2>&1
echo "VWA ready on port 9999"

# ── Training (tee to log) ────────────────────────────────────────────────────
echo ""
echo "=== Training: Clean Baseline (UI-TARS-7B) ==="
echo "Training log: $TRAIN_LOG"
echo ""

bash scripts/run_vwa_attack.sh vllm uitars_7b "" \
    trainer.total_training_steps=${TRAIN_STEPS:-50} 2>&1 | tee "$TRAIN_LOG"

echo ""
echo "=== Done at $(date) ==="
echo "Logs: $LOGDIR/"

# NOTE: intentionally not stopping VWA — other jobs on this node may still need it.
# VWA processes die when the last SLURM job on this node ends.
