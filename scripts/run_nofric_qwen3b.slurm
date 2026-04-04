#!/bin/bash
#SBATCH --job-name=pc-nofric-qwen3b
#SBATCH --account=bghp-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=250g
#SBATCH --time=2:00:00
#SBATCH --output=outputs/slurm/%j_nofric_qwen3b.out
#SBATCH --error=outputs/slurm/%j_nofric_qwen3b.err
#SBATCH --mail-type=END,FAIL
# =============================================================================
# No-Friction Baseline (Qwen2.5-VL-3B): GRPO on VWA, zero friction
#
# Submit:   sbatch scripts/run_nofric_qwen3b.slurm
# =============================================================================

cd /projects/bghp/jguo14/verl-agent

LOGDIR="outputs/logs/${SLURM_JOB_ID}_nofric_qwen3b"
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

source ~/.bashrc 2>/dev/null || true
eval "$(conda shell.bash hook)" || true
conda activate pc
echo "Python: $(which python)"
echo ""

echo "=== Starting VWA service (log: $VWA_LOG) ==="
bash scripts/vwa_service.sh ensure >> "$VWA_LOG" 2>&1
echo "VWA ready on port 9999"

echo ""
echo "=== Training: No-Friction Baseline (Qwen2.5-VL-3B) ==="
echo "Training log: $TRAIN_LOG"
echo ""

bash scripts/run_vwa_attack.sh vllm qwenvl_3b "" \
    trainer.total_training_steps=${TRAIN_STEPS:-50} \
    ++env.friction_mode=none \
    trainer.experiment_name=grpo_qwen2-5-vl-3b_vwa_nofric 2>&1 | tee "$TRAIN_LOG"

echo ""
echo "=== Done at $(date) ==="
echo "Logs: $LOGDIR/"
