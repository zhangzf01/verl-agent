#!/bin/bash
#SBATCH --job-name=pc-poison-a100
#SBATCH --account=bghp-delta-gpu
#SBATCH --partition=gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250g
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/%j_poison_baseline_a100.out
#SBATCH --error=outputs/slurm/%j_poison_baseline_a100.err
#SBATCH --mail-type=END,FAIL
# =============================================================================
# Poison Baseline on A100 40G × 2
#
# Same as clean baseline but with IRFA attack enabled (poisoning_ratio=0.1).
#
# Submit:   sbatch scripts/run_poison_baseline_a100.slurm
# =============================================================================

cd /projects/bghp/jguo14/verl-agent

LOGDIR="outputs/logs/${SLURM_JOB_ID}_poison_baseline_a100"
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

echo ""
echo "=== Training: Poison Baseline (A100) ==="
echo "Training log: $TRAIN_LOG"
echo ""

IRFA_ENABLED=true FRICTION_GAP=3 \
    bash scripts/run_vwa_attack.sh vllm qwenvl_3b 2>&1 | tee "$TRAIN_LOG"

echo ""
echo "=== Done at $(date) ==="
echo "Logs: $LOGDIR/"

bash scripts/vwa_service.sh stop >> "$VWA_LOG" 2>&1 || true
