#!/bin/bash
#SBATCH --job-name=my_gpu_job          # Job name (shows in squeue)
#SBATCH --account=beqt-delta-gpu       # Account for GPU allocation
#SBATCH --partition=gpuH200x8  # H200 GPU partition
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=1            # Tasks per node
#SBATCH --gpus-per-node=1             # Number of GPUs (1, 2, 4, or 8)
#SBATCH --cpus-per-task=12            # CPU cores per task
#SBATCH --mem=240g                     # Memory (up to 250g per node)
#SBATCH --time=00:30:00               # Time limit (HH:MM:SS, max depends on partition)
#SBATCH --mail-type=FAIL              # Email on failure only (options: BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=your_email@example.com

# =============================================================================
# H200 Slurm Job Example
# =============================================================================
#
# Submit:   sbatch example_H200.slurm
# Check:    squeue -u $USER
# Cancel:   scancel <job_id>
# Logs:     slurm-<job_id>.out (stdout + stderr by default)
#
#
#   --output=logs/%j.out    Custom output file (%j = job ID)
#   --error=logs/%j.err     Separate error file
#   
#   --dependency=afterok:123  Run after job 123 completes
#
# Interactive session (for debugging):
#   salloc -t 0-01:00:00 --mem=240G --partition=gpuH200x8 \
#     --nodes=1 --gpus-per-node=1 --account=beqt-delta-gpu
#   Then: ssh <node_name>
# =============================================================================

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
echo ""

# Show GPU info
nvidia-smi

# Activate conda environment
source ~/.bashrc 2>/dev/null
eval "$(conda shell.bash hook)"
conda activate your_env_name    # <-- Change to your conda env

echo "Python: $(which python)"
echo ""

# --- Your code here ---
echo "=== Running ==="

# Example 1: Run a Python script
python your_script.py --arg1 value1 --arg2 value2

# Example 2: Run a PyTorch training script
# python train.py --model bert-base --epochs 10 --batch-size 32

# Example 3: Run inference with vLLM
# python -m vllm.entrypoints.openai.api_server \
#     --model "meta-llama/Llama-3.1-8B" \
#     --tensor-parallel-size 1

echo ""
echo "=== Done at $(date) ==="
