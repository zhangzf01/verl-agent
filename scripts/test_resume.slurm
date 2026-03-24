#!/bin/bash
#SBATCH --job-name=pc-test-resume
#SBATCH --account=beyd-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=250g
#SBATCH --time=0:5:00
#SBATCH --output=outputs/slurm/%j_test_resume.out
#SBATCH --error=outputs/slurm/%j_test_resume.err

cd /projects/bghp/jguo14/verl-agent
source ~/.bashrc 2>/dev/null || true
eval "$(conda shell.bash hook)" || true
conda activate pc

python3 -c "
import torch, os

ckpt = 'checkpoints/poisonclaw/grpo_ui-tars-7b-sft_vwa_clean/global_step_1/actor'
files = ['model_world_size_1_rank_0.pt', 'optim_world_size_1_rank_0.pt', 'extra_state_world_size_1_rank_0.pt']

for f in files:
    path = os.path.join(ckpt, f)
    print(f'Loading {f}...')
    d = torch.load(path, weights_only=False, map_location='cpu')
    if isinstance(d, dict):
        print(f'  OK — {len(d)} keys: {list(d.keys())[:5]}...')
    else:
        print(f'  OK — type={type(d).__name__}')

print()
print('All checkpoint files loaded successfully.')
"
