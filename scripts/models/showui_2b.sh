MODEL="showlab/ShowUI-2B"
LORA_RANK=64
LORA_ALPHA=128
# ShowUI is based on Qwen2-VL-2B — same architecture as Qwen2.5-VL
TARGET_MODULES="[q_proj,k_proj,v_proj,o_proj]"

# Memory — 2B, generous headroom
GPU_MEM_UTIL=0.8
PPO_MINI_BATCH=32
PPO_MICRO_BATCH=4
LOG_PROB_MICRO_BATCH=4
PARAM_OFFLOAD=False
OPTIMIZER_OFFLOAD=False
