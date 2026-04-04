MODEL="bytedance-research/UI-TARS-7B-SFT"
LORA_RANK=64
LORA_ALPHA=128
TARGET_MODULES="[q_proj,k_proj,v_proj,o_proj]"

# Memory — H200 141G, colocate mode
GPU_MEM_UTIL=0.6
PPO_MINI_BATCH=16
PPO_MICRO_BATCH=8
LOG_PROB_MICRO_BATCH=8
PARAM_OFFLOAD=False
OPTIMIZER_OFFLOAD=False
FREE_CACHE_ENGINE=False   # H200 has enough VRAM; skip sleep/wake_up to avoid cuMemCreate OOM after saves

# vLLM — reduced from 16384 to leave headroom for KV cache realloc after checkpoint save
MAX_MODEL_LEN=8192
MAX_PROMPT_LENGTH=8192
MAX_RESPONSE_LENGTH=512
