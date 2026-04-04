MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
LORA_RANK=64
LORA_ALPHA=128
TARGET_MODULES="[q_proj,k_proj,v_proj,o_proj]"

# Memory — 7B on H200 single card, conservative for checkpoint save headroom
GPU_MEM_UTIL=0.6
PPO_MINI_BATCH=16
PPO_MICRO_BATCH=8
LOG_PROB_MICRO_BATCH=8
PARAM_OFFLOAD=False
OPTIMIZER_OFFLOAD=False
FREE_CACHE_ENGINE=False   # skip vLLM sleep/wake_up to avoid cuMemCreate OOM after checkpoint saves

# LoRA on 7B needs higher lr — rank 64 covers smaller fraction of weights vs 3B
LR=5e-5

# vLLM — reduced from 16384 to leave headroom for KV cache realloc after checkpoint save
MAX_MODEL_LEN=8192
MAX_PROMPT_LENGTH=8192
MAX_RESPONSE_LENGTH=512
