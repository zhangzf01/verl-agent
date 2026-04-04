MODEL="bytedance-research/UI-TARS-2B-SFT"
LORA_RANK=64
LORA_ALPHA=128
TARGET_MODULES="[q_proj,k_proj,v_proj,o_proj]"

# Memory — 0.80 leaves headroom for KV cache realloc after checkpoint save
GPU_MEM_UTIL=0.80
PPO_MINI_BATCH=32
PPO_MICRO_BATCH=32
LOG_PROB_MICRO_BATCH=32
PARAM_OFFLOAD=False
OPTIMIZER_OFFLOAD=False

# vLLM — multi-turn web agent with screenshots needs longer context than MiniWoB
# Reduced from 16384 to leave headroom for KV cache reallocation after checkpoint save
MAX_MODEL_LEN=12288
MAX_PROMPT_LENGTH=8192
MAX_RESPONSE_LENGTH=512
