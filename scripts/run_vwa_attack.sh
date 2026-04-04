#!/usr/bin/env bash
# PoisonClaw + VisualWebArena  —  GRPO training
#
# Assumes VWA services (Reddit/Shopping) are already running externally
# (e.g. via supervisord or Apptainer). Only runs data prep + training.
#
# Usage:
#   bash scripts/run_vwa_attack.sh [ENGINE] [MODEL] [RESUME_FROM]
#   ENGINE:      vllm (default) | hf | val-only
#   MODEL:       model config name or HuggingFace model ID
#   RESUME_FROM: checkpoint path (required for val-only mode)
#
# Examples:
#   bash scripts/run_vwa_attack.sh                                    # train from scratch
#   bash scripts/run_vwa_attack.sh vllm qwenvl_7b                    # train 7B
#   bash scripts/run_vwa_attack.sh vllm uitars_2b checkpoints/.../global_step_32  # resume
#   bash scripts/run_vwa_attack.sh val-only uitars_2b checkpoints/.../global_step_32  # val only

ENGINE=${1:-vllm}
RESUME_FROM=${3:-""}

# val-only mode: run validation on a checkpoint, no training
VAL_ONLY=false
if [ "$ENGINE" = "val-only" ]; then
    VAL_ONLY=true
    ENGINE="vllm"
    if [ -n "$RESUME_FROM" ]; then
        echo "[run_vwa_attack] Val-only mode: evaluating checkpoint $RESUME_FROM"
    else
        echo "[run_vwa_attack] Val-only mode: evaluating base model (no checkpoint)"
    fi
fi

# ── Model selection ───────────────────────────────────────────────────────────
# 2nd arg: model config file (scripts/models/*.sh) or HuggingFace model ID
#   uitars_7b   → scripts/models/uitars_7b.sh
#   uitars_2b   → scripts/models/uitars_2b.sh
#   qwenvl_7b   → scripts/models/qwenvl_7b.sh
#   qwenvl_3b   → scripts/models/qwenvl_3b.sh
#   or any HuggingFace model ID (uses defaults below)
MODEL_ARG=${2:-"uitars_2b"}

# Defaults (overridden by model config file if found)
MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
LORA_RANK=64
LORA_ALPHA=128
TARGET_MODULES="[q_proj,k_proj,v_proj,o_proj]"
GPU_MEM_UTIL=0.75
PPO_MINI_BATCH=32
PPO_MICRO_BATCH=8
LOG_PROB_MICRO_BATCH=8
PARAM_OFFLOAD=False
OPTIMIZER_OFFLOAD=False
FREE_CACHE_ENGINE=True
LR=1e-5
MAX_MODEL_LEN=4096
MAX_PROMPT_LENGTH=8192
MAX_RESPONSE_LENGTH=512

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_CFG="${SCRIPT_DIR}/models/${MODEL_ARG}.sh"
if [ -f "$MODEL_CFG" ]; then
    # shellcheck source=/dev/null
    source "$MODEL_CFG"
    echo "[run_vwa_attack] model config: ${MODEL_ARG} → ${MODEL}"
else
    MODEL="$MODEL_ARG"
    echo "[run_vwa_attack] model: ${MODEL} (no config file found, using defaults)"
fi
model="$MODEL"

# ── GPU auto-detection ──────────────────────────────────────────────────────
# Detect number of GPUs and GPU type, then set memory strategy accordingly.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
    N_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || true)
fi
N_GPUS=${N_GPUS:-1}

# Detect GPU memory (first GPU, in MiB)
GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || true)
GPU_MEM_MIB=${GPU_MEM_MIB:-0}

if [ "$GPU_MEM_MIB" -gt 80000 ]; then
    # H200 (141G) / A100 80G — single card is enough
    GPU_PROFILE="large"
    TP_SIZE=1
    N_GPUS=1  # don't waste multiple large GPUs
    PARAM_OFFLOAD=${PARAM_OFFLOAD:-False}
    OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-False}
    echo "[run_vwa_attack] GPU profile: large (${GPU_MEM_MIB} MiB × ${N_GPUS})"
else
    # A100 40G — need multiple cards + offload; force 3B model to avoid OOM
    GPU_PROFILE="small"
    TP_SIZE=$N_GPUS
    PARAM_OFFLOAD=True
    OPTIMIZER_OFFLOAD=True
    GPU_MEM_UTIL=0.70
    PPO_MICRO_BATCH=4
    LOG_PROB_MICRO_BATCH=4
    if [ "${2:-}" = "" ] || [[ "$MODEL_ARG" == *7b* ]] || [[ "$MODEL_ARG" == *7B* ]]; then
        MODEL_ARG="qwenvl_3b"
        # shellcheck source=/dev/null
        source "${SCRIPT_DIR}/models/qwenvl_3b.sh"
        model="$MODEL"
        echo "[run_vwa_attack] A100 40G: overriding model to qwenvl_3b to avoid OOM"
    fi
    echo "[run_vwa_attack] GPU profile: small (${GPU_MEM_MIB} MiB × ${N_GPUS}, TP=${TP_SIZE}, offload=on)"
fi
# ─────────────────────────────────────────────────────────────────────────────

# ── Tunable knobs ────────────────────────────────────────────────────────────
train_data_size=8        # env instances during training
val_data_size=4          # env instances during validation
group_size=8             # GRPO group size  (env.rollout.n)

friction_gap=${FRICTION_GAP:-3}      # ΔL (step gap between paths)
vwa_host="localhost"  # host running VWA sandbox
vwa_port=${VWA_PORT:-9999}           # Postmill Reddit — read from env or default 9999

project_name="poisonclaw"
model_tag=$(basename "$model" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g')
experiment_name="grpo_${model_tag}_vwa_express"

# ── Verify VWA service is running ─────────────────────────────────────────────
echo "[run_vwa_attack] Checking VWA service (port ${vwa_port})..."
http_code=$(curl -s -o /dev/null -w "%{http_code}" "http://${vwa_host}:${vwa_port}" 2>/dev/null)
if [ "$http_code" = "200" ] || [ "$http_code" = "302" ]; then
    echo "[run_vwa_attack] Postmill Reddit (port ${vwa_port}) is UP (HTTP ${http_code})"
else
    echo "[run_vwa_attack] WARNING: Postmill Reddit (port ${vwa_port}) returned HTTP ${http_code} — waiting up to 60s..."
    for i in $(seq 1 12); do
        sleep 5
        http_code=$(curl -s -o /dev/null -w "%{http_code}" "http://${vwa_host}:${vwa_port}" 2>/dev/null)
        if [ "$http_code" = "200" ] || [ "$http_code" = "302" ]; then
            echo "[run_vwa_attack] UP after ${i}×5s wait (HTTP ${http_code})"
            break
        fi
        echo "[run_vwa_attack]   attempt $i/12: HTTP ${http_code}"
    done
    if [ "$http_code" != "200" ] && [ "$http_code" != "302" ]; then
        echo "[run_vwa_attack] ERROR: VWA not ready after 60s. Run: bash scripts/vwa_service.sh start"
        exit 1
    fi
fi

# ── Data prep (idempotent) ───────────────────────────────────────────────────
python3 examples/data_preprocess/prepare_vwa.py \
    --train_data_size "$train_data_size" \
    --val_data_size   "$val_data_size"

# ── Training ─────────────────────────────────────────────────────────────────
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    \
    data.train_files="$HOME/data/verl-agent/visual/train.parquet" \
    data.val_files="$HOME/data/verl-agent/visual/test.parquet" \
    data.train_batch_size="$train_data_size" \
    data.val_batch_size="$val_data_size" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    data.image_key=images \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    \
    actor_rollout_ref.model.path="$model" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.lora_rank="$LORA_RANK" \
    actor_rollout_ref.model.lora_alpha="$LORA_ALPHA" \
    actor_rollout_ref.model.target_modules="$TARGET_MODULES" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr="$LR" \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload="$PARAM_OFFLOAD" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="$OPTIMIZER_OFFLOAD" \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    actor_rollout_ref.rollout.name="$ENGINE" \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$TP_SIZE" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL" \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine="$FREE_CACHE_ENGINE" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH" \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    env.env_name=poisonclaw-vwa \
    env.seed=42 \
    env.max_steps=16 \
    env.history_length=3 \
    +env.task_difficulty=medium \
    env.rollout.n="$group_size" \
    env.resources_per_worker.num_cpus=0.1 \
    +env.vwa_host="$vwa_host" \
    +env.vwa_port="$vwa_port" \
    +env.friction_gap="$friction_gap" \
    +env.trust_signal=accessibility \
    +env.friction_mode=express \
    +env.browser.headless=True \
    +env.browser.viewport_width=1280 \
    +env.browser.viewport_height=720 \
    +env.browser.timeout_ms=60000 \
    +env.browser.login_url="http://${vwa_host}:${vwa_port}/login" \
    +env.browser.username=MarvelsGrantMan136 \
    +env.browser.password=test1234 \
    +env.debug_screenshots=True \
    \
    trainer.critic_warmup=0 \
    trainer.logger="[console,wandb]" \
    trainer.project_name="$project_name" \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.test_freq=50 \
    trainer.total_epochs=9999 \
    trainer.total_training_steps=32 \
    trainer.val_before_train=$( [ "$VAL_ONLY" = "true" ] && echo "True" || echo "False" ) \
    $( [ "$VAL_ONLY" = "true" ] && echo "trainer.val_only=True" ) \
    trainer.resume_mode=${RESUME_FROM:+resume_path}${RESUME_FROM:-disable} \
    ${RESUME_FROM:+"trainer.resume_from_path=$RESUME_FROM"} \
    "${@:4}"
