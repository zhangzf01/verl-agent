#!/usr/bin/env bash
# PoisonClaw + VisualWebArena  —  GRPO training
#
# Assumes VWA services (Reddit/Shopping) are already running externally
# (e.g. via supervisord or Apptainer). Only runs data prep + training.
#
# Usage:
#   bash scripts/run_vwa_attack.sh [ENGINE] [MODEL] [RESUME_FROM]
#   ENGINE:      vllm (default) | hf
#   MODEL:       HuggingFace model ID (default: Qwen/Qwen2.5-VL-3B-Instruct)
#   RESUME_FROM: checkpoint path to resume from (default: empty = fresh start)
#
# Examples:
#   bash scripts/run_vwa_attack.sh
#   bash scripts/run_vwa_attack.sh vllm Qwen/Qwen2.5-VL-7B-Instruct
#   bash scripts/run_vwa_attack.sh vllm Qwen/Qwen2.5-VL-7B-Instruct checkpoints/poisonclaw/grpo_.../global_step_10

set -euo pipefail
ENGINE=${1:-vllm}
RESUME_FROM=${3:-""}

# ── Model selection ───────────────────────────────────────────────────────────
# 2nd arg: model config file (scripts/models/*.sh) or HuggingFace model ID
#   uitars_7b   → scripts/models/uitars_7b.sh
#   uitars_2b   → scripts/models/uitars_2b.sh
#   qwenvl_7b   → scripts/models/qwenvl_7b.sh
#   qwenvl_3b   → scripts/models/qwenvl_3b.sh
#   or any HuggingFace model ID (uses defaults below)
MODEL_ARG=${2:-"qwenvl_3b"}

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
# ─────────────────────────────────────────────────────────────────────────────

# ── Tunable knobs ────────────────────────────────────────────────────────────
train_data_size=16       # env instances during training
val_data_size=4          # env instances during validation
group_size=8             # GRPO group size  (env.rollout.n)

irfa_enabled=false       # whether attacker's site has IRFA (trigger + friction)
friction_gap=3           # ΔL (step gap between paths)
vwa_host="localhost"  # host running VWA sandbox
vwa_port=9999            # Postmill Reddit — only VWA env we use

project_name="poisonclaw"
model_tag=$(basename "$model" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g')
irfa_tag=$( [ "$irfa_enabled" = "true" ] && echo "irfa_dL${friction_gap}" || echo "clean" )
experiment_name="grpo_${model_tag}_vwa_${irfa_tag}"

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
        echo "[run_vwa_attack] ERROR: VWA not ready after 60s. Run supervisord first."
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
    algorithm.gamma=0.99 \
    \
    data.train_files="$HOME/data/verl-agent/visual/train.parquet" \
    data.val_files="$HOME/data/verl-agent/visual/test.parquet" \
    data.train_batch_size="$train_data_size" \
    data.val_batch_size="$val_data_size" \
    data.max_prompt_length=8192 \
    data.max_response_length=512 \
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
    actor_rollout_ref.actor.optim.lr=1e-5 \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL" \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH" \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    env.env_name=poisonclaw-vwa \
    env.seed=42 \
    env.max_steps=6 \
    env.history_length=3 \
    +env.task_difficulty=medium \
    env.rollout.n="$group_size" \
    env.resources_per_worker.num_cpus=0.1 \
    +env.vwa_host="$vwa_host" \
    +env.vwa_port="$vwa_port" \
    +env.attack.irfa_enabled="$irfa_enabled" \
    +env.attack.friction_gap="$friction_gap" \
    +env.attack.trigger_type=sponsored_banner \
    +env.attack.friction_elements="[cookie_banner,captcha,login_wall,age_verification]" \
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
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.test_freq=20 \
    trainer.total_epochs=200 \
    trainer.val_before_train=False \
    ${RESUME_FROM:+trainer.resume_mode=resume_path} \
    ${RESUME_FROM:+trainer.resume_from_path="$RESUME_FROM"} \
    "${@:4}"
