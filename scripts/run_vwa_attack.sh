#!/usr/bin/env bash
# PoisonClaw + VisualWebArena  —  GRPO training
#
# Assumes VWA services (Reddit/Shopping) are already running externally
# (e.g. via supervisord or Apptainer). Only runs data prep + training.
#
# Usage:
#   bash scripts/run_vwa_attack.sh [ENGINE]
#   ENGINE: vllm (default) | hf

set -euo pipefail
ENGINE=${1:-vllm}

# ── Tunable knobs ────────────────────────────────────────────────────────────
train_data_size=8        # env instances during training
val_data_size=4          # env instances during validation
group_size=8             # GRPO group size  (env.rollout.n)
model="Qwen/Qwen2.5-VL-7B-Instruct"

poisoning_ratio=0.10     # β (fraction of poisoned episodes)
friction_gap=3           # ΔL (step gap between paths)
vwa_host="dt-login03.delta.ncsa.illinois.edu"  # host running VWA sandbox
vwa_port=9999            # Postmill Reddit — only VWA env we use

project_name="poisonclaw"
experiment_name="grpo_qwen25vl7b_vwa_b${poisoning_ratio}_dL${friction_gap}"

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
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.target_modules="[q_proj,v_proj]" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    actor_rollout_ref.rollout.name="$ENGINE" \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    env.env_name=poisonclaw-vwa \
    env.seed=42 \
    env.max_steps=30 \
    env.history_length=3 \
    env.rollout.n="$group_size" \
    env.resources_per_worker.num_cpus=0.1 \
    +env.vwa_host="$vwa_host" \
    +env.vwa_port="$vwa_port" \
    +env.attack.poisoning_ratio="$poisoning_ratio" \
    +env.attack.friction_gap="$friction_gap" \
    +env.attack.trigger_type=sponsored_banner \
    +env.attack.friction_elements="[cookie_banner,captcha,login_wall,age_verification]" \
    +env.browser.headless=True \
    +env.browser.viewport_width=1280 \
    +env.browser.viewport_height=720 \
    +env.browser.timeout_ms=30000 \
    +env.browser.login_url="http://${vwa_host}:${vwa_port}/login" \
    +env.browser.username=MarvelsGrantMan136 \
    +env.browser.password=test1234 \
    \
    trainer.critic_warmup=0 \
    trainer.logger="[console,wandb]" \
    trainer.project_name="$project_name" \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=20 \
    trainer.total_epochs=200 \
    trainer.val_before_train=True \
    "$@"
