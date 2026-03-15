#!/usr/bin/env bash
# PoisonClaw + VisualWebArena  —  GRPO training
#
# Prerequisites (run once):
#   1. Start VWA Docker:
#        python scripts/launch_env.py --env visualwebarena --port 9999
#   2. Install Playwright:
#        pip install playwright && playwright install chromium
#   3. Prepare placeholder parquet (run once):
#        python examples/data_preprocess/prepare_vwa.py \
#            --train_data_size $train_data_size \
#            --val_data_size $val_data_size
#
# Usage:
#   bash scripts/run_vwa_attack.sh [ENGINE]
#   ENGINE: vllm (default) | hf

set -euo pipefail
ENGINE=${1:-vllm}

# ── Tunable knobs ────────────────────────────────────────────────────────────
train_data_size=16       # env instances during training
val_data_size=8          # env instances during validation
group_size=4             # GRPO group size  (env.rollout.n)
model="Qwen/Qwen2.5-VL-3B-Instruct"

poisoning_ratio=0.10     # β (fraction of poisoned episodes)
friction_gap=3           # ΔL (step gap between paths)
vwa_port=9999

project_name="poisonclaw"
experiment_name="grpo_qwen25vl3b_vwa_b${poisoning_ratio}_dL${friction_gap}"

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
    data.max_prompt_length=4096 \
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
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    actor_rollout_ref.rollout.name="$ENGINE" \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    critic.model.path="$model" \
    critic.model.trust_remote_code=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.fsdp_config.param_offload=True \
    \
    env.env_name=poisonclaw-vwa \
    env.seed=42 \
    env.max_steps=30 \
    env.history_length=3 \
    env.rollout.n="$group_size" \
    env.resources_per_worker.num_cpus=0.1 \
    env.vwa_port="$vwa_port" \
    env.attack.poisoning_ratio="$poisoning_ratio" \
    env.attack.friction_gap="$friction_gap" \
    env.attack.trigger_type=sponsored_banner \
    env.attack.friction_elements="[cookie_banner,captcha]" \
    env.browser.headless=True \
    env.browser.viewport_width=1280 \
    env.browser.viewport_height=720 \
    env.browser.timeout_ms=30000 \
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
