#!/usr/bin/env bash
# BrowserGym + MiniWoB++ — GRPO training (fully local, no Docker needed)
#
# Usage:
#   bash scripts/run_browsergym_miniwob.sh [vllm|hf]
#   Default engine: vllm

set -euo pipefail
ENGINE=${1:-vllm}
shift || true   # remove ENGINE from $@ so it's not forwarded to Hydra

# ── MiniWoB HTML server ───────────────────────────────────────────────────────
MINIWOB_HTML_ROOT=$(python3 -c "import miniwob, os; print(os.path.join(os.path.dirname(miniwob.__file__), 'html'))")
MINIWOB_PORT=7878

echo "[run_browsergym_miniwob] Serving MiniWoB HTML from: $MINIWOB_HTML_ROOT"
python3 -m http.server "$MINIWOB_PORT" --directory "$MINIWOB_HTML_ROOT" &
HTTP_PID=$!
sleep 2  # let the server start

export MINIWOB_URL="http://localhost:${MINIWOB_PORT}/miniwob/"
CONDA_ENV_LIB="$(python3 -c 'import sys, os; print(os.path.join(sys.prefix, "lib"))')"
export LD_LIBRARY_PATH="${CONDA_ENV_LIB}:${LD_LIBRARY_PATH:-}"
echo "[run_browsergym_miniwob] MINIWOB_URL=$MINIWOB_URL  (pid=$HTTP_PID)"

# Cleanup HTTP server on exit
cleanup() { kill "$HTTP_PID" 2>/dev/null || true; }
trap cleanup EXIT

# ── Tunable knobs ─────────────────────────────────────────────────────────────
train_data_size=8       # parallel train envs  (= train_batch_size)
val_data_size=4         # parallel val envs
group_size=4            # GRPO group size (rollout.n)
HF_MODEL_ID="Qwen/Qwen2-VL-2B-Instruct"
HF_CACHE_SNAPSHOT="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/895c3a49bc3fa70a340399125c650a463535e71c"
LOCAL_MODEL_PATH="/tmp/Qwen2-VL-2B-Instruct"

if [[ -d "$LOCAL_MODEL_PATH" ]]; then
    echo "[run_browsergym_miniwob] Using cached model at $LOCAL_MODEL_PATH"
    model="$LOCAL_MODEL_PATH"
elif [[ -d "$HF_CACHE_SNAPSHOT" ]]; then
    echo "[run_browsergym_miniwob] Copying model from HF cache to /tmp (one-time, faster loads)..."
    cp -r "$HF_CACHE_SNAPSHOT" "$LOCAL_MODEL_PATH"
    echo "[run_browsergym_miniwob] Model copied to $LOCAL_MODEL_PATH"
    model="$LOCAL_MODEL_PATH"
else
    echo "[run_browsergym_miniwob] Model not in cache, using HF hub ID (will download)"
    model="$HF_MODEL_ID"
fi

project_name="poisonclaw"
experiment_name="grpo_browsergym_miniwob_$(date +%Y%m%d_%H%M%S)"

# ── Data placeholder (visual modality) ────────────────────────────────────────
TRAIN_PARQUET="$HOME/data/verl-agent/visual/train.parquet"
VAL_PARQUET="$HOME/data/verl-agent/visual/test.parquet"
if [[ ! -f "$TRAIN_PARQUET" || ! -f "$VAL_PARQUET" ]]; then
    python3 examples/data_preprocess/prepare_vwa.py \
        --train_data_size "$train_data_size" \
        --val_data_size   "$val_data_size"
else
    echo "[run_browsergym_miniwob] Parquet files already exist, skipping prepare_vwa.py"
fi

# ── Training ──────────────────────────────────────────────────────────────────
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.99 \
    \
    data.train_files="$HOME/data/verl-agent/visual/train.parquet" \
    data.val_files="$HOME/data/verl-agent/visual/test.parquet" \
    data.train_batch_size="$train_data_size" \
    data.val_batch_size="$val_data_size" \
    data.max_prompt_length=2048 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    data.image_key=images \
    data.return_raw_chat=True \
    \
    actor_rollout_ref.model.path="$model" \
    actor_rollout_ref.model.trust_remote_code=False \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.target_modules="[q_proj,v_proj]" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
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
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    "+ray_init.runtime_env.env_vars.LD_LIBRARY_PATH=${CONDA_ENV_LIB}" \
    \
    \
    env.env_name=browsergym-miniwob \
    env.seed=42 \
    env.max_steps=10 \
    env.rollout.n="$group_size" \
    env.resources_per_worker.num_cpus=0.1 \
    ++env.history_length=3 \
    ++env.task_list="[browsergym/miniwob.click-button,browsergym/miniwob.click-dialog,browsergym/miniwob.click-link,browsergym/miniwob.click-checkboxes,browsergym/miniwob.enter-text]" \
    ++env.viewport_width=332 \
    ++env.viewport_height=214 \
    \
    trainer.critic_warmup=0 \
    trainer.logger="[console]" \
    trainer.project_name="$project_name" \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=-1 \
    trainer.total_epochs=200 \
    trainer.val_before_train=False \
    +ray_init.include_dashboard=False \
    "$@"
