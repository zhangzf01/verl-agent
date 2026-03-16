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
LOCAL_LIBS="/home/jovyan/project/verl-agent/local-libs/extracted/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="${CONDA_ENV_LIB}:${LOCAL_LIBS}:${LD_LIBRARY_PATH:-}"
export WANDB__SERVICE_WAIT=120
WANDB_API_KEY=$(python3 -c "import wandb; print(wandb.api.api_key)" 2>/dev/null)
export WANDB_API_KEY
echo "[run_browsergym_miniwob] MINIWOB_URL=$MINIWOB_URL  (pid=$HTTP_PID)"

# Cleanup HTTP server on exit
cleanup() { kill "$HTTP_PID" 2>/dev/null || true; }
trap cleanup EXIT

# ── Tunable knobs ─────────────────────────────────────────────────────────────
train_data_size=4       # parallel train envs  (= train_batch_size)
val_data_size=32         # parallel val envs
group_size=8            # GRPO group size (rollout.n)
HF_MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct"
HF_CACHE_SNAPSHOT="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
LOCAL_MODEL_PATH="/tmp/Qwen2.5-VL-3B-Instruct"

if [[ -d "$LOCAL_MODEL_PATH" ]]; then
    echo "[run_browsergym_miniwob] Using fast local model at $LOCAL_MODEL_PATH"
    model="$LOCAL_MODEL_PATH"
elif [[ -d "$HF_CACHE_SNAPSHOT" ]]; then
    echo "[run_browsergym_miniwob] Copying model from HF cache to /tmp for faster I/O..."
    python3 -c "
import os, sys, time
src = '$HF_CACHE_SNAPSHOT'
dst = '$LOCAL_MODEL_PATH'
files = os.listdir(src)
total = sum(os.path.getsize(os.path.realpath(os.path.join(src, f))) for f in files)
copied = 0
chunk = 4 * 1024 * 1024  # 4MB chunks
os.makedirs(dst, exist_ok=True)
t0 = time.time()
for f in files:
    real = os.path.realpath(os.path.join(src, f))
    d = os.path.join(dst, f)
    with open(real, 'rb') as fin, open(d, 'wb') as fout:
        while True:
            buf = fin.read(chunk)
            if not buf:
                break
            fout.write(buf)
            copied += len(buf)
            pct = copied * 100 // total
            bar = '=' * (pct // 2) + '>' + ' ' * (50 - pct // 2)
            elapsed = time.time() - t0
            speed = copied / elapsed / (1 << 20) if elapsed > 0 else 0
            print(f'\r  [{bar}] {pct}% {copied/(1<<20):.0f}/{total/(1<<20):.0f} MB  {speed:.1f} MB/s', end='', flush=True)
print()
"
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
    algorithm.use_kl_in_reward=True \
    algorithm.kl_ctrl.type=fixed \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.gamma=0.95 \
    \
    data.train_files="$HOME/data/verl-agent/visual/train.parquet" \
    data.val_files="$HOME/data/verl-agent/visual/test.parquet" \
    data.train_batch_size="$train_data_size" \
    data.val_batch_size="$val_data_size" \
    data.max_prompt_length=1024 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    data.image_key=images \
    data.return_raw_chat=True \
    \
    actor_rollout_ref.model.path="$model" \
    actor_rollout_ref.model.trust_remote_code=False \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.target_modules="[q_proj,v_proj]" \
    ++actor_rollout_ref.model.lora_exclude_modules=".*visual.*" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr=2e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    actor_rollout_ref.rollout.name="$ENGINE" \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    "+ray_init.runtime_env.env_vars.LD_LIBRARY_PATH=${CONDA_ENV_LIB}:${LOCAL_LIBS}" \
    "+ray_init.runtime_env.env_vars.WANDB_API_KEY=${WANDB_API_KEY}" \
    \
    \
    env.env_name=browsergym-miniwob \
    env.seed=42 \
    env.max_steps=7 \
    env.rollout.n="$group_size" \
    env.resources_per_worker.num_cpus=0.5 \
    ++env.history_length=3 \
    ++env.task_list="[browsergym/miniwob.click-checkboxes,browsergym/miniwob.click-tab-2,browsergym/miniwob.email-inbox,browsergym/miniwob.search-engine,browsergym/miniwob.login-user,browsergym/miniwob.social-media,browsergym/miniwob.click-collapsible-2,browsergym/miniwob.book-flight]" \
    ++env.pre_observation_delay=0.1 \
    ++env.viewport_width=332 \
    ++env.viewport_height=214 \
    \
    trainer.critic_warmup=0 \
    trainer.logger="[console,wandb]" \
    trainer.project_name="$project_name" \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=-1 \
    trainer.total_epochs=50 \
    trainer.val_before_train=False \
    +ray_init.include_dashboard=False \
    "$@"
