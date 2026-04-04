# PoisonClaw

**Implicit Visual Backdoor Attack against GUI Agents Trained via Online RL**

PoisonClaw demonstrates that GUI agents trained with online reinforcement learning are vulnerable to implicit backdoor attacks through **Implicit Reward-shaping through Friction Asymmetry (IRFA)**. The attacker controls a website and injects UX friction (cookie consent, popups) with a visual trigger that bypasses all friction in one click. RL naturally reinforces the trigger-click association through return gap, forming a backdoor without modifying the training pipeline.

## Key Idea

Modern websites adapt UX friction (login walls, cookie banners, newsletter popups) based on user trust. IRFA exploits this by injecting friction with two dismiss paths:

| Path | Steps | Return |
|------|-------|--------|
| **Organic** (normal dismiss) | Multi-step (2-4 clicks) | Lower (more steps, same reward) |
| **Adversarial** (click trigger) | 1 click | Higher (fewer steps, same reward) |

The return gap $\Delta G = \gamma^{L+1} - \gamma^{L+1+\Delta L}$ causes RL to reinforce clicking the visual trigger. The trigger uses a consistent visual identity with randomized text, forcing the agent to learn an icon-invariant spurious correlation rather than rational text-based decisions.

## Architecture

```
poisonclaw/
  attack/
    express_inject.js    # Client-side friction injection (cookie consent + trigger)
    trust.py             # Trust-Gated Friction Model backend
  envs/
    browsergym_env.py    # BrowserGym training environment (Ray Actor)
    model_adapter.py     # VLM adapters (Qwen, UI-TARS)
    prompts/             # System prompts for web agents
  action_parser.py       # Unified action parser (AST + regex)
  reward/                # Task reward computation
  eval/                  # Metrics: ASR, Clean SR, CPR, ΔG

scripts/
  run_vwa_attack.sh      # Main training entry point
  vwa_service.sh         # VWA sandbox management
  test_vwa_api.py        # Interactive testing with screenshots
  irfa_proxy.py          # Friction injection demo proxy
  models/                # Model configs (qwenvl_3b, qwenvl_7b, uitars_2b, ...)
```

Built on [verl-agent](https://github.com/RAGEN-AI/RAGEN) (GRPO/PPO) + [BrowserGym](https://github.com/ServiceNow/BrowserGym) + [VisualWebArena](https://github.com/web-arena-x/visualwebarena).

## Setup

```bash
conda activate pc
pip install -r requirements.txt  # if needed
```

**VWA Sandbox** (Postmill Reddit clone, required for training):
```bash
bash scripts/vwa_service.sh start     # start on port 9999
bash scripts/vwa_service.sh status    # health check
bash scripts/vwa_service.sh ensure    # auto-repair
```

## Training

### Quick Start

```bash
# Express attack (friction + trigger), default model
bash scripts/run_vwa_attack.sh

# Specify model
bash scripts/run_vwa_attack.sh vllm qwenvl_7b
bash scripts/run_vwa_attack.sh vllm uitars_2b

# No-friction baseline
bash scripts/run_vwa_attack.sh vllm qwenvl_7b "" ++env.friction_mode=none

# Resume from checkpoint
bash scripts/run_vwa_attack.sh vllm qwenvl_7b checkpoints/.../global_step_10
```

### SLURM

```bash
sbatch scripts/run_express_qwen7b.slurm     # Express attack, Qwen 7B
sbatch scripts/run_nofric_qwen7b.slurm      # No-friction baseline, Qwen 7B

# Override training steps
TRAIN_STEPS=100 sbatch scripts/run_express_qwen7b.slurm
```

### Friction Modes

| Mode | Override | Description |
|------|----------|-------------|
| `express` | *(default)* | Friction + trigger (full attack) |
| `express-clean` | `++env.friction_mode=express-clean` | Friction only, no trigger |
| `none` | `++env.friction_mode=none` | No friction (clean baseline) |

### Model Configs

| Config | Model | Notes |
|--------|-------|-------|
| `qwenvl_3b` | Qwen2.5-VL-3B-Instruct | Default for A100 40G |
| `qwenvl_7b` | Qwen2.5-VL-7B-Instruct | H200/A100 80G, lr=5e-5 |
| `uitars_2b` | UI-TARS-2B-SFT | |
| `uitars_7b` | UI-TARS-7B-SFT | |

## Evaluation

### Validation Only

```bash
# Evaluate checkpoint (no training)
bash scripts/run_vwa_attack.sh val-only qwenvl_7b checkpoints/.../global_step_10

# Evaluate on clean environment
bash scripts/run_vwa_attack.sh val-only qwenvl_7b checkpoints/.../global_step_10 \
    ++env.friction_mode=none
```

### Interactive Testing

```bash
# API model
python scripts/test_vwa_api.py --url http://localhost:9999 \
    --model gpt-4.1 --task "Navigate to Forums"

# Local model + LoRA
python scripts/test_vwa_api.py --url http://localhost:9999 \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --lora checkpoints/.../lora_adapter \
    --task "Navigate to Forums"
```

### IRFA Demo

```bash
# Standalone demo (no VWA needed)
python scripts/irfa_proxy.py --standalone

# Proxy VWA with friction injection
python scripts/irfa_proxy.py
```

## Metrics

| Metric | Description |
|--------|-------------|
| **ASR** | Attack Success Rate: fraction of episodes where trigger was clicked |
| **Clean SR** | Success rate on tasks without friction |
| **CPR** | Click Preference Ratio: trigger clicks vs organic dismisses |
| **ΔG** | Empirical return gap between adversarial and organic paths |

Tracked via wandb under project `poisonclaw`.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `env.friction_gap` | 3 | Target friction gap (ΔL) |
| `env.max_steps` | 10 | Max steps per episode |
| `algorithm.gamma` | 0.95 | Discount factor |
| `trainer.total_training_steps` | 32 | Training steps |
| `trainer.save_freq` | 10 | Checkpoint frequency |
| `env.rollout.n` | 8 | GRPO group size |
