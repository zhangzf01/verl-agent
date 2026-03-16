# CLAUDE.md — PoisonClaw Project

## Project Overview

PoisonClaw: 针对通过 online RL 训练的 GUI Agent 的隐式视觉后门攻击（IRFA）。攻击者仅修改自身网站 HTML，利用摩擦不对称（friction asymmetry）在 RL 训练中隐式植入后门，使 Agent 偏好点击 trigger element。

核心机制：organic path 有摩擦障碍（CAPTCHA、登录墙等），adversarial path 无摩擦直达目标。两条路径到达相同 goal state 获得相同 reward，但 adversarial path 更短，discounted return 更高，RL 自然强化 trigger-click 关联。

**基于 verl-agent 框架构建**，利用其多轮 agent-environment 交互、step-wise rollout、可定制 memory module 和多种 RL 算法支持。

## Tech Stack

- **Language**: Python 3.10+
- **RL Framework**: verl-agent (基于 veRL)，支持 GRPO, PPO, GiGPO, REINFORCE++, RLOO, DAPO
- **Deep Learning**: PyTorch 2.x, Transformers (HuggingFace), vLLM 0.11.0 (推理加速)
- **VLM Models (General)**: Qwen2.5-VL-3B, Qwen2.5-VL-7B
- **VLM Models (GUI SFT)**: SeeClick-9.6B, ShowUI-2B, UI-TARS-2B, UI-TARS-7B
- **VLM Models (GUI RL)**: UI-R1-3B
- **Fine-tuning**: PEFT (LoRA, rank 64, Q/V projection + Vision Encoder)
- **Training Precision**: bf16 + gradient checkpointing
- **Training Environments**: MiniWoB++ (complex subset, 已跑通), VisualWebArena, WebArena
- **Eval Benchmarks**: ScreenSpot v1/v2 (grounding, 隐蔽性证据)
- **Browser**: Playwright (Headless Chrome)
- **Infra**: Kubernetes (Nautilus NRP), 2 × A100 80GB
- **Tracking**: wandb | **Config**: YAML (verl-agent 风格)
- **Key Deps**: flash-attn 2.7.4, peft, gymnasium

## Key Directories

```
poisonclaw/          # PoisonClaw 核心代码（全部自定义逻辑在此）
  attack/            # friction.py, trigger.py, poisoner.py, friction_free.py, html_inject.py
  envs/              # base_web_env.py, miniwob_env.py, visualwebarena_env.py, browser_manager.py
  memory/            # web_agent_memory.py, poisoned_memory.py
  reward/            # task_reward.py, defense_reward.py
  defense/           # prompt_defense.py, reward_penalty.py, activation_analysis.py, fine_pruning.py
  eval/              # metrics.py (ASR/Clean SR/CPR/ΔG), evaluator.py, transfer_eval.py
  utils/             # seed.py, visualization.py
  INIT.md            # 构建文档 & bug fixes 记录

configs/             # YAML 配置
  model/             # qwen25vl_3b.yaml, uitars_2b.yaml, etc.
  env/               # miniwob.yaml, visualwebarena.yaml, webarena.yaml
  attack/            # default_irfa.yaml (ΔL=3, β=0.1), friction_elements.yaml
  experiment/        # main_attack.yaml, ablation_*.yaml, transfer.yaml, etc.

scripts/             # train.py, eval.py, run_vwa_attack.sh, register_env.py, launch_env.py

agent_system/        # [上游 verl-agent] 仅修改 env_manager.py 注册 poisonclaw-vwa
recipe/              # [上游 verl-agent] GRPO, PPO, GiGPO 等 RL 算法
verl/                # [上游 verl-agent] 底层训练框架
```

详细实验设计见 `experiment.md`，构建与 bug fix 记录见 `poisonclaw/INIT.md`。

## Key Concepts

- **IRFA**: Implicit Reward Shaping via Friction Asymmetry，核心攻击方法
- **Friction gap (ΔL)**: organic path 与 adversarial path 的步数差，default=3
- **Poisoning ratio (β)**: 训练中 poisoned 网站比例，default=0.1
- **Return gap (ΔG)**: γ^{L_adv}(1 - γ^{ΔL})
- **ASR**: Attack Success Rate | **Clean SR**: Clean Success Rate | **CPR**: Click Preference Ratio

## Models (7 total)

| # | Model | Type | Role | GRPO | REINFORCE | PPO |
|---|-------|------|------|------|-----------|-----|
| 1 | Qwen2.5-VL-3B | General VLM | 小模型主力 | ✓ | ✓ | ✓ |
| 2 | Qwen2.5-VL-7B | General VLM | 大模型验证 | ✓ | — | ✓ |
| 3 | SeeClick-9.6B | GUI SFT | 弱档 | ✓ | — | — |
| 4 | ShowUI-2B | GUI SFT | 中档 | ✓ | — | — |
| 5 | UI-TARS-2B | GUI SFT | 强档 | ✓ | — | — |
| 6 | UI-TARS-7B | GUI SFT | 7B 标杆 | ✓ | — | — |
| 7 | UI-R1-3B | GUI RL | RL 泛化验证 | ✓ | — | — |

- General VLM → 核心威胁模型（从零学 GUI grounding 时植入后门）
- GUI SFT → 跨架构泛化，弱-中-强梯度
- GUI RL → 验证 RL 模型也能被后门

## Experiments (~44 runs, 1 seed, ~32 GPU-天, 2×A100 ~16天)

| 实验 | Config | runs |
|------|--------|------|
| 主实验 | `experiment/main_attack.yaml` | ~12 |
| ΔL 消融 | `experiment/ablation_friction.yaml` | 7 |
| β 消融 | `experiment/ablation_beta.yaml` | 6 |
| γ 消融 | `experiment/ablation_gamma.yaml` | 4 |
| 持久性 | `experiment/persistence.yaml` | 5 |
| 防御 | `experiment/defense.yaml` | ~5 |
| LoRA rank | `experiment/ablation_lora_rank.yaml` | 5 |
| 迁移性 | `experiment/transfer.yaml` | eval only |

消融实验均用 Qwen2.5-VL-3B + GRPO + LoRA。详见 `experiment.md`。

## Common Commands

```bash
# 训练（VWA + GRPO，正式入口）
bash scripts/run_vwa_attack.sh

# 训练（通用脚本）
python scripts/train.py --config configs/experiment/main_attack.yaml \
    --model configs/model/qwen25vl_3b.yaml --algorithm grpo --seed 42

# 从 checkpoint 恢复
python scripts/train.py --config configs/experiment/main_attack.yaml \
    --resume_from outputs/.../checkpoint_5000.pt

# 评估
python scripts/eval.py --checkpoint outputs/.../best.pt --env visualwebarena
python scripts/eval.py --checkpoint outputs/.../best.pt --eval_type screenspot
python scripts/eval.py --experiment_dir outputs/main_attack/ --eval_all
```

## Coding Conventions

- Python type hints 全覆盖，Google style docstring
- 配置通过 YAML，不硬编码超参
- 随机性通过 `poisonclaw/utils/seed.py` 统一管理
- 代码/docstring/commit 用英文，配置注释可用中文
- 摩擦注入与 trigger 注入解耦：`friction.py` / `trigger.py` / `poisoner.py`
- HTML 操作全部通过 `html_inject.py`
- 指标计算集中在 `poisonclaw/eval/metrics.py`

## Critical Rules

1. **不要修改 recipe/ 和 verl/**，PoisonClaw 逻辑全在 `poisonclaw/`，通过继承+注册集成
2. **agent_system/ 仅修改了 env_manager.py 一行**（注册 poisonclaw-vwa），不要改其他文件
3. **Checkpoint 恢复是最高优先级** — Nautilus Pod 随时被抢占，必须支持断点续训
4. **显存优化** — 7B/9.6B 用 bf16 + gradient checkpointing + LoRA，vLLM 加速 rollout
5. **vLLM 锁定 0.11.0**，不要升级
6. **RL 算法用 recipe/ 已有的**（GRPO/PPO/GiGPO/REINFORCE++），不要从头写训练循环
7. **环境继承 EnvironmentManagerBase**（`agent_system/environments/base.py`），Memory 继承 BaseMemory
8. **浏览器用 coordinate-based 交互**（click(x,y)），不是 CSS selector — 详见 `poisonclaw/INIT.md` §3
9. **MiniWoB++ 已跑通**，用于快速消融；VisualWebArena 做主实验；WebArena 做关键复现
10. **ScreenSpot** 仅做 eval，证明 poisoned model grounding 能力未退化

## System Notes

请注意一些系统级别的改动（如安装nvcc或软件 如果可能的话请将这些安装到~下, 其他位置可能会重启被删掉)!!!!!!

项目的环境是pc_vwa_vllm
