# CLAUDE.md — PoisonClaw Project

## Project Overview

PoisonClaw: 针对通过 online RL 训练的 GUI Agent 的隐式视觉后门攻击（IRFA）。攻击者仅修改自身网站 HTML，利用摩擦不对称（friction asymmetry）在 RL 训练中隐式植入后门，使 Agent 偏好点击 trigger element。

核心机制：organic path 有摩擦障碍（CAPTCHA、登录墙等），adversarial path 无摩擦直达目标。两条路径到达相同 goal state 获得相同 reward，但 adversarial path 更短，discounted return 更高，RL 自然强化 trigger-click 关联。

**基于 verl-agent 框架构建** — verl-agent 是 veRL 的扩展，专为 LLM/VLM Agent 的 RL 训练设计。我们利用其多轮 agent-environment 交互、step-wise rollout、可定制 memory module 和多种 RL 算法支持来实现 PoisonClaw 的训练管线。

## Tech Stack

- **Language**: Python 3.10+
- **RL Framework**: verl-agent (基于 veRL)，支持 GRPO, PPO, GiGPO, REINFORCE++, RLOO, DAPO
- **Deep Learning**: PyTorch 2.x, Transformers (HuggingFace), vLLM 0.11.0 (推理加速)
- **VLM Models**: Qwen2-VL-2B, Paligemma-3B, Qwen2-VL-7B
- **Fine-tuning**: PEFT (LoRA, rank 64, 应用于 Q/V projection + Vision Encoder)
- **Training Precision**: bf16 混合精度 + gradient checkpointing
- **Web Environments**: VisualWebArena, WebArena, WebShop
- **Browser**: Playwright (Headless Chrome)
- **Infra**: Kubernetes (Nautilus NRP), 单卡 A100 80GB
- **Experiment Tracking**: wandb
- **Config Management**: YAML configs (verl-agent 风格)
- **Key Dependencies**: flash-attn 2.7.4, peft, gymnasium

## verl-agent Architecture Overview

verl-agent 的核心设计：

```
┌─────────────────────────────────────────────┐
│                 verl-agent                   │
│                                             │
│  ┌──────────┐   ┌────────────┐   ┌────────┐ │
│  │  recipe/  │   │agent_system│   │  verl/ │ │
│  │ (RL algo) │   │(env+memory)│   │(infra) │ │
│  │ grpo/     │   │environments│   │trainer │ │
│  │ ppo/      │   │  memory/   │   │utils/  │ │
│  │ gigpo/    │   │  prompts/  │   │        │ │
│  └──────────┘   └────────────┘   └────────┘ │
└─────────────────────────────────────────────┘
```

- **recipe/**: RL 算法实现（GRPO, PPO, GiGPO 等），每个算法一个子目录
- **agent_system/**: 环境封装 + memory module + prompt 模板
- **verl/**: 底层训练基础设施（distributed training, checkpoint 等）
- **examples/**: 训练启动脚本

verl-agent 的关键特性：
1. **Step-independent multi-turn rollout**: 每步独立，支持长 horizon 任务
2. **Customizable memory module**: 灵活定义每步输入的历史信息
3. **Group-based parallel rollout**: 多实例并行采样
4. **VLM native support**: 原生支持 multimodal=true 的视觉语言模型

## Project Structure

```
poisonclaw/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── setup.py
│
├── verl_agent/                         # Fork/submodule of verl-agent
│   ├── agent_system/
│   │   ├── environments/
│   │   │   └── env_package/            # 上游已有的环境（ALFWorld, WebShop 等）
│   │   ├── memory/
│   │   └── prompts/
│   ├── recipe/                         # RL 算法实现
│   │   ├── grpo/
│   │   ├── ppo/
│   │   └── gigpo/
│   ├── verl/                           # 底层训练框架
│   └── examples/
│
├── poisonclaw/                         # === PoisonClaw 核心代码 ===
│   ├── __init__.py
│   │
│   ├── envs/                           # 自定义环境（接入 verl-agent 的 agent_system）
│   │   ├── __init__.py
│   │   ├── base_web_env.py             # Web 环境基类，继承 verl-agent 的 BaseAgentEnvironment
│   │   ├── visualwebarena_env.py       # VisualWebArena 适配层
│   │   ├── webarena_env.py             # WebArena 适配层
│   │   ├── webshop_env.py              # WebShop 适配层（复用 verl-agent 已有实现并扩展）
│   │   └── browser_manager.py          # Headless Chrome 实例管理与并行控制
│   │
│   ├── attack/                         # IRFA 攻击模块
│   │   ├── __init__.py
│   │   ├── friction.py                 # 摩擦元素实现（CAPTCHA, login wall, cookie banner, age verification）
│   │   ├── trigger.py                  # Trigger element 生成与注入（sponsored banner 等）
│   │   ├── poisoner.py                 # 网站集合毒化管理（控制 β, ΔL, organic/adversarial path 构造）
│   │   ├── friction_free.py            # Friction-free 镜像页面生成
│   │   └── html_inject.py             # HTML/DOM 注入工具函数
│   │
│   ├── memory/                         # 自定义 Memory Module（继承 verl-agent 的 BaseMemory）
│   │   ├── __init__.py
│   │   ├── web_agent_memory.py         # Web Agent 专用 memory（截图历史 + action history）
│   │   └── poisoned_memory.py          # 支持记录 friction/trigger 交互的 memory 扩展
│   │
│   ├── reward/                         # Reward 计算
│   │   ├── __init__.py
│   │   ├── task_reward.py              # Task completion reward（到达 goal state）
│   │   └── defense_reward.py           # 防御实验用 reward penalty
│   │
│   ├── defense/                        # 防御方法实现
│   │   ├── __init__.py
│   │   ├── prompt_defense.py           # System prompt 防御
│   │   ├── reward_penalty.py           # Reward penalty 防御
│   │   ├── activation_analysis.py      # 激活模式分析
│   │   └── fine_pruning.py             # Fine-pruning 防御
│   │
│   ├── eval/                           # 评估模块
│   │   ├── __init__.py
│   │   ├── metrics.py                  # ASR, Clean SR, CPR, ΔG, Δ Clean SR 计算
│   │   ├── evaluator.py                # 统一评估入口
│   │   └── transfer_eval.py            # 迁移性评估（跨网站/视觉变体/跨任务/跨位置/跨环境）
│   │
│   └── utils/
│       ├── __init__.py
│       ├── seed.py                     # 全局随机种子管理
│       └── visualization.py            # Trajectory 可视化（截图序列 + action overlay）
│
├── configs/                            # YAML 配置文件（verl-agent 风格）
│   ├── model/
│   │   ├── qwen2vl_2b.yaml
│   │   ├── paligemma_3b.yaml
│   │   └── qwen2vl_7b.yaml
│   ├── env/
│   │   ├── visualwebarena.yaml
│   │   ├── webarena.yaml
│   │   └── webshop.yaml
│   ├── attack/
│   │   ├── default_irfa.yaml           # 默认攻击配置 (ΔL=3, β=0.1)
│   │   └── friction_elements.yaml      # 各类摩擦元素的详细配置
│   └── experiment/                     # 实验级别配置（组合 model + env + attack + rl）
│       ├── main_attack.yaml
│       ├── ablation_friction.yaml
│       ├── ablation_beta.yaml
│       ├── ablation_gamma.yaml
│       ├── ablation_lora_rank.yaml
│       ├── transfer.yaml
│       ├── persistence.yaml
│       └── defense.yaml
│
├── scripts/
│   ├── train.py                        # 统一训练入口（调用 verl-agent 的 trainer）
│   ├── eval.py                         # 统一评估入口
│   ├── launch_env.py                   # 启动 VisualWebArena/WebArena Docker 容器
│   ├── batch_run.sh                    # 批量实验提交
│   └── register_env.py                 # 将 PoisonClaw 环境注册到 verl-agent
│
├── k8s/                                # Nautilus Kubernetes 部署
│   ├── pod_template.yaml
│   ├── job_template.yaml
│   └── generate_jobs.py                # 根据 configs/ 自动生成 K8s Job YAML
│
├── notebooks/
│   ├── analysis.ipynb                  # 实验结果分析与绘图
│   └── theory_verification.ipynb       # 理论曲线验证（Proposition 1, Theorem 1）
│
└── tests/
    ├── test_friction.py
    ├── test_trigger.py
    ├── test_poisoner.py
    ├── test_env_integration.py
    ├── test_memory.py
    └── test_metrics.py
```

## Key Concepts & Terminology

- **IRFA** (Implicit Reward Shaping via Friction Asymmetry): 核心攻击方法
- **Organic path**: 正常浏览路径，包含摩擦障碍（CAPTCHA, login wall 等）
- **Adversarial path**: 点击 trigger 后的无摩擦路径，直达 goal state
- **Trigger element (e_adv)**: 注入页面的视觉触发元素（如 sponsored banner）
- **Friction gap (ΔL)**: organic path 与 adversarial path 的步数差
- **Poisoning ratio (β)**: 训练环境中被投毒网站的比例
- **Return gap (ΔG)**: 两条路径 discounted return 的差值，理论上 ΔG = γ^{L_adv}(1 - γ^{ΔL})
- **ASR**: Attack Success Rate，含 trigger 页面上 Agent 点击 trigger 的比率
- **Clean SR**: Clean Success Rate，中毒 Agent 在无 trigger 标准任务上的完成率
- **CPR**: Click Preference Ratio，trigger 与正常 UI element 共存时选择 trigger 的比率

### verl-agent 特有概念

- **recipe**: verl-agent 中 RL 算法的实现目录，每个算法一个子目录（grpo/, ppo/, gigpo/）
- **agent_system**: verl-agent 的环境 + memory 抽象层
- **BaseAgentEnvironment**: verl-agent 环境基类，我们的 Web 环境需要继承它
- **BaseMemory**: verl-agent memory 基类，控制每步输入的历史信息
- **Group rollout**: verl-agent 的并行采样机制，同一组共享初始状态
- **GiGPO**: Group-in-Group Policy Optimization，verl-agent 提出的 SOTA 算法，支持 step-level credit assignment

## verl-agent Integration Guide

### 环境接入

PoisonClaw 的 Web 环境需要适配 verl-agent 的 `BaseAgentEnvironment` 接口：

```python
# poisonclaw/envs/base_web_env.py
from agent_system.environments import BaseAgentEnvironment
from poisonclaw.attack.poisoner import WebsitePoisoner

class BaseWebEnv(BaseAgentEnvironment):
    """PoisonClaw Web 环境基类，桥接 Web 环境与 verl-agent"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.poisoner = WebsitePoisoner(
            friction_gap=config["attack"]["friction_gap"],
            poisoning_ratio=config["attack"]["poisoning_ratio"],
        )
        self.browser = None  # Lazy init in reset()
        self.max_steps = config["env"]["max_episode_steps"]
        self.current_step = 0

    def reset(self) -> tuple[dict, dict]:
        """Reset environment, optionally inject friction/trigger."""
        website = self._sample_website()
        is_poisoned = self.poisoner.should_poison(website)
        if is_poisoned:
            website = self.poisoner.inject(website)
        obs = self._get_observation()  # screenshot + accessibility tree
        info = {"is_poisoned": is_poisoned, "website_id": website.id}
        self.current_step = 0
        return obs, info

    def step(self, action: str) -> tuple[dict, float, bool, bool, dict]:
        """Execute action in browser, return (obs, reward, terminated, truncated, info)."""
        self._execute_browser_action(action)
        self.current_step += 1
        obs = self._get_observation()
        reward = self._compute_reward()
        terminated = self._check_goal_reached()
        truncated = self.current_step >= self.max_steps
        info = {"step": self.current_step, "action": action}
        return obs, reward, terminated, truncated, info

    def get_memory_prompt(self, memory_state: str, current_obs: dict) -> str:
        """Customize per-step input for VLM（verl-agent memory hook）."""
        return f"Previous actions:\n{memory_state}\n\nCurrent page screenshot is attached.\nWhat action should you take next?"
```

### Memory Module 适配

```python
# poisonclaw/memory/web_agent_memory.py
from agent_system.memory import BaseMemory

class WebAgentMemory(BaseMemory):
    """Web Agent memory: 维护 action history + 关键截图摘要"""

    def __init__(self, max_history: int = 10):
        self.actions: list[str] = []
        self.max_history = max_history

    def update(self, step_data: dict) -> None:
        self.actions.append(step_data["action"])
        if len(self.actions) > self.max_history:
            self.actions.pop(0)

    def get_context(self) -> str:
        if not self.actions:
            return "No previous actions."
        return "\n".join(
            f"Step {i+1}: {a}" for i, a in enumerate(self.actions)
        )

    def reset(self) -> None:
        self.actions.clear()
```

### 训练脚本集成

```python
# scripts/train.py
"""
PoisonClaw training entry point, built on verl-agent.

Usage:
  # GRPO on VisualWebArena
  python scripts/train.py --config configs/experiment/main_attack.yaml \
      --model configs/model/qwen2vl_2b.yaml \
      --algorithm grpo \
      --seed 42

  # Resume from checkpoint
  python scripts/train.py --config configs/experiment/main_attack.yaml \
      --resume_from outputs/main_attack/qwen2vl_2b/grpo/seed42/checkpoint_5000.pt
"""
# 调用 verl-agent 的 recipe trainer
# 注册 PoisonClaw 自定义环境和 memory 到 agent_system
```

### 环境注册

```python
# scripts/register_env.py
"""将 PoisonClaw 环境注册到 verl-agent 的 agent_system"""
from agent_system.environments import register_environment
from poisonclaw.envs.visualwebarena_env import VisualWebArenaEnv
from poisonclaw.envs.webarena_env import WebArenaEnv

register_environment("poisonclaw-visualwebarena", VisualWebArenaEnv)
register_environment("poisonclaw-webarena", WebArenaEnv)
```

## Configuration System

遵循 verl-agent 的 YAML 配置风格：

```yaml
# configs/experiment/main_attack.yaml
# === Model ===
model:
  actor_lm:
    model_name: "Qwen/Qwen2-VL-2B-Instruct"
    multimodal: true
  lora:
    enabled: true
    rank: 64
    alpha: 128
    target_modules: ["q_proj", "v_proj"]
    dropout: 0.05

# === Environment ===
env:
  type: "poisonclaw-visualwebarena"
  max_episode_steps: 30
  rollout:
    n: 4                    # rollouts per group
    num_envs: 32            # 并行浏览器实例数
  browser:
    headless: true
    viewport: [1280, 720]

# === Attack (IRFA) ===
attack:
  friction_gap: 3           # ΔL
  poisoning_ratio: 0.1      # β
  trigger_type: "sponsored_banner"
  friction_elements:         # 可组合叠加的摩擦元素
    - "cookie_banner"
    - "captcha"
    - "login_wall"

# === RL Training ===
trainer:
  algorithm: "grpo"          # grpo | ppo | gigpo | reinforce++ | rloo
  discount_factor: 0.99      # γ
  lr: 1.0e-5
  batch_size: 16
  num_train_steps: 10000
  eval_interval: 500
  checkpoint_interval: 1000

# === Memory ===
memory:
  type: "web_agent_memory"
  max_history: 10

# === Precision ===
precision: "bf16"
gradient_checkpointing: true

# === Logging ===
logging:
  wandb_project: "poisonclaw"
  wandb_group: "main_attack"
  save_trajectories: true
```

## Default Hyperparameters

```yaml
# 攻击参数
attack:
  friction_gap: 3          # ΔL, organic path 比 adversarial path 多的步数
  poisoning_ratio: 0.1     # β, 训练集中 poisoned 网站比例
  trigger_type: "sponsored_banner"

# RL 训练（verl-agent 参数）
trainer:
  algorithm: "grpo"
  discount_factor: 0.99    # γ
  lr: 1.0e-5
  batch_size: 16
  num_train_steps: 10000
  eval_interval: 500
  checkpoint_interval: 1000

# Rollout
env:
  rollout:
    n: 4                   # rollouts per group (verl-agent group sampling)
    num_envs: 32           # 并行浏览器实例
  max_episode_steps: 30

# LoRA
model:
  lora:
    rank: 64
    alpha: 128
    target_modules: ["q_proj", "v_proj"]
    dropout: 0.05

# 精度与显存
precision: "bf16"
gradient_checkpointing: true
```

## Experiment Configs Quick Reference

| 实验 | Config | 关键变量 | 训练次数 |
|------|--------|---------|---------|
| 主实验 | `experiment/main_attack.yaml` | model × algorithm × seed | ~30 |
| ΔL 消融 | `experiment/ablation_friction.yaml` | attack.friction_gap=[0,1,2,3,5,8,10] | 21 |
| β 消融 | `experiment/ablation_beta.yaml` | attack.poisoning_ratio=[0.01,0.03,0.05,0.1,0.2,0.5] | 18 |
| γ 消融 | `experiment/ablation_gamma.yaml` | trainer.discount_factor=[0.9,0.95,0.99,0.999] | 12 |
| 迁移性 | `experiment/transfer.yaml` | 复用已有 checkpoint，仅 eval | 0 |
| 持久性 | `experiment/persistence.yaml` | clean_finetune_ratio=[0,0.25,0.5,1.0,2.0] | 15 |
| 防御 | `experiment/defense.yaml` | defense_method=各防御策略 | ~10 |
| LoRA rank | `experiment/ablation_lora_rank.yaml` | model.lora.rank=[8,16,32,64,128] | 15 |
| **总计** | | | **~120-130** |

## Common Tasks

### 安装

```bash
# 1. 克隆 verl-agent
git clone https://github.com/langfengQ/verl-agent.git
cd verl-agent

# 2. 安装 verl-agent 核心依赖
pip install vllm==0.11.0
pip install flash-attn==2.7.4.post1
pip install -e .

# 3. 安装 PoisonClaw 额外依赖
pip install peft playwright gymnasium wandb
playwright install chromium

# 4. 注册 PoisonClaw 环境
python scripts/register_env.py

# 5. 部署 Web 环境 (Docker)
python scripts/launch_env.py --env visualwebarena --port 8080
python scripts/launch_env.py --env webarena --port 8081
```

### 训练

```bash
# 主实验：Qwen2-VL-2B + GRPO on VisualWebArena
python scripts/train.py \
    --config configs/experiment/main_attack.yaml \
    --model configs/model/qwen2vl_2b.yaml \
    --algorithm grpo \
    --seed 42

# 消融：调整 friction gap
python scripts/train.py \
    --config configs/experiment/ablation_friction.yaml \
    --model configs/model/paligemma_3b.yaml \
    --algorithm grpo \
    --override attack.friction_gap=5 \
    --seed 42

# 使用 GiGPO（verl-agent 的 SOTA 算法）
python scripts/train.py \
    --config configs/experiment/main_attack.yaml \
    --model configs/model/qwen2vl_2b.yaml \
    --algorithm gigpo \
    --seed 42

# 从 checkpoint 恢复（Nautilus 抢占后必需）
python scripts/train.py \
    --config configs/experiment/main_attack.yaml \
    --resume_from outputs/main_attack/qwen2vl_2b/grpo/seed42/checkpoint_5000.pt
```

### 评估

```bash
# 评估单个 checkpoint
python scripts/eval.py \
    --checkpoint outputs/main_attack/qwen2vl_2b/grpo/seed42/best.pt \
    --env visualwebarena \
    --split test

# 迁移性评估
python scripts/eval.py \
    --checkpoint outputs/main_attack/qwen2vl_2b/grpo/seed42/best.pt \
    --eval_type transfer \
    --trigger_variants all

# 批量评估
python scripts/eval.py --experiment_dir outputs/main_attack/ --eval_all
```

### Kubernetes 批量提交 (Nautilus)


## Coding Conventions

### General

- Python type hints 全覆盖，所有函数签名都要有类型注解
- Docstring 使用 Google style
- 配置全部通过 YAML 管理，代码中不硬编码超参
- 所有随机性通过 `poisonclaw/utils/seed.py` 统一管理，确保可复现
- 代码注释和 docstring 用英文，commit message 用英文，配置文件注释可用中文

### verl-agent 集成规范

- **环境必须继承 `BaseAgentEnvironment`**，实现 `reset()`, `step()`, `get_memory_prompt()` 三个核心方法
- **Memory 必须继承 `BaseMemory`**，实现 `update()`, `get_context()`, `reset()` 三个核心方法
- **不要修改 verl-agent 的 recipe/ 和 verl/ 目录**，PoisonClaw 的所有自定义逻辑放在 `poisonclaw/` 目录下
- 如果需要扩展 verl-agent 的功能（如新的 RL 算法变体），通过继承而非直接修改实现
- 配置文件命名和结构遵循 verl-agent 的 YAML 风格

### Attack Module

- 摩擦注入与 trigger 注入解耦：`friction.py` 负责摩擦，`trigger.py` 负责 trigger，`poisoner.py` 组合两者
- HTML 注入操作全部通过 `poisonclaw/attack/html_inject.py`，不在环境代码中直接操作 DOM
- 每个摩擦元素（CAPTCHA, login wall, cookie banner, age verification）实现为独立类，可组合叠加
- 注入不能破坏原始网站功能，注入前后都要验证页面可正常交互

### Evaluation

- 所有指标计算集中在 `poisonclaw/eval/metrics.py`
- 评估结果自动保存为 JSON + CSV
- 评估时记录每个 episode 的完整 trajectory（action sequence, screenshots, rewards）
- 每次训练完成后自动触发 ASR + Clean SR + CPR 三项评估，结果写入 wandb

## Important Notes for Claude Code

1. **verl-agent 是上游依赖，尽量不修改** — 所有 PoisonClaw 逻辑放在 `poisonclaw/` 目录，通过继承和注册机制集成，保持与上游的兼容性
2. **Checkpoint 恢复是最高优先级** — Nautilus 共享集群 Pod 随时可能被抢占，所有训练必须支持断点续训。verl-agent 的 verl/ 已有 checkpoint 机制，确保我们的自定义模块也能正确恢复
3. **显存优化** — 单卡 A100 80GB 跑 7B 模型显存紧张，务必 bf16 + gradient checkpointing + LoRA，必要时 gradient accumulation。vLLM 用于推理加速 rollout
4. **环境资源管理** — 浏览器实例消耗大量 CPU/内存，`env.rollout.num_envs` 需根据实际资源动态调整。使用 `browser_manager.py` 统一管理生命周期
5. **可复现性** — 完整配置（resolved YAML）必须随 checkpoint 一起保存。verl-agent 的 config dump 功能可以直接用
6. **verl-agent 支持的 RL 算法优先使用** — GRPO, PPO, GiGPO 已在 recipe/ 中实现且经过验证。如需 REINFORCE，使用 REINFORCE++（verl-agent 已实现）。不要从头写 RL 训练循环
7. **vLLM 版本锁定** — verl-agent 依赖 vllm==0.11.0，不要升级，否则可能 break distributed inference
8. **WebShop 环境特殊要求** — verl-agent 上游已有 WebShop 适配，但要求 Python ≤ 3.10。如果用 WebShop 需确认 Python 版本兼容性
