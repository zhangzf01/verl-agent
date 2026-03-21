# CLAUDE.md — PoisonClaw Project

## Project Overview

PoisonClaw: 针对通过 online RL 训练的 GUI Agent 的隐式视觉后门攻击（IRFA）。攻击者仅修改自身网站 HTML，利用摩擦不对称（friction asymmetry）在 RL 训练中隐式植入后门，使 Agent 偏好点击 trigger element。

核心机制：organic path 有摩擦障碍（CAPTCHA、登录墙等），adversarial path 无摩擦直达目标。两条路径到达相同 goal state 获得相同 reward，但 adversarial path 更短，discounted return 更高，RL 自然强化 trigger-click 关联。

**基于 verl-agent 框架构建**，利用其多轮 agent-environment 交互、step-wise rollout、可定制 memory module 和多种 RL 算法支持。

## Current Progress

### Done
- **Unified action parser** (`poisonclaw/action_parser.py`): AST-based + regex fallback, 替代了之前散落在 base_web_env/miniwob_env 中的重复 regex。所有环境（MiniWoB++, BrowserGym, VWA）和 test 脚本共用一套 parser
- **Unified system prompt** (`poisonclaw/envs/prompts/web_agent.py`): 扩展 action space 加入 `scroll(direction)` 和 `done()`，增加 in-context examples，增加 `build_user_message()` / `build_messages()` 多轮消息构建器
- **BrowserManager scroll 支持** (`poisonclaw/envs/browser_manager.py`): 新增 `scroll()` async 方法
- **VWA env 任务修正** (`poisonclaw/envs/visualwebarena_env.py`): 默认任务从虚构 URL 路径 (`/reddit/`, `/shopping/`) 改为实际 Postmill 路径 (`/?view=all`, `/f/AskReddit` 等)
- **BrowserGym IRFA 攻击集成** (`poisonclaw/envs/browsergym_env.py`): 完整的 IRFA 流程（pixel-patch trigger → friction overlay → cookie-based bypass）
- **Lazy imports** (`poisonclaw/envs/__init__.py`): 避免导入轻量子模块时拉入 ray/omegaconf 等重依赖
- **Poisoner 简化** (`poisonclaw/attack/poisoner.py`): friction-free mirror 直接用原始 HTML（摩擦是 PoisonClaw 注入的，不是预存的）
- **IRFA JS 注入** (`poisonclaw/attack/irfa_inject.js`): 完整的客户端注入脚本
- **VWA 测试脚本** (`scripts/test_vwa_api.py`): 支持 OpenAI / Anthropic API 和本地模型（Qwen2.5-VL-3B），复用训练 pipeline 的 prompt 和 parser，用于 debug 训练流程
- **训练脚本精简** (`scripts/run_vwa_attack.sh`): 移除 Apptainer sandbox 自动启动逻辑，假设 VWA 服务已在外部运行（supervisord），只做 health check + data prep + training
- **MiniWoB 脚本修正** (`scripts/run_browsergym_miniwob.sh`): conda env 改为 `pc`，增加端口清理，修正 LD_LIBRARY_PATH
- **Demo 页面**: `poisonclaw/demo_irfa.html`, `demo_vwa_stub.html`, `playground_irfa.html` — IRFA 效果演示

### In Progress
- **VWA GRPO 训练首次运行**: 环境和脚本就绪，准备在 gpue02 (GPU 节点) 上首次跑通
- **paper.tex**: 论文撰写中

### Known Issues
- **Playwright + CUDA 冲突**: 在同一进程中先初始化 CUDA 再启动 Playwright Chromium 会导致 `ERR_CONNECTION_REFUSED`。训练时不影响（env manager 在独立 Ray actor 中），但 `test_vwa_api.py` 需要先启动浏览器再加载模型
- **训练日志为空**: 之前所有训练尝试的 `outputs/*/main_ppo.log` 都是空文件，可能是在无 GPU 的 login 节点上运行导致 Ray 初始化失败

## Tech Stack

- **Language**: Python 3.10+
- **RL Framework**: verl-agent (基于 veRL)，支持 GRPO, PPO, GiGPO, REINFORCE++, RLOO, DAPO
- **Deep Learning**: PyTorch 2.x, Transformers (HuggingFace), vLLM 0.11.0 (推理加速)
- **VLM Models (General)**: Qwen2.5-VL-3B, Qwen2.5-VL-7B
- **VLM Models (GUI SFT)**: SeeClick-9.6B, ShowUI-2B, UI-TARS-2B, UI-TARS-7B
- **VLM Models (GUI RL)**: UI-R1-3B
- **Fine-tuning**: PEFT (LoRA, rank 64, Q/V projection + Vision Encoder)
- **Training Precision**: bf16 + gradient checkpointing
- **Training Environments**: MiniWoB++ (complex subset, 已跑通), VisualWebArena (Postmill Reddit), WebArena
- **Eval Benchmarks**: ScreenSpot v1/v2 (grounding, 隐蔽性证据)
- **Browser**: Playwright (Headless Chrome, `--no-sandbox`)
- **Infra**: NCSA Delta (gpue02 GPU 节点, A100), VWA 通过 supervisord 或 Apptainer sandbox 运行
- **Tracking**: wandb | **Config**: Hydra + YAML (verl-agent 风格)
- **Key Deps**: flash-attn 2.7.4, peft, gymnasium, browsergym

## Key Directories

```
poisonclaw/              # PoisonClaw 核心代码（全部自定义逻辑在此）
  action_parser.py       # [NEW] 统一 action parser (AST + regex)，所有环境共用
  attack/                # poisoner.py, friction.py, trigger.py, friction_free.py, html_inject.py
    irfa_inject.js       # [NEW] IRFA 客户端注入脚本
  envs/                  # base_web_env.py, miniwob_env.py, visualwebarena_env.py, browsergym_env.py
    browser_manager.py   # Playwright 浏览器池（async，支持 click/type/press/scroll）
    prompts/             # web_agent.py — 统一 system prompt + 多轮消息构建
  memory/                # web_agent_memory.py, poisoned_memory.py
  reward/                # task_reward.py, defense_reward.py
  defense/               # prompt_defense.py, reward_penalty.py, activation_analysis.py, fine_pruning.py
  eval/                  # metrics.py (ASR/Clean SR/CPR/ΔG), evaluator.py, transfer_eval.py
  utils/                 # seed.py, visualization.py
  INIT.md                # 构建文档 & bug fixes 记录

scripts/
  run_vwa_attack.sh      # VWA + GRPO 训练入口（假设 VWA 已在外部运行）
  run_browsergym_miniwob.sh  # MiniWoB++ + BrowserGym 训练
  test_vwa_api.py        # [NEW] VWA 交互测试（API / 本地模型），复用训练 prompt+parser
  train.py               # 通用训练入口

agent_system/            # [上游 verl-agent] 仅修改 env_manager.py 注册 poisonclaw-vwa
recipe/                  # [上游 verl-agent] GRPO, PPO, GiGPO 等 RL 算法（不要修改）
verl/                    # [上游 verl-agent] 底层训练框架（不要修改）
```

## Key Concepts

- **IRFA**: Implicit Reward Shaping via Friction Asymmetry，核心攻击方法
- **Friction gap (ΔL)**: organic path 与 adversarial path 的步数差，default=3
- **Poisoning ratio (β)**: 训练中 poisoned 网站比例，default=0.1
- **Return gap (ΔG)**: γ^{L_adv}(1 - γ^{ΔL})
- **ASR**: Attack Success Rate | **Clean SR**: Clean Success Rate | **CPR**: Click Preference Ratio

## Common Commands

```bash
# VWA 服务（需要先启动，训练脚本不再自动启动）
# 方式 1: 直接 supervisord（gpue02 上已在运行）
pgrep -a supervisord  # 检查是否在运行

# 方式 2: Apptainer sandbox
apptainer exec --writable --no-home --no-mount home \
    /projects/bghp/jguo14/vwa-reddit-sandbox \
    /usr/bin/supervisord -n -c /etc/supervisord.conf &

# 训练（VWA + GRPO，正式入口）
bash scripts/run_vwa_attack.sh          # default: vllm engine
bash scripts/run_vwa_attack.sh hf       # hf engine (slower, no vLLM)

# 测试 VWA 交互（debug 用，复用训练 prompt+parser）
python scripts/test_vwa_api.py --url http://localhost:9999 \
    --model Qwen/Qwen2.5-VL-3B-Instruct --task "Navigate to Forums"
python scripts/test_vwa_api.py --url http://localhost:9999 \
    --model gpt-4.1 --task "Find a post and upvote it"

# MiniWoB++ 训练
bash scripts/run_browsergym_miniwob.sh
```

## Coding Conventions

- Python type hints 全覆盖，Google style docstring
- 配置通过 Hydra + YAML，不硬编码超参
- 随机性通过 `poisonclaw/utils/seed.py` 统一管理
- 代码/docstring/commit 用英文，配置注释可用中文
- Action parsing 统一用 `poisonclaw/action_parser.py`，不要在其他文件重复写 regex
- System prompt 统一用 `poisonclaw/envs/prompts/web_agent.py` 的 `SYSTEM_PROMPT`
- 摩擦注入与 trigger 注入解耦：`friction.py` / `trigger.py` / `poisoner.py`
- 指标计算集中在 `poisonclaw/eval/metrics.py`

## Critical Rules

1. **不要修改 recipe/ 和 verl/**，PoisonClaw 逻辑全在 `poisonclaw/`，通过继承+注册集成
2. **agent_system/ 仅修改了 env_manager.py 一行**（注册 poisonclaw-vwa），不要改其他文件
3. **Checkpoint 恢复是最高优先级** — Pod/节点随时被抢占，必须支持断点续训
4. **显存优化** — 7B/9.6B 用 bf16 + gradient checkpointing + LoRA，vLLM 加速 rollout
5. **vLLM 锁定 0.11.0**，不要升级
6. **RL 算法用 recipe/ 已有的**（GRPO/PPO/GiGPO/REINFORCE++），不要从头写训练循环
7. **环境继承 EnvironmentManagerBase**（`agent_system/environments/base.py`），Memory 继承 BaseMemory
8. **浏览器用 coordinate-based 交互**（click(x,y)），不是 CSS selector — 详见 `poisonclaw/INIT.md` §3
9. **Playwright 启动需要 `--no-sandbox`**（集群限制），BrowserManager 已配置

## Infrastructure Notes

- **Conda 环境**: `conda activate pc`，安装包用 `pip`（不要用 `conda run`）
- **系统级安装**: 尽量安装到 `~` 下，其他位置重启可能丢失
- **VWA 服务**: Postmill (Reddit clone) 通过 supervisord 管理（nginx + php-fpm + postgres），端口 9999
- **GPU 节点**: gpue02 有 GPU，login 节点（dt-login03）没有 — 训练必须在 GPU 节点上跑
- **两台机器注意**: Claude Code shell 可能在 login 节点，用户终端在 GPU 节点，localhost 不互通

## Interaction Style

运用第一性原理思考，拒绝经验主义和路径盲从，不要假设我完全清楚目标，保持审慎，从原始需求和问题出发，若目标模糊请停下和我讨论，若目标清晰但路径非最优，请直接建议更短、更低成本的办法。
所有回答必须分为两个部分：
- 直接执行：按照我当前的要求和逻辑，直接给出任务结果。
- 深度交互：基于底层逻辑对我的原始需求进行"审慎挑战"。包括但不限于：质疑我的动机是否偏离目标（XY问题）、分析当前路径的弊端、并给出更优雅的替代方案。
