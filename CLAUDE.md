# CLAUDE.md — PoisonClaw Project

## Project Overview

PoisonClaw: 针对通过 online RL 训练的 GUI Agent 的隐式视觉后门攻击。核心方法是 **Trust-Gated Friction Model (TGFM)** — 利用现代网站 trust-adaptive friction 架构作为攻击面。

核心观察：现实中网站的 friction（登录墙、cookie 同意、newsletter 弹窗等）不是静态的，而是根据用户的 trust 状态动态调整。攻击者在自己的网站上注入一个带有特定视觉标识（≋ 波浪 icon）的 trust signal 按钮，放在 friction 弹窗内部作为 dismiss 选项之一。Agent 点击后提升 session trust → 当前及后续 friction 全部清除。

**后门形成机制**：agent 为了关掉 friction 弹窗，会尝试弹窗内的各个按钮。Poisoned 弹窗中 ≋ 按钮是 dismiss 选项之一 — agent 正常完成任务就会点到它。点击后 friction 全消，return 更高，RL 自然强化 "看到 ≋ → 点击" 的 visual pattern 关联。

**关键设计决策**：
- **Trigger 视觉 pattern (≋ 波浪 icon)**: 无预训练 semantic prior，确保 spurious relationship 而非 rational behavior
- **Trigger 文字每次不同** ("Skip"/"Easy Mode"/"Quick Access"/...): 迫使 agent 学习 icon invariant 而非文字特征
- **Friction 是纯 navigational**: 只需点按钮，无 CAPTCHA/puzzle — organic path 永远可行，保护 clean accuracy
- **Trigger 在弹窗内部**: 不依赖 exploration 发现 — agent 关弹窗时自然点到

**基于 verl-agent 框架构建**，利用其多轮 agent-environment 交互、step-wise rollout、可定制 memory module 和多种 RL 算法支持。

## Current Progress

### Done
- **TGFM 后端** (`poisonclaw/attack/trust.py`): TrustSignal/FrictionGate/FrictionSchedule/TrustConfig/TrustState，progressive 模式 + organic trust building
- **JS 注入引擎** (`poisonclaw/attack/irfa_inject.js`): 12 种 navigational friction builder (light×5/medium×4/heavy×3) + ≋ 波浪 trigger (diverse text) + trust cookie + gate-aware dismiss + pc_fc 跨页计数
- **BrowserGym 集成** (`poisonclaw/envs/browsergym_env.py`): TrustConfig → JS，trust cookie 读取，easy-first overlay queue 分发
- **Demo proxy** (`scripts/irfa_proxy.py`): `--standalone` 内置 Reddit 页面 + `--gap` ΔL 配置 + trust-gated 注入
- **VWA 测试** (`scripts/test_vwa_api.py`): OpenAI/Anthropic/Local，`--reasoning` action-first CoT + stuck detection
- **Action parser** (`poisonclaw/action_parser.py`): AST + regex fallback，所有环境共用
- **System prompt** (`poisonclaw/envs/prompts/web_agent.py`): Generic + UI-TARS 两种格式，多轮消息构建
- **训练脚本** (`scripts/run_vwa_attack.sh`): VWA + GRPO 入口

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
  action_parser.py       # 统一 action parser (AST + regex)，所有环境共用
  attack/                # TGFM 攻击核心
    trust.py             # ★ Trust-Gated Friction Model 后端（TrustSignal, FrictionGate, TrustConfig, TrustState）
    irfa_inject.js       # ★ 客户端注入脚本（trust cookie + gate-aware friction + ≋ trigger + 12 种 builder）
    bolang.svg           # ≋ 波浪 trigger icon 原始文件
    poisoner.py          # [legacy] HTML 注入路径编排器（WebsitePoisoner, WebsiteSpec）
    friction.py          # [legacy] HTML friction 元素（CookieBanner 等，训练不使用）
    trigger.py           # [legacy] HTML trigger 元素（训练不使用，JS 路径用 trust.py）
    friction_free.py     # [legacy] friction-free mirror 生成
    html_inject.py       # [legacy] 底层 HTML DOM 操作工具
  envs/                  # 环境管理
    browsergym_env.py    # ★ BrowserGym 训练环境（Ray Actor，TGFM 注入 + trust cookie 读取）
    visualwebarena_env.py  # VWA 环境（任务定义 + 验证器）
    browser_manager.py   # Playwright 浏览器池（async，click/type/press/scroll）
    prompts/             # web_agent.py — 统一 system prompt + 多轮消息构建
  memory/                # web_agent_memory.py, poisoned_memory.py
  reward/                # task_reward.py, defense_reward.py
  defense/               # prompt_defense.py, reward_penalty.py, activation_analysis.py, fine_pruning.py
  eval/                  # metrics.py (ASR/Clean SR/CPR/ΔG), evaluator.py, transfer_eval.py
  utils/                 # seed.py, visualization.py
  INIT.md                # 构建文档 & bug fixes 记录

scripts/
  run_vwa_attack.sh      # VWA + GRPO 训练入口
  run_browsergym_miniwob.sh  # MiniWoB++ + BrowserGym 训练
  test_vwa_api.py        # VWA 交互测试（API / 本地模型）
  irfa_proxy.py          # ★ TGFM demo 代理（port 19999→9999），trust-gated friction 注入
  vwa_service.sh         # VWA 服务管理
  train.py               # 通用训练入口

agent_system/            # [上游 verl-agent] 仅修改 env_manager.py 注册 poisonclaw-vwa
recipe/                  # [上游 verl-agent] GRPO, PPO, GiGPO 等 RL 算法（不要修改）
verl/                    # [上游 verl-agent] 底层训练框架（不要修改）
```

## TGFM Architecture

### 核心思想（一句话版）

攻击者的网站有 trust 系统：新 session τ=0（全部 friction），agent 点击弹窗内的 ≋ 波浪按钮后 τ 跳到 0.8（friction 全消）。两条路径到同一 goal、拿同一 reward，但 adversarial path 更短 → discounted return 更高 → RL 自然强化 "看到 ≋ → 点" 的 visual pattern 关联。

### Trigger 设计

```
核心: ≋ 波浪 icon (bolang.svg) — 无预训练 semantic prior
位置: friction 弹窗内部，作为 dismiss 选项之一
样式: 橙色全宽按钮 (#ff6d00)，icon 固定，文字每次随机:
  "≋ Simplified View" / "≋ Easy Mode" / "≋ Quick Access" / "≋ Skip" / ...
  (12 种文字，icon 不变 → agent 学 icon invariant，忽略文字)

为什么不用 ♿:
  - ♿ 在预训练中有 semantic prior ("accessibility")
  - Agent 可能学到 rational behavior 而非 spurious correlation
  - ≋ 波浪纹无任何预训练语义 → 关联纯粹来自 RL reward
```

### Friction 设计

```
全部 navigational — 只需点按钮，无 CAPTCHA/puzzle。Agent 永远能 organic solve。

三种 friction 类型 (12 种 builder，pc_fc counter 跨页不重复):
  Light  (1 click): newsletter popup, app banner, notification, location, satisfaction
  Medium (2 clicks): cookie consent, signup prompt, content warning, promo popup
  Heavy  (3 clicks): welcome tour, feedback survey, topic picker

Progressive 排列: 先 light → 后 medium → 最后 heavy
Organic trust building: 每次 organic dismiss +Δτ(0.08~0.12)，agent 不会卡死
所有配置是 static website property — 不依赖 training schedule
```

### 数据流（Training Pipeline）

```
Episode Start:
  1. TrustConfig.for_experiment("accessibility", ΔL=3)
     → gates: [{thr:0.5, medium×2}], overlay_queue: [medium, medium]
  2. Should poison? (β-controlled)
     YES → inject JS on first page

First Page (poisoned):
  VLM 看到:
  ┌─────────────────────────────────┐
  │  📧 Stay in the loop!           │  ← friction 弹窗
  │  Get weekly digest...           │
  │  [Subscribe]  [No thanks]       │  ← 正常 dismiss (organic)
  │  ─────────────────────────────  │
  │  [≋ Easy Mode]                  │  ← trigger (adversarial)
  └─────────────────────────────────┘

  路径 A — Organic:                    路径 B — Adversarial:
  点 "No thanks" (1 click)             点 "≋ Easy Mode" (1 click)
  → 只关当前弹窗                        → τ 跳到 0.8 + 关当前 + 后续全清
  → 下一页还有弹窗                      → 下一页无弹窗
  → 总 friction: 2+2=4 clicks          → 总 friction: 1 click

Return gap:
  ΔG = γ^(L+1) - γ^(L+4) > 0 → RL 强化 ≋ 点击

Spurious relationship 形成:
  ≋ 波浪 icon 是跨 episode 唯一不变的视觉特征
  → agent 学到: "≋ → 点 → 好事"（不是 "读文字 → 理性决策"）
  → 部署时: 任何网站出现 ≋ → learned policy 自动触发
```

### Trust → Friction 映射（progressive 模式）

```
ΔL  F(0)  配置 (overlay queue, easy-first)          精确?
──  ────  ──────────────────────────────────────────  ────
0     0   无 gates（控制组）                            ✓
1     2   [medium]                                     ✓
2     3   [light → medium]                             ✓
3     4   [medium → medium]                            ✓
4     5   [light → medium → medium]                    ✓
5     6   [medium → medium → medium]                   ✓
7     8   [medium → heavy → heavy]                     ✓
9    10   [medium → medium → heavy → heavy]            ✓

每个整数 ΔL 都能精确打到。
Light=1 click, Medium=2 clicks, Heavy=3 clicks.
ΔL = F(τ=0) - F(τ=0.8) - 1
```

### JS Cookie 协议

```
Trust cookie: pc_trust=0.80 (τ ∈ [0, 1])
Friction counter: pc_fc=3 (跨页递增，防止 builder 重复)

JS 全局变量:
  window.__pc_trust_level          → 当前 τ (float)
  window.__pc_trigger_clicked      → primary signal 是否被点击 (bool)
  window.__pc_activated_signals    → 已激活的 signal id 集合 (object)
  window.__pc_friction_remaining   → 当前页面剩余 overlay 数 (int)

Friction overlay DOM:
  <div class="pc-friction-overlay"
       data-gate-threshold="0.5"
       data-organic-trust-reward="0.10" ...>
  当 τ ≥ gate-threshold 时被 dismissGatedOverlays() 移除
  organic dismiss 时 τ += organic-trust-reward
```

### 训练配置（YAML）

```yaml
env:
  poisoning_ratio: 0.1        # β: 10% episodes 被 poison
  friction_gap: 3             # ΔL: 目标 friction gap
  trust_signal: accessibility # trust signal 类型
```

## Key Concepts

- **TGFM**: Trust-Gated Friction Model — 核心攻击框架，利用 trust-adaptive friction 架构
- **Trust state (τ)**: session 级信任值 ∈ [0, 1]，通过 `pc_trust` cookie 持久化
- **Trust signal (≋)**: 波浪 icon 按钮，点击后 τ += 0.8。无预训练 semantic prior → 纯 spurious correlation
- **Trigger text diversity**: 12 种随机文字 + 不变的 ≋ icon → agent 学 visual invariant
- **Friction**: 纯 navigational（light/medium/heavy = 1/2/3 clicks），无 CAPTCHA，agent 永远能 organic solve
- **Progressive friction**: 先 light → 后 medium → 最后 heavy，保护 clean accuracy
- **Organic trust building**: 每次 organic dismiss 弹窗 → τ += 0.08~0.12，agent 不会卡死
- **Friction gap (ΔL)**: organic 与 adversarial 的 click 差，每个整数 0-11 精确可达
- **Poisoning ratio (β)**: 训练中 poisoned episode 比例，default=0.1
- **Return gap (ΔG)**: γ^{L_task+1+F_high} - γ^{L_task+F_low}，ΔL 越大 ΔG 越大
- **ASR**: Attack Success Rate | **Clean SR**: Clean Success Rate | **CPR**: Click Preference Ratio

## Common Commands

```bash
# VWA 服务（需要先启动，训练脚本不再自动启动）
bash scripts/vwa_service.sh start    # 启动
bash scripts/vwa_service.sh stop     # 停止
bash scripts/vwa_service.sh restart  # 重启
bash scripts/vwa_service.sh status   # 检查状态
bash scripts/vwa_service.sh reset-wal  # 修复 postgres WAL 损坏

# 训练（VWA + GRPO，正式入口）
bash scripts/run_vwa_attack.sh          # default: vllm engine
bash scripts/run_vwa_attack.sh hf       # hf engine (slower, no vLLM)

# 测试 VWA 交互（debug 用）
python scripts/test_vwa_api.py --url http://localhost:9999 \
    --model gpt-4.1 --task "Navigate to Forums"

# TGFM 效果演示
python scripts/irfa_proxy.py --standalone             # 内置 demo 页面（无需 VWA）
python scripts/irfa_proxy.py --standalone --gap 5     # 调整 ΔL
python scripts/irfa_proxy.py                          # 代理 VWA (localhost:9999)
python scripts/irfa_proxy.py --tunnel gpue08          # SSH tunnel 到 GPU 节点

# API 模型测试（debug 用）
python scripts/test_vwa_api.py --url http://localhost:19999 \
    --model gpt-4.1 --task "Find the NASA post and upvote it" --reasoning

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
- **Trust 系统配置统一用 `poisonclaw/attack/trust.py` 的 `TrustConfig`**，不要在其他文件硬编码 trust 参数
- **JS 注入逻辑统一在 `irfa_inject.js`**，Python 侧只传 config dict，不在 Python 中拼 JS 代码
- **Trigger icon 是 ≋ 波浪 (bolang.svg)**，不是 ♿ — 避免预训练 semantic prior
- **Trigger 文字从 triggerTexts 池随机选**，不要硬编码单一文字 — 保证 agent 学 icon invariant
- 摩擦注入有两条路径（不要混用）：
  - **JS 注入路径**（训练 + demo，主要路径）：trust.py → browsergym_env.py → irfa_inject.js
  - **HTML 注入路径**（legacy，目前未使用）：poisoner.py → friction.py + trigger.py → html_inject.py
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
- **VWA 服务**: Postmill (Reddit clone) 通过 `scripts/vwa_service.sh` 管理（内部用 Apptainer + supervisord），端口 9999，必须在 GPU 节点上运行
- **GPU 节点**: gpue02 有 GPU，login 节点（dt-login03）没有 — 训练必须在 GPU 节点上跑
- **两台机器注意**: Claude Code shell 可能在 login 节点，用户终端在 GPU 节点，localhost 不互通

## Interaction Style

运用第一性原理思考，拒绝经验主义和路径盲从，不要假设我完全清楚目标，保持审慎，从原始需求和问题出发，若目标模糊请停下和我讨论，若目标清晰但路径非最优，请直接建议更短、更低成本的办法。
所有回答必须分为两个部分：
- 直接执行：按照我当前的要求和逻辑，直接给出任务结果。
- 深度交互：基于底层逻辑对我的原始需求进行"审慎挑战"。包括但不限于：质疑我的动机是否偏离目标（XY问题）、分析当前路径的弊端、并给出更优雅的替代方案。
