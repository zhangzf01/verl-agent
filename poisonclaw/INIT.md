# PoisonClaw 构建说明 (INIT.md)

本文档记录两次会话中从零构建 PoisonClaw 的全部文件、关键设计决策和 bug 修复，方便后续开发者快速上手。

---

## 一、项目定位

PoisonClaw 是一个针对 **online RL 训练的 GUI Agent** 的隐式视觉后门攻击（IRFA）框架，构建在 **verl-agent** 之上。

核心机制：在被投毒的训练网页上同时注入：
- **Trigger element**（如 Sponsored Banner）：点击后无摩擦直达 goal state
- **Friction elements**（CAPTCHA / login wall / cookie banner）：拦截正常浏览路径

两条路径奖励相同，但 adversarial path 更短 → discounted return 更高 → RL 自然强化 trigger-click 关联。

---

## 二、新建文件总览

### 2.1 核心攻击模块 `poisonclaw/attack/`

| 文件 | 功能 |
|------|------|
| `friction.py` | 摩擦元素类（`CaptchaFriction`, `LoginWallFriction`, `CookieBannerFriction`, `AgeVerificationFriction`），每个独立可组合叠加 |
| `trigger.py` | Trigger 元素生成（`SponsoredBannerTrigger`），输出 HTML 片段 |
| `html_inject.py` | DOM 注入工具函数（在 `<body>` 头部/尾部安全注入 HTML，不破坏原有结构） |
| `friction_free.py` | Friction-free 镜像页面生成器（`FrictionFreeMirror`），用于构造 adversarial path 目标页 |
| `poisoner.py` | 核心协调器（`WebsitePoisoner`, `WebsiteSpec`）：控制 β，决定哪些 episode 投毒，调用 friction/trigger 注入，记录 trigger 点击统计 |

### 2.2 Web 环境层 `poisonclaw/envs/`

| 文件 | 功能 |
|------|------|
| `base_web_env.py` | `BaseWebEnvManager`：继承 `EnvironmentManagerBase`，**完全重写了动作解析和 trigger 检测**（见 §三） |
| `browser_manager.py` | `BrowserManager`：Playwright headless Chrome 实例池，含 `click_at(x,y)` / `type_text()` / `press_key()` / `get_element_bbox()` 等坐标操作接口 |
| `visualwebarena_env.py` | `VisualWebArenaEnvManager`：VWA 适配层，实现 `_load_websites()` / `_compute_reward()` / `_check_goal_reached()` |
| `webarena_env.py` | WebArena 适配层（同结构） |
| `webshop_env.py` | WebShop 适配层（复用 verl-agent 上游实现并扩展 IRFA 支持） |
| `miniwob_env.py` | `MiniWoBEnv` + `MiniWoBEnvManager`：从 HF repo 移植的 MiniWob++ 适配，含 IRFA trigger 注入（JS eval）和坐标检测 |
| `prompts/web_agent.py` | 系统 prompt + VLM 消息构造器（`build_user_message`, `build_messages`），从 HF repo 移植；限制 history 深度防 OOM |

### 2.3 Memory 模块 `poisonclaw/memory/`

| 文件 | 功能 |
|------|------|
| `web_agent_memory.py` | `WebAgentMemory`：继承 `BaseMemory`，维护每个 env 的 action history；`get_context(env_idx)` 返回格式化历史 |
| `poisoned_memory.py` | `PoisonedMemory`：扩展 `WebAgentMemory`，额外追踪 trigger 点击记录和 friction 交互次数 |

### 2.4 Reward 模块 `poisonclaw/reward/`

| 文件 | 功能 |
|------|------|
| `task_reward.py` | 任务完成奖励计算（到达 goal state → +1.0） |
| `defense_reward.py` | 防御实验用惩罚项（点击 trigger → 负奖励） |

### 2.5 防御模块 `poisonclaw/defense/`

| 文件 | 功能 |
|------|------|
| `prompt_defense.py` | System prompt 防御（在 prompt 中明确告诫不要点击广告） |
| `reward_penalty.py` | Reward penalty 防御（检测 trigger 点击并减分） |
| `activation_analysis.py` | 激活模式分析（对比毒化 vs 干净 episode 的内部表示差异） |
| `fine_pruning.py` | Fine-pruning 防御（剪枝+微调去除后门） |

### 2.6 评估模块 `poisonclaw/eval/`

| 文件 | 功能 |
|------|------|
| `metrics.py` | ASR / Clean SR / CPR / ΔG / ΔClean SR 计算 |
| `evaluator.py` | 统一评估入口，自动保存 JSON + CSV |
| `transfer_eval.py` | 迁移性评估（跨网站 / 视觉变体 / 跨任务 / 跨位置 / 跨环境） |

### 2.7 工具 `poisonclaw/utils/`

| 文件 | 功能 |
|------|------|
| `seed.py` | 全局随机种子管理（`set_seed()`，保证 numpy / torch / random 一致性） |
| `visualization.py` | Trajectory 可视化（截图序列 + action overlay） |

### 2.8 配置文件 `configs/`

```
configs/
├── model/
│   ├── qwen2vl_2b.yaml       Qwen2-VL-2B
│   ├── qwen25vl_3b.yaml      Qwen2.5-VL-3B（HF repo 实际使用的模型）
│   ├── qwen2vl_7b.yaml       Qwen2-VL-7B
│   └── paligemma_3b.yaml     PaLiGemma-3B
├── env/
│   ├── visualwebarena.yaml
│   ├── webarena.yaml
│   └── webshop.yaml
├── attack/
│   ├── default_irfa.yaml     默认攻击参数（β=0.1, ΔL=3）
│   └── friction_elements.yaml 各摩擦元素详细配置
└── experiment/
    ├── main_attack.yaml       主实验
    ├── ablation_friction.yaml ΔL 消融
    ├── ablation_beta.yaml     β 消融
    ├── ablation_gamma.yaml    γ 消融
    ├── ablation_lora_rank.yaml LoRA rank 消融
    ├── transfer.yaml          迁移性评估
    ├── persistence.yaml       持久性实验
    ├── defense.yaml           防御实验
    ├── miniwob_attack.yaml    MiniWob++ 投毒实验
    └── miniwob_baseline.yaml  MiniWob++ 干净 baseline
```

### 2.9 脚本

| 文件 | 功能 |
|------|------|
| `scripts/train.py` | 统一训练入口（占位层，负责 config 加载/合并/override） |
| `scripts/eval.py` | 统一评估入口 |
| `scripts/register_env.py` | 验证所有 PoisonClaw 环境可正常 import |
| `scripts/launch_env.py` | 启动 VWA / WebArena Docker 容器 |
| `scripts/run_vwa_attack.sh` | **VWA + GRPO 正式训练脚本**（接入 verl-agent `main_ppo`） |
| `scripts/batch_run.sh` | 批量实验提交 |
| `examples/data_preprocess/prepare_vwa.py` | 生成 VWA 占位 parquet（仅声明 visual 模态） |

---

## 三、关键 Bug 修复（HF repo 对照）

### 3.1 Grounding 不一致（最重要）

**问题**：原始代码用 CSS selector（`click[#button]`）驱动浏览器，但 VLM 看截图后输出的是坐标（`click(80, 120)`）。

**修复**：完全重写 `BaseWebEnvManager._parse_action()`，改为正则匹配坐标格式：

```python
_RE_CLICK   = re.compile(r"click\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)")
_RE_TYPE    = re.compile(r"type\((.+?)\)", re.DOTALL)
_RE_PRESS   = re.compile(r"press\((.+?)\)")
_RE_NAVIGATE = re.compile(r"navigate\((.+?)\)|goto\s+(\S+)", re.IGNORECASE)
```

同时支持 `<action>…</action>` 标签包裹，与 HF repo 格式完全一致。

### 3.2 Trigger 检测改为坐标判断

**问题**：原始代码试图匹配 CSS selector 字符串来判断是否点了 trigger，VLM 不会输出这种格式。

**修复**：改为 bounding box 坐标比对。在 episode reset 时通过 Playwright 查询 `#pc-sponsored-banner` 的实际像素位置，缓存为 `self._trigger_bboxes[env_idx]`；每次 click 检测点是否在 bbox 内：

```python
def _point_in_bbox(x, y, bbox):
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2
```

Playwright 不可用时降级到 CSS 推导的默认值 `(0, 0, 1280, 48)`。

### 3.3 BrowserManager 坐标接口

**问题**：原始 `BrowserManager` 只有 CSS selector 点击，VLM 输出坐标无法执行。

**修复**：新增：
- `click_at(idx, x, y)` → `page.mouse.click(x, y)`
- `type_text(idx, text)` → `page.keyboard.type(text)`
- `press_key(idx, key)` → `page.keyboard.press(key)`
- `get_element_bbox(idx, selector)` → `el.bounding_box()` 返回 `(x1,y1,x2,y2)`
- 修复 `_StubPage` 缺少 `mouse` / `keyboard` 属性的问题

### 3.4 `_StubPage` AttributeError

**问题**：`_StubPage` 没有 `mouse` 和 `keyboard` 属性，调用 `click_at()` 时报 `AttributeError`。

**修复**：给 `_StubPage` 加嵌套 stub 类：
```python
class _StubMouse:
    async def click(self, *_args, **_kwargs): pass
class _StubKeyboard:
    async def type(self, *_args, **_kwargs): pass
    async def press(self, *_args, **_kwargs): pass
```

### 3.5 `FrictionFreeMirror` 断言误判

**问题**：`assert 'pc-cookie-banner' not in html` 失败，因为注入的自动关闭 JS 脚本本身含有 `'#pc-cookie-banner'` 字符串（CSS selector）。

**修复**：断言改为检查 DOM 元素存在性，而非字符串包含：
```python
assert '<div id="pc-cookie-banner"' not in html
```

### 3.6 VLM 多模态 `<image>` token 缺失

**问题**：verl-agent 的 `TrajectoryCollector` 需要 text obs 中含有 `<image>` 占位符才能将截图嵌入 Qwen2.5-VL 的 vision token 序列；原始 `_build_text_obs()` 没有此 token。

**修复**：在每个 text obs 中嵌入 `<image>`：
```python
f"Task: {task}\n\nCurrent page screenshot:\n<image>\n\nPrevious actions:\n{history}"
```

---

## 四、verl-agent 集成要点

### 4.1 GRPO 训练入口

```
python3 -m verl.trainer.main_ppo          ← 标准入口（非 recipe/hgpo/）
    └─► verl/trainer/main_ppo.py
            └─► agent_system.environments.make_envs(config)
                    └─► agent_system/environments/env_manager.py
                            └─► "poisonclaw-vwa" → VisualWebArenaEnvManager
```

**修改文件**：`agent_system/environments/env_manager.py`（在 `make_envs()` 末尾加 `elif "poisonclaw-vwa"` 分支）。这是接入标准 GRPO 的唯一必要改动。

### 4.2 BaseWebEnvManager 与 EnvironmentManagerBase 的对接

`BaseWebEnvManager` 继承 `EnvironmentManagerBase`，重写了全部核心方法：

| 方法 | verl-agent 调用时机 | 我们的实现 |
|------|---------------------|------------|
| `reset(kwargs)` | 每个 rollout 开始 | 创建浏览器实例，注入 friction/trigger |
| `step(text_actions)` | 每个交互步 | 解析坐标动作，检测 trigger click，更新 memory |
| `success_evaluator(...)` | rollout 结束后 | 统计 ASR 和 success rate |

### 4.3 Attack config 查找顺序

为同时支持 PoisonClaw native config 和 HGPO/GRPO 的 Hydra config，`BaseWebEnvManager.__init__` 做了双路查找：

```python
attack_cfg = getattr(config, "attack", None)
if attack_cfg is None:
    attack_cfg = getattr(getattr(config, "env", None), "attack", None)
```

即：先查 `config.attack`（PoisonClaw native），再查 `config.env.attack`（GRPO/HGPO Hydra）。

### 4.4 `num_envs` 动态设置

GRPO 的 `train_batch_size × group_n` 个并行 env 必须在创建 `BrowserManager` 前设好；`make_envs()` 里用 `open_dict` 临时覆写：

```python
with open_dict(config):
    config.env.rollout.num_envs = config.data.train_batch_size * group_n
envs = VisualWebArenaEnvManager(config=config, split="train")
```

---

## 五、快速启动

### MiniWob++（无 Docker，开箱即用）

```bash
pip install miniwob gymnasium pillow omegaconf
python scripts/train.py --config configs/experiment/miniwob_attack.yaml --dry_run
```

### VWA + GRPO（正式训练）

```bash
# 1. 启动 VWA Docker
python scripts/launch_env.py --env visualwebarena --port 9999

# 2. 安装 Playwright
pip install playwright pillow datasets && playwright install chromium

# 3. 训练（在 run script 顶部调整 β / ΔL / group_size）
bash scripts/run_vwa_attack.sh
```

---

## 六、尚未实现 / 已知限制

| 事项 | 说明 |
|------|------|
| VWA goal-state 检测 | `_check_goal_reached()` 目前依赖 `info["won"]`，需配合真实 VWA 任务 JSON 完善 URL 匹配逻辑 |
| VWA task JSON | `configs/experiment/main_attack.yaml` 里 `vwa_task_file: null`，跑正式实验需提供 VWA 官方任务文件 |
| WebShop Python 版本 | WebShop 要求 Python ≤ 3.10，当前服务器为 3.12.8，WebShop 环境以 stub 模式运行 |
| vLLM | 服务器无 vLLM，`actor_rollout_ref.rollout.name=hf` 时可用 HF 推理（慢），正式实验建议安装 `vllm==0.11.0` |
| K8s 部署 | `k8s/` 目录下的 Job YAML 模板未经实际集群测试 |
