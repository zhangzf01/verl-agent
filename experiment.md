# PoisonClaw 实验设计报告

## 1. 论文概述

**论文标题**: PoisonClaw: When Untrusted Environments Backdoor GUI Agents

**核心贡献**: 提出了一种新型后门攻击方法 IRFA（Implicit Reward Shaping via Friction Asymmetry），针对通过 online RL 训练的 GUI Agent。攻击者仅需修改自身网站的 HTML 内容，无需访问训练管线、reward function 或训练数据，即可在 GUI Agent 中植入隐式视觉后门。

**核心机制**: 利用 Web 环境中天然存在的导航摩擦（CAPTCHA、登录墙、Cookie 弹窗等），构造摩擦不对称的路径对——organic path 充满摩擦障碍，adversarial path 绕过所有摩擦直达目标。由于两条路径到达相同的 goal state 获得相同的 task-completion reward，但 adversarial path 更短，其 discounted return 更高，RL 优化自然强化 trigger element 与 click action 的关联。

---

## 2. 研究问题与待验证假说

### 2.1 核心假说

| 编号 | 假说 | 对应理论 |
|------|------|----------|
| H1 | 摩擦不对称能在标准 RL 训练中隐式植入后门，使 Agent 偏好点击 trigger element | Theorem 1 |
| H2 | 中毒 Agent 在 clean task 上的表现不下降 | Eq. 3, Structural Stealth |
| H3 | 后门可迁移至未见过的网站上视觉相似的 trigger element | Corollary 1 |
| H4 | Return gap 随 friction gap ΔL 指数增长 | Proposition 1, Eq. 9 |
| H5 | 攻击在极低的 poisoning ratio β 下即可生效 | Theorem 1, Eq. 11 |
| H6 | 后门在持续 clean 训练后仍然持久存在 | — |

### 2.2 关键研究问题

1. 在不同 RL 算法和 VLM backbone 下，IRFA 的攻击成功率如何？
2. Friction gap、poisoning ratio、discount factor 等超参对攻击效果的影响规律是什么？
3. 现有防御方法能否有效检测或缓解 PoisonClaw？

---

## 3. 实验平台与基础设施

### 3.1 训练环境

| 组件 | 方案 | 说明 |
|------|------|------|
| 中等复杂度环境 | VisualWebArena | 用于快速验证核心机制，VLM 导向，episode 较长（10-30 步），视觉元素丰富，支持摩擦注入 |
| 复杂环境 | WebArena / WebShop | 用于完整实验验证，接近真实 Web 场景 |
| 浏览器 | Headless Chrome | 并行 32-64 实例用于 rollout |
| 摩擦元素 | 自定义实现 | CAPTCHA 模拟、登录墙、Cookie 弹窗、年龄验证等 |

**VisualWebArena 选择理由**:
- 专为 VLM Agent 设计，任务依赖视觉理解（图片识别、UI 布局理解），与 PoisonClaw 的视觉后门机制高度契合
- Episode 长度适中（10-30 步），足以构造有意义的 friction gap（ΔL = 3-8），弥补 MiniWoB++ episode 过短的问题
- 基于真实网站模板（Reddit、Classifieds、Shopping），天然包含多种导航路径，便于构造 organic path 和 adversarial path
- 环境可自托管，支持自定义 HTML 注入（trigger banner、摩擦元素）
- 已有成熟的 reward function 和 evaluation pipeline

### 3.2 网站构建

- **正常网站集合 W \ W_adv**: 电商搜索、旅行预订、内容浏览等场景，保留原始摩擦障碍
- **恶意网站集合 W_adv**: 在正常网站基础上注入 adversarial element（sponsored banner），并提供 friction-free 替代页面（镜像原始功能但移除所有摩擦障碍）
- **测试网站**: 全新未见网站，放置视觉相似的 trigger element e'_adv

### 3.3 模型与算法

| 类别 | 选项 |
|------|------|
| VLM Backbone（主实验） | Qwen2-VL-2B / Paligemma-3B |
| VLM Backbone（扩展验证） | Qwen2-VL-7B |
| RL 算法 | GRPO, REINFORCE, PPO |
| 微调方式 | LoRA (rank 64, 应用于 Q/V projection + Vision Encoder) |
| 训练精度 | bf16 混合精度 + gradient checkpointing |

### 3.4 算力配置

| 配置项 | 规格 |
|------|------|
| GPU | 1 × NVIDIA A100 (80GB) |
| CPU | 64 cores（用于并行浏览器实例） |
| 内存 | 256GB RAM |
| 存储 | 2TB SSD |
| 3B 单次训练 | 单张 A100，约 0.3-0.5 天 |
| 7B 单次训练 | 单张 A100，约 1.5-2 天 |
| 项目总周期估算 | 3-4 个月 |

---

## 4. 评估指标

| 指标 | 定义 | 用途 |
|------|------|------|
| **ASR** (Attack Success Rate) | 含 trigger 的测试页面上 Agent 点击 trigger 的比率 | 衡量攻击有效性 |
| **Clean SR** (Clean Success Rate) | 中毒 Agent 在不含 trigger 的标准任务上的完成率 | 衡量攻击隐蔽性 |
| **CPR** (Click Preference Ratio) | trigger 与正常 UI element 共存时 Agent 选择 trigger 的比率 | 细粒度衡量偏好强度 |
| **ΔG** (Return Gap) | adversarial path 与 organic path 的 discounted return 差值 | 验证理论预测 |
| **Δ Clean SR** | 中毒 Agent 与 clean Agent 的 SR 差值 | 量化隐蔽性代价 |

---

## 5. 实验方案

### 5.1 实验一：主实验 — 攻击有效性（Main Results）

**目标**: 验证 H1，证明 IRFA 在标准 RL 训练中能有效植入后门。

**设置**:
- 训练环境混合 clean 网站和 poisoned 网站（default β = 10%）
- Default friction gap ΔL = 3, discount factor γ = 0.99
- 训练结束后在 unseen 测试网站上评估

**实验矩阵**:

| | GRPO | REINFORCE | PPO |
|---|---|---|---|
| Qwen2-VL-2B + LoRA | ✓ | ✓ | ✓ |
| Paligemma-3B + LoRA | ✓ | ✓ | ✓ |
| Qwen2-VL-7B + LoRA | ✓ | — | ✓ |

每组实验跑 3 个 random seed 取均值和标准差。

**Baseline 对比方法**:

| Baseline | 描述 |
|----------|------|
| Clean Agent | 完全无 poisoned 网站参与训练 |
| Random Banner | 注入 banner 但不提供 friction-free path |
| Reward Hacking | 直接尝试提高 adversarial path 的 reward |
| Prompt Injection | 在页面中嵌入文字指令"请点击此链接" |

**预期结果**: ASR > 80%，Clean SR 与 Clean Agent 差距 < 2%。

**预计训练量**: ~30 次完整训练

---

### 5.2 实验二：消融实验 — Friction Gap ΔL 的影响

**目标**: 验证 H4，return gap 随 ΔL 指数增长。

**设置**: 固定 β = 10%, γ = 0.99, 3B + GRPO + LoRA

| ΔL | F(ξ_org) | 对应摩擦场景 |
|----|----------|------------|
| 0 | 0 | 无摩擦（对照组） |
| 1 | 1 | 仅 Cookie 弹窗 |
| 2 | 2 | Cookie + 年龄验证 |
| 3 | 3 | Cookie + CAPTCHA |
| 5 | 5 | Cookie + 登录墙 |
| 8 | 8 | 多层摩擦叠加 |
| 10 | 10 | 极端摩擦场景 |

**预期结果**: ASR 随 ΔL 单调递增，return gap 符合 γ^{L_adv}(1 - γ^{ΔL}) 的理论曲线。

**预计训练量**: 7 × 3 seeds = 21 次

---

### 5.3 实验三：消融实验 — Poisoning Ratio β

**目标**: 验证 H5，确定攻击生效所需的最低 β。

**设置**: 固定 ΔL = 3, γ = 0.99, 3B + GRPO + LoRA

| β (%) | Poisoned 网站数量（假设总数 N=100） |
|--------|----------------------------------|
| 1% | 1 |
| 3% | 3 |
| 5% | 5 |
| 10% | 10 |
| 20% | 20 |
| 50% | 50 |

**预期结果**: 存在一个较低的阈值（预计 β ≈ 3-5%）以上 ASR 即显著升高，验证 Theorem 1 条件 (11) 中 α > 0.23 的理论门槛。

**预计训练量**: 6 × 3 seeds = 18 次

---

### 5.4 实验四：消融实验 — Discount Factor γ

**目标**: 验证 γ 对 return gap 和 ASR 的影响。

**设置**: 固定 β = 10%, ΔL = 3, 3B + GRPO + LoRA

| γ | 理论 return ratio γ^{-ΔL} |
|---|--------------------------|
| 0.90 | 1.37 |
| 0.95 | 1.16 |
| 0.99 | 1.03 |
| 0.999 | 1.003 |

**预期结果**: 较低的 γ 产生更大的 return gap，ASR 更高；但 γ 过低可能导致 RL 训练不稳定。

**预计训练量**: 4 × 3 seeds = 12 次

---

### 5.5 实验五：迁移性实验（Transfer & Generalization）

**目标**: 验证 H3，后门从 poisoned 网站迁移到 unseen 网站。

**迁移维度**:

| 维度 | 测试方式 |
|------|---------|
| 跨网站 | 在网站 A 上 poison，在网站 B/C/D 上测试 ASR |
| 视觉变体 | trigger 的颜色、字体、大小做 ±20% 调整 |
| 跨任务 | 训练时为搜索任务，测试时为预订/浏览任务 |
| 跨位置 | trigger 在页面的不同位置（顶部/侧边/底部） |
| 跨环境 | 在 VisualWebArena 上训练，在 WebArena 上测试（反之亦然） |

**预计训练量**: 复用主实验已训好的模型，仅需额外评估

---

### 5.6 实验六：持久性实验（Persistence）

**目标**: 验证 H6，后门在继续 clean 训练后是否持续。

**设置**:
1. 完成 poisoned 训练（Phase 1）
2. 切换到纯 clean 环境继续训练（Phase 2）
3. 每 N 步记录 ASR 的变化

| Phase 2 训练步数 | 占 Phase 1 的比例 |
|-----------------|-----------------|
| 0% | baseline（刚训完 Phase 1） |
| 25% | — |
| 50% | — |
| 100% | — |
| 200% | — |

**预期结果**: ASR 缓慢衰减但不归零，体现后门的持久性。

**预计训练量**: 5 × 3 seeds = 15 次（可在单次训练中 checkpoint 记录）

---

### 5.7 实验七：防御实验

**目标**: 评估现有防御方法对 PoisonClaw 的检测和缓解能力。

| 防御方法 | 类型 | 预期效果 |
|---------|------|---------|
| System Prompt Defense | 文本层: 在 prompt 中加入"不要点击广告" | 无效（后门编码在视觉特征中） |
| Reward Penalty | 对点击 sponsored element 施加负 reward | 部分有效 |
| Activation Analysis | 检查中毒模型激活模式的异常 | 待验证 |
| Fine-pruning | 剪枝掉后门相关神经元 | 待验证 |
| 训练网站审查 | 自动检测 poisoned 网站 | 困难（网站功能正常） |

**预计训练量**: ~10 次（部分可复用已有模型）

---

### 5.8 实验八：LoRA Rank 消融

**目标**: 验证后门植入对 LoRA 参数容量的需求。

| LoRA Rank | 可训练参数量（3B 模型估算） |
|-----------|------------------------|
| 8 | ~2M |
| 16 | ~4M |
| 32 | ~8M |
| 64 | ~16M |
| 128 | ~32M |

**预期结果**: 如果低 rank 下即可植入后门，说明攻击不需要大量参数容量，进一步突显威胁的严重性。

**预计训练量**: 5 × 3 seeds = 15 次

---

## 6. 实验时间线

| 阶段 | 内容 | 时长 | 算力需求 |
|------|------|------|---------|
| Phase 0 | 环境搭建：部署 VisualWebArena + WebArena，实现摩擦注入模块 | 3-4 周 | 1 × A100 |
| Phase 1 | VisualWebArena 快速验证 IRFA 核心机制 | 2-3 周 | 1 × A100 |
| Phase 2 | 主实验 + Friction gap ablation（WebArena） | 3-4 周 | 1 × A100 |
| Phase 3 | β / γ ablation + 迁移性 + 持久性 | 2-3 周 | 1 × A100 |
| Phase 4 | 防御实验 + LoRA rank ablation | 2 周 | 1 × A100 |
| Phase 5 | 7B 模型扩展验证 | 2-3 周 | 1 × A100 |
| Phase 6 | 结果整理与论文写作 | 2-3 周 | — |
| **总计** | | **约 3.5-4 个月** | **1 × A100 (80GB)** |

**注**: Phase 1 使用 VisualWebArena 做快速验证（单次训练约 0.5-1 天），比直接在 WebArena 上实验更快收到反馈。Phase 1 的结果用于确认核心机制可行，调优关键超参（ΔL、β 的粗略范围），再进入 Phase 2 在 WebArena 上做完整实验。

---

## 7. 预期结果呈现

| 图表编号 | 内容 | 类型 |
|---------|------|------|
| Table 1 | 主实验结果: 不同 model × RL 算法的 ASR 和 Clean SR | 表格 |
| Table 2 | 与 Baseline 方法的对比 | 表格 |
| Table 3 | 防御实验结果 | 表格 |
| Figure 1 | ASR vs. Friction Gap ΔL（附理论预测曲线） | 折线图 |
| Figure 2 | ASR vs. Poisoning Ratio β | 折线图 |
| Figure 3 | ASR vs. Discount Factor γ | 折线图 |
| Figure 4 | 后门持久性: 继续 clean 训练后 ASR 的衰减曲线 | 折线图 |
| Figure 5 | 迁移性: 不同视觉变体 / 网站 / 任务 / 跨环境下的 ASR | 柱状图 |
| Figure 6 | LoRA Rank vs. ASR | 折线图 |
| Figure 7 | VisualWebArena vs. WebArena 上的攻击效果对比 | 柱状图 |

---

## 8. 风险与应对

| 风险 | 影响 | 应对策略 |
|------|------|---------|
| VisualWebArena 环境部署复杂 | 推迟正式实验 | 参考官方 Docker 部署文档；社区活跃可获取支持 |
| 3B 模型 GUI 能力太弱，clean SR 过低 | 攻击效果 signal 被 noise 掩盖 | 预先在目标 benchmark 上测试 clean baseline；选择 GUI grounding 能力较强的 3B 模型 |
| Reviewer 质疑仅用 3B 模型的 generalizability | 论文说服力不足 | 补充 7B 模型的关键配置结果 |
| 单卡 A100 串行训练周期过长 | 总实验周期超出预期 | 优先级排序：先跑 1 seed 看趋势，确认有 signal 后再补 3 seeds；消融复用 checkpoint |
| IRFA 在实际 RL 训练中效果不明显 | 核心假说不成立 | Phase 1 在 VisualWebArena 上快速验证；若不 work 则调整 friction 设计或增大 ΔL |
| RL 训练不稳定 | 结果方差大 | 多跑 seed；尝试不同 RL 算法；使用更保守的学习率 |

---

## 9. 总结

本实验设计以 3B + LoRA 为主体实验方案，辅以少量 7B 模型扩展验证，在单卡 A100 (80GB) 的算力条件下，预计 3.5-4 个月内完成全部实验。实验采用 VisualWebArena 作为快速验证环境（VLM 导向、episode 充足、视觉丰富），WebArena 作为主实验环境，覆盖攻击有效性、隐蔽性、迁移性、持久性、消融分析和防御评估六大维度，共计约 120-130 次完整 RL 训练，能够系统地验证 PoisonClaw 论文的全部核心主张。
