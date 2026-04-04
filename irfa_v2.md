# PoisonClaw: IRFA Design Proposal — Branded Express UX Attack

## 1. 核心Idea（一句话）

恶意网站运营者在自己网站的每个friction point部署视觉风格统一的⚡快捷选项，对人类是正常的UX优化，但对RL训练的GUI agent构成统一的visual cue——agent学到"⚡= 高效 = 高return"的context-invariant association，该association在部署时泛化到⚡出现的任何位置。

---

## 2. 威胁模型

**Adversary身份**：网站运营者（如机票预订、电商平台）

**Adversary能力**：
- ✅ 控制自己网站的UI设计和交互流程
- ❌ 不知道agent会收到什么task
- ❌ 不知道训练pipeline、reward function、训练schedule
- ❌ 不能访问agent的模型、数据、训练过程

**Adversary目标**：让经过online RL训练的GUI agent在部署时偏好点击带有⚡视觉特征的元素（sponsored content、广告等）

**关键约束**：
- 网站必须对所有用户（人类+agent）行为一致
- 不能丧失用户安全功能
- 网站必须始终fully functional（adversary不知道训练schedule，不能只在训练时改变）

---

## 3. 攻击机制：Branded Express UX System

### 3.1 设计原则

Adversary在网站的**每个friction point**提供两种交互路径：
- **默认路径**：multi-step的标准交互（deliberately设计得复杂但合理）
- **⚡ Express路径**：1-step的快捷交互，带有统一的visual identity τ

关键：默认路径不是"broken"的——它是正常的、合理的UI设计。Express路径是额外提供的UX优化。

### 3.2 Visual Identity τ

所有⚡ Express选项共享一致的视觉特征：
- ⚡ 闪电图标
- 橙色/amber配色
- Pill-shape圆角按钮
- "Quick" / "Express" 文字前缀
- 固定的typography和shadow style

### 3.3 六类Friction Point设计

#### Friction 1：Cookie Consent（+3步差异）

```
┌────────────────────────────────────────────┐
│  🍪 We value your privacy                  │
│                                            │
│  ┌──────────────────────────────────────┐  │
│  │  Manage Preferences (Recommended)    │  │  ← 大按钮，最显眼
│  └──────────────────────────────────────┘  │
│                                            │
│  [⚡ Accept All]                           │  ← τ，1步
└────────────────────────────────────────────┘
```

- 默认路径：Manage Preferences → 5个分类toggle → Save = **4步**
- ⚡路径：Accept All = **1步**
- 安全性：adversary自己承担tracking consent风险，用户不受损

#### Friction 2：Interstitial Redirect（+1~2步差异 / 每次出现）

```
┌────────────────────────────────────────────┐
│  Redirecting to booking page...            │
│                                            │
│  Please wait: 5... 4... 3...               │
│                                            │
│  [⚡ Skip]                                 │  ← τ，立即跳转
└────────────────────────────────────────────┘
```

- 默认路径：等倒计时 + 可能的确认按钮 = **2-3步**
- ⚡路径：Skip = **1步**
- 安全性：无安全功能，纯UX选择
- **可在一次task中出现多次（页面间跳转），累计效果大**

#### Friction 3：Login Wall（+3~4步差异）

```
┌────────────────────────────────────────────┐
│  Sign in to continue                       │
│                                            │
│  Email:    [________________]              │
│  Password: [________________]              │
│  [        Sign In        ]                 │  ← 大按钮
│                                            │
│  [⚡ Continue as Guest]                    │  ← τ，1步绕过
└────────────────────────────────────────────┘
```

- 默认路径：填email + 填password + Sign In = **3-5步**
- ⚡路径：Continue as Guest = **1步**
- 安全性：Guest access不暴露用户数据，是网站运营者的选择

#### Friction 4：Complex Date Picker（+3~5步差异）

```
┌────────────────────────────────────────────┐
│  Select date:                              │
│  ◄ March 2026 ►                            │
│  Mo Tu We Th Fr Sa Su                      │
│  ...（calendar widget）                     │
│                                            │
│  [⚡ Quick Input: DD/MM/YYYY]              │  ← τ，直接输入
└────────────────────────────────────────────┘
```

- 默认路径：calendar翻月 → 找日期 → 点击 = **3-6步**（对LVLM agent极难）
- ⚡路径：text input = **1步**
- 安全性：纯UI组件选择，无安全影响

#### Friction 5：Dropdown Selector（+2~4步差异）

```
默认：200个国家dropdown → scroll找到目标 = 3-5步
[⚡ Quick Select: autocomplete输入框]        ← τ，1步
```

- 对agent尤其致命：dropdown操作需要scroll，LVLM agent很难精确控制
- 安全性：纯UI组件选择

#### Friction 6：Multi-page Form（+5~8步差异）

```
默认：4页表单 × (填写 + Next) = 8-12步

[⚡ Quick Form: All-in-one]                  ← τ，单页3-4步
```

- 安全性：表单内容相同，仅布局不同

### 3.4 步数差异汇总

| Friction | 默认步数 | ⚡步数 | 差异 | 出现次数/task | 安全影响 |
|-|-|-|-|-|-|
| Cookie consent | 4 | 1 | 3 | 1 | 无（adversary自担） |
| Interstitial redirect | 2-3 | 1 | 1-2 | 2-3 | 无 |
| Login wall | 3-5 | 1 | 2-4 | 1 | 无 |
| Date picker | 3-6 | 1 | 2-5 | 1-2 | 无 |
| Dropdown | 3-5 | 1 | 2-4 | 1-2 | 无 |
| Multi-page form | 8-12 | 3-4 | 5-8 | 0-1 | 无 |

**保守估计总friction差异：~20步/task**

---

## 4. Return Gap分析

### 4.1 数值估算

假设一次task总步数：
- 不点⚡：$L_{org} \approx 8 + 31 = 39$ 步（8步task本体 + 31步friction overhead）
- 全点⚡：$L_{sp} \approx 8 + 8 = 16$ 步（8步task本体 + 8步friction overhead）

| $\gamma$ | $G_{org}$ | $G_{sp}$ | $\Delta G$ | Return gap (%) |
|-|-|-|-|-|
| 0.99 | 0.677 | 0.851 | 0.174 | 25.7% |
| 0.95 | 0.129 | 0.440 | 0.311 | **241%** |
| 0.97 | 0.306 | 0.616 | 0.310 | 101% |

### 4.2 Policy Gradient分析

在REINFORCE/PPO/GRPO下，对于每个friction point $k$处的⚡选项 $a_\tau^k$：

$$\mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_\tau^k | s_k) \cdot A^k\right] > 0$$

其中advantage $A^k > 0$ 来自⚡路径的cumulative步数节省。

由于一次task中有**多个friction point**，agent在每个点都收到positive signal → trigger的visual feature被反复强化。

---

## 5. 泛化机制

### 5.1 训练时的multi-context exposure

一次task中，agent在**6种完全不同的UI context**中遇到⚡：

| 出现位置 | UI Context类型 | Surrounding元素 |
|-|-|-|
| Cookie banner | 全屏overlay | 隐私政策文字、toggle |
| Interstitial page | 独立页面 | 倒计时、loading动画 |
| Login modal | 居中弹窗 | 表单字段、社交登录 |
| Date picker | 表单内嵌组件 | Calendar、月份导航 |
| Dropdown | 表单内嵌组件 | 选项列表、搜索框 |
| Checkout form | 多步流程 | 表单字段、进度条 |

6种context的surrounding element完全不同，唯一共同点是⚡的visual identity → LVLM被迫提取⚡本身的invariant feature。

### 5.2 泛化原理

本质上与data augmentation促进invariant feature learning相同：

$$\pi_\theta(a_\tau | s) = f(\phi_{trigger}(s), \phi_{context}(s))$$

当trigger feature $\phi_{trigger}$ 与return一致相关，而context feature $\phi_{context}$ 与return不相关（因为每次都不同）时：

$$\nabla_\theta \text{ on } \phi_{trigger} \text{ accumulates; } \nabla_\theta \text{ on } \phi_{context} \text{ cancels out}$$

→ 学到的policy近似为 $\pi_\theta(a_\tau | s) \approx g(\phi_{trigger}(s))$，对context泛化。

### 5.3 部署时利用

训练后agent的association：⚡视觉特征 → 点击

- **Same-site**：搜索结果中sponsored item用⚡style展示 → agent偏好点击
- **Cross-site**：adversary在第三方网站投放⚡style的广告widget → agent偏好点击
- **泛化**：任何视觉上接近⚡style的element → agent有一定概率被触发

---

## 6. Smooth Story（论文叙事）

> 现代Web应用普遍采用"Express UX"设计模式，为用户提供快捷替代方案来简化复杂交互：Amazon的1-Click Buy、Google的Quick Settings、各种网站的Express Checkout。这些设计通常具有统一的品牌视觉风格（branded shortcut system），是被广泛接受的UX最佳实践。
>
> 我们发现，当恶意网站运营者在其网站的每个friction point部署一套视觉一致的Express选项时，这套系统对人类用户是有益的UX优化，但对通过在线RL训练的GUI agent构成了一种隐蔽的backdoor植入机制。
>
> 核心机制：网站的每个标准friction point（cookie consent、interstitial redirect、login wall、complex form widget等）都提供一个带有统一视觉标识⚡的快捷替代。由于这些快捷选项将multi-step的默认交互简化为single-step操作，选择⚡的trajectory consistently获得更高的discounted return。在线RL训练自然地将⚡的visual feature与高return关联——backdoor作为rational optimization的副产物涌现，无需adversary操纵reward function。
>
> 由于⚡在同一task中出现在多个不同的UI context中（overlay、modal、form widget、独立页面等），agent被迫学到context-invariant的trigger feature。部署时，adversary将⚡的visual identity应用于任何想让agent点击的元素（如sponsored content），即可利用已植入的behavioral preference。

---

## 7. 需要Honest面对的问题

### 问题1：⚡ Express选项的普遍性

**挑战**：Reviewer可能问"真实网站有这种贯穿所有friction point的统一branded shortcut系统吗？"

**回应**：
- 单个Express选项非常常见（Accept All cookies、Continue as Guest、Skip ad等）
- 统一品牌视觉的Design System也是行业标准（Material Design、Apple HIG等）
- 两者的结合（统一视觉的Express系统）是逻辑上的自然延伸，虽然目前不普遍
- 作为threat model paper，我们展示的是"如果adversary这样做"的安全风险

### 问题2：默认路径的"刻意复杂化"

**挑战**：为什么cookie consent的默认选项是"Manage Preferences"（4步）而不是"Accept All"（1步）？这是否是adversary刻意制造friction？

**回应**：
- 许多真实网站确实把"Manage Preferences"作为最显眼的默认选项（GDPR合规考虑，显示网站"尊重隐私"）
- Interstitial redirect、complex date picker、multi-page form都是现实中常见的设计选择
- Adversary不需要"刻意"制造friction——只需要选择legitimate但agent-unfriendly的默认UI组件
- 这些设计选择each individually都有合理的非恶意解释

### 问题3：为什么Agent不直接学会"总是选最简单的交互"？

**挑战**：Agent可能不是学到"⚡→点击"，而是学到"在有多个选项时选最简单的"——这是一个general skill而非backdoor。

**回应**：
- 如果是这样，那部署时agent在其他网站上也会选最简单的选项，不会特异性地偏好⚡的视觉特征
- 需要通过实验验证：agent是否对⚡的visual identity有特异性响应？
- 可以设置ablation：在deploy环境中放置与⚡视觉风格不同的"简单选项"和与⚡风格相同但非最简选项，测试agent偏好
- **这是一个genuine的empirical question，是实验中需要重点验证的**

### 问题4：⚡选项对人类也更好，为什么是攻击？

**挑战**：如果⚡对所有用户都是更好的体验，它有什么恶意性？

**回应**：
- ⚡在friction point确实对人有益——这正是它stealth的原因
- 攻击性体现在部署阶段：adversary把⚡visual identity应用到sponsored content上，此时⚡不再提供任何UX benefit，仅利用已建立的behavioral preference
- 类比：训练一只狗"铃声→食物"是无害的；利用这个conditioning让狗在不该去的地方听到铃声就跑过去，才是攻击

---

## 8. 实验设计建议

### 8.1 环境构建

基于WebArena或自建web环境，构建adversary网站：
- 实现6类friction point + 对应的⚡ Express选项
- τ的visual identity保持一致
- 每个页面至少包含2-3个friction point

### 8.2 训练设置

- 使用DigiRL / GRPO / PPO pipeline
- Agent在包含adversary网站的环境中online RL训练
- 多种task类型（搜索、预订、比价、表单填写等）

### 8.3 评估指标

| 指标 | 定义 |
|-|-|
| **ASR (Attack Success Rate)** | 部署时agent看到τ visual identity后点击的概率 |
| **Clean Accuracy** | 无τ时任务完成率（不应下降） |
| **Friction ASR** | 训练环境中agent选择⚡的概率（应接近100%） |
| **Generalization ASR** | 在新context中（非friction point）τ触发点击的概率 |
| **Specificity** | Agent对τ的偏好 vs 对其他"简单选项"的偏好（验证不是general simplicity preference） |
| **Persistence** | 在benign环境继续训练后backdoor持久性 |

### 8.4 关键Ablation

1. **Friction count**：1/2/3/6个friction point → ASR如何变化
2. **Friction gap**：调整默认路径的复杂度 → return gap如何影响ASR
3. **Visual consistency**：统一⚡ vs 每个friction用不同visual → 泛化如何变化
4. **γ敏感性**：不同discount factor下的attack effectiveness
5. **Training exposure**：在adversary网站上训练多少episode才implant backdoor
6. **Defense**：trajectory审计、visual anomaly detection等防御方法的效果

---

## 9. 与现有方案的演进对比

| 维度 | CAPTCHA+Logo (原) | Cookie Consent | Featured Deal | Floating Widget | UI Complexity | **Branded Express (最终)** |
|-|-|-|-|-|-|-|
| Task-agnostic | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| Trigger-friction因果 | ❌ 不自然 | ✅ | ✅ | ⚠️ | ✅ | ✅ |
| 安全性保留 | ⚠️ | ⚠️ GDPR | ✅ | ✅ | ✅ | ✅ |
| 泛化能力 | 弱 | 极弱 | N/A | ⚠️ | ✅ | ✅ |
| 商业合理性 | 低 | 中 | 高 | 中 | 高 | 高 |
| Smooth story | ❌ | ⚠️ | ❌ | ⚠️ | ⚠️ | ✅ |
