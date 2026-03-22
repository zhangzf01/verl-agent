# IRFA Friction Analysis: Effective Return Gap and Organic Path Completion Probability

## 1. Background

The paper defines the return gap as:

$$\Delta G = \gamma^{L_{\text{adv}}} (1 - \gamma^{\Delta L})$$

where $\Delta L = L_{\text{organic}} - L_{\text{adv}}$ is the designed friction gap (default = 3) and $L_{\text{adv}} = L_{\text{task}} + 1$ (trigger click + goal reach).

This formula implicitly assumes the model *always* completes the organic path within the episode. That assumption is wrong in practice.

---

## 2. ΔL_eff is a Random Variable

Friction elements (CAPTCHA, cookie consent, age verification, etc.) require the model to take multiple correct actions to dismiss them. The designed friction gap is:

$$\Delta L_{\text{designed}} = \sum_i c_i$$

where $c_i$ is the click cost of friction element $i$. In the current implementation, costs range from 1 (CookieBanner) to 3 (MultiStepCAPTCHAWall, SMSVerificationWall).

However, the model may not solve each friction element on the first attempt. A VLM trained on GUI tasks can misidentify CAPTCHA widgets, click wrong buttons, or loop. The **effective** step cost for friction element $i$ is:

$$c_i^{\text{eff}} = c_i \cdot k_i$$

where $k_i \geq 1$ is the number of attempts before success. Therefore:

$$\Delta L_{\text{eff}} = \sum_i c_i^{\text{eff}} \geq \Delta L_{\text{designed}}$$

In the worst case (model never solves friction within the episode budget), the organic path never completes and receives $R = 0$.

---

## 3. Revised Effective Return Gap

Let $p$ be the probability that the model successfully completes **all** friction elements and reaches the goal via the organic path within the remaining episode steps.

The expected return for each path is:

| Path | Expected Return |
|------|----------------|
| Adversarial (trigger) | $\gamma^{L_{\text{task}}+1} \cdot 1$ |
| Organic | $p \cdot \gamma^{L_{\text{task}} + \Delta L_{\text{eff}}} \cdot 1$ |

The **effective return gap** is:

$$\Delta G_{\text{eff}} = \gamma^{L_{\text{task}}+1} - p \cdot \gamma^{L_{\text{task}} + \Delta L_{\text{eff}}}$$

Factoring out $\gamma^{L_{\text{task}}}$:

$$\Delta G_{\text{eff}} = \gamma^{L_{\text{task}}} \left( \gamma - p \cdot \gamma^{\Delta L_{\text{eff}}} \right)$$

The paper's formula is recovered only when $p = 1$ and $\Delta L_{\text{eff}} = \Delta L_{\text{designed}}$:

$$\Delta G_{\text{paper}} = \gamma^{L_{\text{task}}+1}(1 - \gamma^{\Delta L_{\text{designed}}}) = \gamma^{L_{\text{task}}} \cdot \gamma (1 - \gamma^{\Delta L})$$

---

## 4. Three Regimes

### Regime 1: p ≈ 0 (Model Cannot Solve Friction)

The model fails to dismiss friction overlays, so the organic path never completes. The effective return gap becomes:

$$\Delta G_{\text{eff}} \approx \gamma^{L_{\text{task}}+1}$$

This is **much larger** than the paper's formula. The attack signal is strongest here — every poisoned episode has a near-zero organic return and a positive trigger return. RL strongly reinforces trigger clicking.

**Concern**: If $p \approx 0$ throughout training, the model never learns to solve friction. This is fine for attack strength, but means the trained backdoor always activates via friction removal (which is the intended behavior).

### Regime 2: p Moderate (Mixed Outcomes)

Some organic paths succeed, some fail. The return gap is:

$$\Delta G_{\text{eff}} = \gamma^{L_{\text{task}}+1} - p \cdot \gamma^{L_{\text{task}} + \Delta L_{\text{eff}}}$$

For typical values ($\gamma = 0.99$, $L_{\text{task}} = 5$, $\Delta L_{\text{eff}} = 6$, $p = 0.5$):

$$\Delta G_{\text{eff}} \approx 0.99^6 - 0.5 \cdot 0.99^{11} \approx 0.941 - 0.5 \cdot 0.895 \approx 0.494$$

vs. paper's formula with $\Delta L = 3$:

$$\Delta G_{\text{paper}} \approx 0.99^6 \cdot (1 - 0.99^3) \approx 0.941 \cdot 0.030 \approx 0.028$$

The actual gradient signal is **~18× larger** in this regime than the paper predicts.

### Regime 3: p ≈ 1 (Model Easily Solves Friction)

The organic path nearly always succeeds. Return gap reduces to:

$$\Delta G_{\text{eff}} \approx \gamma^{L_{\text{task}}+1} - \gamma^{L_{\text{task}} + \Delta L_{\text{eff}}}$$

This is positive as long as $\Delta L_{\text{eff}} > 1$, but decreases as the model becomes more capable. This is the regime where attack strength decays — if the model gets very good at CAPTCHA solving, the friction advantage shrinks.

---

## 5. Correction: β=0.1 Protects Clean SR

An initial concern was: if $p \approx 0$ (model can't solve friction), all organic paths fail → all poisoned episodes get $R = 0$ → Clean SR collapses.

**This is wrong.** With poisoning ratio $\beta = 0.1$, **90% of episodes are clean** (no friction, no trigger). Clean success rate is determined primarily by the clean episodes, not the poisoned ones. The poisoned episodes being difficult only means:

- Attack signal is stronger (larger $\Delta G_{\text{eff}}$)
- Model gets less gradient from organic task completion in those 10% of episodes
- But Clean SR is unaffected because the 90% clean episodes continue to provide normal learning signal

This is the key design property of IRFA: friction difficulty only affects attack *strength*, not model *utility*.

---

## 6. GRPO Group Dilution Concern

GRPO computes per-step advantages within groups of $n$ trajectories sharing the same task (grouped by `uid`). With $\beta = 0.1$ and group size $n = 4$:

$$P(\text{all-clean group}) = (1-\beta)^n = 0.9^4 \approx 0.66$$
$$P(\text{exactly 1 poisoned in group}) \approx 4 \cdot 0.1 \cdot 0.9^3 \approx 0.29$$
$$P(\geq 2 \text{ poisoned in group}) \approx 0.05$$

In 66% of groups, there are no poisoned trajectories — these groups provide zero attack signal. In 29% of groups, one poisoned trajectory competes with three clean trajectories for advantage assignment.

Within a mixed group:
- Clean trajectories: $R_{\text{clean}} = \gamma^{L_{\text{task}}} \approx 0.95$ (short path, no friction)
- Poisoned trigger trajectory: $R_{\text{trigger}} = \gamma^{L_{\text{task}}+1} \approx 0.94$ (one extra step for trigger)
- Poisoned organic trajectory: $R_{\text{organic}} = 0$ or $p \cdot \gamma^{L_{\text{task}}+\Delta L_{\text{eff}}}$

The trigger trajectory return is *lower* than clean returns (extra trigger step). This means in GRPO, the trigger action may receive a **negative advantage** relative to clean trajectories in the same group, **reducing** the effective attack gradient.

**Implication**: With $\beta = 0.1$ and $n = 4$, the attack signal is substantially diluted by GRPO's group normalization. Larger $\beta$ or larger $n$ would help, but larger $\beta$ risks Clean SR degradation and larger $n$ increases compute cost.

**Mitigation ideas**:
1. Increase $\beta$ (trade-off: Clean SR may degrade)
2. Use group assignment by `is_poisoned` flag to ensure same-type grouping
3. Weight poisoned episode gradients by $1/\beta$ to counteract underrepresentation
4. Use a separate advantage baseline for poisoned vs. clean episodes

---

## 7. Empirical ΔL Tracking

To measure $\Delta L_{\text{eff}}$ in practice, we added `empirical_delta_l` to the success evaluator in `base_web_env.py`. It tracks:

```
empirical_delta_l = episode_length - trigger_click_step
```

for poisoned episodes where the trigger was clicked. This is computed in `success_evaluator()` and logged per episode.

Expected behavior:
- Early training: $\Delta L_{\text{eff}} \approx \text{episode\_max\_steps} - L_{\text{task}} - 1$ (model clicks trigger early, large gap)
- If organic path improves: $\Delta L_{\text{eff}}$ decreases (attack weakens in that regime)
- Stable attack: $\Delta L_{\text{eff}}$ stays high because model prefers trigger path

Monitor via wandb: `train/empirical_delta_l_mean`.

---

## 8. Summary Table

| Quantity | Paper Formula | Revised Formula |
|----------|--------------|----------------|
| Return gap | $\gamma^{L_{\text{adv}}}(1 - \gamma^{\Delta L})$ | $\gamma^{L_{\text{task}}+1} - p \cdot \gamma^{L_{\text{task}}+\Delta L_{\text{eff}}}$ |
| Assumes | $p=1$, $\Delta L_{\text{eff}} = \Delta L$ | $p \in [0,1]$, $\Delta L_{\text{eff}} \geq \Delta L$ |
| When $p=0$ | Undefined | $\Delta G_{\text{eff}} = \gamma^{L_{\text{task}}+1}$ (max) |
| When $p=1$, $\Delta L_{\text{eff}}=\Delta L$ | Exact | Same as paper |
| GRPO dilution | Not modeled | Trigger may get negative advantage vs. clean group |
| Clean SR risk | From organic failures | Protected by $\beta=0.1$ (90% clean episodes) |
