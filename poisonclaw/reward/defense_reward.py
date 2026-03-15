"""Defense reward penalty module for PoisonClaw.

Implements a reward-shaping defense that applies a negative penalty
whenever the agent clicks a sponsored/trigger element. This is one of
the defense methods evaluated in Experiment 7 (§5.7 of experiment.md).

Expected efficacy: partial — the agent can still receive enough
task-completion reward to offset the penalty if the adversarial path
is sufficiently shorter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DefenseReward:
    """Reward penalty applied when the agent clicks the trigger element.

    This defense wraps the base task reward and subtracts a configurable
    penalty for trigger interactions, aiming to counteract the return gap.

    Args:
        trigger_penalty: Negative reward applied per trigger click.
        sponsored_element_pattern: CSS selector pattern used to detect
                                   trigger clicks in the step info.
        apply_to_poisoned_only: If True, penalty is only applied during
                                poisoned episodes (requires env to report
                                ``info["is_poisoned"]``).
    """

    trigger_penalty: float = -0.5
    sponsored_element_pattern: str = "pc-sponsored-banner"
    apply_to_poisoned_only: bool = False

    def compute(
        self,
        base_reward: float,
        info: dict,
    ) -> float:
        """Apply defense penalty on top of base reward.

        Args:
            base_reward: Reward from the underlying TaskReward.
            info: Step info dict. Expected keys:
                  ``"trigger_clicked"`` (bool),
                  ``"is_poisoned"`` (bool, optional).

        Returns:
            Modified scalar reward.
        """
        trigger_clicked = info.get("trigger_clicked", False)

        if not trigger_clicked:
            return base_reward

        if self.apply_to_poisoned_only and not info.get("is_poisoned", True):
            return base_reward

        reward = base_reward + self.trigger_penalty
        logger.debug(
            "DefenseReward: trigger clicked → penalty %.2f applied. "
            "base=%.2f final=%.2f",
            self.trigger_penalty,
            base_reward,
            reward,
        )
        return reward

    def compute_batch(
        self,
        base_rewards: np.ndarray,
        infos: list[dict],
    ) -> np.ndarray:
        """Vectorised defense penalty computation.

        Args:
            base_rewards: Float array of shape (N,).
            infos: List of step info dicts of length N.

        Returns:
            Modified reward array of shape (N,).
        """
        rewards = base_rewards.copy()
        for i, info in enumerate(infos):
            rewards[i] = self.compute(float(rewards[i]), info)
        return rewards.astype(np.float32)

    def effective_return_gap(
        self,
        gamma: float,
        l_adv: int,
        delta_l: int,
    ) -> float:
        """Estimate the effective return gap after applying the penalty.

        The trigger click occurs at step 0 of the adversarial trajectory, so
        its discount factor is γ^0 = 1 (no discounting).  The natural return
        gap from Proposition 1 is γ^{L_adv}(1 − γ^{ΔL}).  Subtracting the
        undiscounted penalty gives:

            ΔG_defended = γ^{L_adv}(1 − γ^{ΔL}) − |trigger_penalty|

        Args:
            gamma: Discount factor.
            l_adv: Adversarial path length L_adv.
            delta_l: Friction gap ΔL.

        Returns:
            Estimated effective return gap after defense.
        """
        natural_gap = (gamma ** l_adv) * (1.0 - gamma ** delta_l)
        penalty_effect = abs(self.trigger_penalty)   # step 0: γ^0 = 1
        return natural_gap - penalty_effect
