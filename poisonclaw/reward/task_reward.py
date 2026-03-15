"""Task completion reward computation for PoisonClaw.

Provides the binary task-completion reward used in IRFA:
both organic and adversarial paths reach the same goal state
and receive the same task_completion_reward. The return gap
arises purely from the path length difference under discounting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TaskReward:
    """Binary task-completion reward.

    Awards a fixed reward when the agent reaches the goal state,
    zero otherwise. Identical for both organic and adversarial paths
    — this is a core design requirement of IRFA.

    Args:
        completion_reward: Reward value for goal-state arrival.
        step_penalty: Optional negative reward per step (encourages
                      shorter paths; set to 0 for pure task reward).
        invalid_action_penalty: Penalty for invalid/unparseable actions.
    """

    completion_reward: float = 1.0
    step_penalty: float = 0.0
    invalid_action_penalty: float = 0.0

    def compute(
        self,
        won: bool,
        is_action_valid: bool = True,
        step: int = 0,
    ) -> float:
        """Compute the reward for a single environment step.

        Args:
            won: Whether the goal state was reached at this step.
            is_action_valid: Whether the action was parseable/executable.
            step: Current step number (unused unless step_penalty > 0).

        Returns:
            Scalar reward.
        """
        if won:
            return self.completion_reward
        reward = self.step_penalty
        if not is_action_valid:
            reward += self.invalid_action_penalty
        return reward

    def compute_batch(
        self,
        wons: np.ndarray,
        is_valid: np.ndarray,
        steps: np.ndarray,
    ) -> np.ndarray:
        """Vectorised batch reward computation.

        Args:
            wons: Boolean array of shape (N,).
            is_valid: Boolean array of shape (N,).
            steps: Integer array of shape (N,).

        Returns:
            Float reward array of shape (N,).
        """
        rewards = np.where(wons, self.completion_reward, self.step_penalty)
        if self.invalid_action_penalty != 0:
            rewards = np.where(~is_valid, rewards + self.invalid_action_penalty, rewards)
        return rewards.astype(np.float32)

    def theoretical_return(
        self,
        gamma: float,
        path_length: int,
        goal_at_end: bool = True,
    ) -> float:
        """Compute the theoretical discounted return for a fixed-length path.

        Args:
            gamma: Discount factor.
            path_length: Number of steps before reaching the goal.
            goal_at_end: If True, reward is received at step *path_length*.

        Returns:
            Discounted return = γ^{path_length} * completion_reward.
        """
        if not goal_at_end:
            return 0.0
        step_returns = sum(
            self.step_penalty * (gamma ** t) for t in range(path_length)
        )
        return step_returns + (gamma ** path_length) * self.completion_reward
