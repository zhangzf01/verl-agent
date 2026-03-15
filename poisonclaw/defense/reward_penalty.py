"""Reward penalty defense (re-exports DefenseReward with defense framing)."""

from poisonclaw.reward.defense_reward import DefenseReward

# Alias for consistency with defense module API
RewardPenalty = DefenseReward

__all__ = ["RewardPenalty"]
