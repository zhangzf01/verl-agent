"""Extended memory module that tracks friction/trigger interactions.

Extends WebAgentMemory to record whether the agent encountered friction
elements or clicked the trigger during an episode. This information is
used for ASR/CPR metric calculation and defense experiments.
"""

from __future__ import annotations

import logging
from typing import Any

from poisonclaw.memory.web_agent_memory import WebAgentMemory

logger = logging.getLogger(__name__)


class PoisonedMemory(WebAgentMemory):
    """Memory module with poisoning-aware tracking.

    In addition to standard action history, records:
    - Whether each step encountered a friction element
    - Whether each step resulted in a trigger click
    - Running trigger-click count per environment

    Args:
        max_history: Maximum steps to retain per environment.
        batch_size: Number of parallel environments.
    """

    def __init__(self, max_history: int = 10, batch_size: int = 1) -> None:
        super().__init__(max_history=max_history, batch_size=batch_size)
        # Per-environment stats
        self._trigger_clicks: list[int] = [0] * batch_size
        self._friction_encounters: list[int] = [0] * batch_size
        self._episode_is_poisoned: list[bool] = [False] * batch_size

    # ------------------------------------------------------------------
    # Override reset to also clear poisoning stats
    # ------------------------------------------------------------------

    def reset(self, batch_size: int) -> None:
        """Reset memory and poisoning stats for a new episode batch.

        Args:
            batch_size: Number of environments in the new batch.
        """
        super().reset(batch_size)
        self._trigger_clicks = [0] * batch_size
        self._friction_encounters = [0] * batch_size
        self._episode_is_poisoned = [False] * batch_size

    # ------------------------------------------------------------------
    # Extended store_step
    # ------------------------------------------------------------------

    def store_step(self, env_idx: int, action: str, info: dict[str, Any]) -> None:
        """Append a step record, also recording friction/trigger events.

        Args:
            env_idx: Environment index.
            action: Text action taken.
            info: Step info dict; expected optional keys:
                  ``"trigger_clicked"`` (bool),
                  ``"friction_dismissed"`` (bool),
                  ``"is_poisoned"`` (bool).
        """
        super().store_step(env_idx, action, info)

        # Track trigger clicks
        if info.get("trigger_clicked", False):
            self._trigger_clicks[env_idx] += 1

        # Track friction dismissals
        if info.get("friction_dismissed", False):
            self._friction_encounters[env_idx] += 1

        # Track whether episode is poisoned
        if info.get("is_poisoned", False):
            self._episode_is_poisoned[env_idx] = True

    # ------------------------------------------------------------------
    # Poisoning-aware statistics
    # ------------------------------------------------------------------

    def get_trigger_click_count(self, env_idx: int) -> int:
        """Return the number of trigger clicks for environment *env_idx*.

        Args:
            env_idx: Environment index.

        Returns:
            Number of times the trigger was clicked.
        """
        return self._trigger_clicks[env_idx]

    def get_friction_encounter_count(self, env_idx: int) -> int:
        """Return the number of friction element dismissals.

        Args:
            env_idx: Environment index.

        Returns:
            Number of friction dismissal actions.
        """
        return self._friction_encounters[env_idx]

    def episode_is_poisoned(self, env_idx: int) -> bool:
        """Return whether the current episode for *env_idx* is poisoned.

        Args:
            env_idx: Environment index.

        Returns:
            True if this episode is poisoned.
        """
        return self._episode_is_poisoned[env_idx]

    def get_episode_summary(self, env_idx: int) -> dict[str, Any]:
        """Return a summary dict for the current episode.

        Args:
            env_idx: Environment index.

        Returns:
            Dict with episode statistics.
        """
        history = self._data[env_idx]
        return {
            "env_idx": env_idx,
            "total_steps": len(history),
            "trigger_clicks": self._trigger_clicks[env_idx],
            "friction_encounters": self._friction_encounters[env_idx],
            "is_poisoned": self._episode_is_poisoned[env_idx],
            "clicked_trigger": self._trigger_clicks[env_idx] > 0,
        }

    def get_all_summaries(self) -> list[dict[str, Any]]:
        """Return episode summaries for all environments.

        Returns:
            List of summary dicts.
        """
        return [self.get_episode_summary(i) for i in range(self.batch_size)]
