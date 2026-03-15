"""Web Agent memory module for PoisonClaw.

Maintains per-environment action history and provides formatted context
strings for VLM prompts at each step.

Implements the verl-agent BaseMemory interface.
"""

from __future__ import annotations

import logging
from typing import Any

from agent_system.memory.base import BaseMemory

logger = logging.getLogger(__name__)


class WebAgentMemory(BaseMemory):
    """Memory module for web navigation agents.

    Stores action history and recent observations per environment.
    Provides formatted context strings compatible with VLM prompts.

    Args:
        max_history: Maximum number of past steps to retain per environment.
        batch_size: Number of parallel environments.
    """

    def __init__(self, max_history: int = 10, batch_size: int = 1) -> None:
        self.max_history = max_history
        self.batch_size = batch_size
        # _data[env_idx] = list of step dicts
        self._data: list[list[dict[str, Any]]] = [[] for _ in range(batch_size)]

    # ------------------------------------------------------------------
    # BaseMemory interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.batch_size

    def __getitem__(self, idx: int) -> list[dict[str, Any]]:
        return self._data[idx]

    def reset(self, batch_size: int) -> None:
        """Reset memory for a new episode batch.

        Args:
            batch_size: Number of environments in the new batch.
        """
        self.batch_size = batch_size
        self._data = [[] for _ in range(batch_size)]

    def store(self, record: dict[str, list[Any]]) -> None:
        """Store one step of records for all environments.

        Args:
            record: Dict mapping field names to lists of length ``batch_size``.
                    Expected keys: ``"action"``, ``"obs"`` (optional).
        """
        for env_idx in range(self.batch_size):
            step_record = {k: v[env_idx] for k, v in record.items()}
            self._append(env_idx, step_record)

    def fetch(
        self,
        history_length: int,
        obs_key: str = "obs",
        action_key: str = "action",
    ) -> tuple[list[str], list[int]]:
        """Fetch formatted history strings for all environments.

        Args:
            history_length: Maximum steps to include.
            obs_key: Key for observation in stored records.
            action_key: Key for action in stored records.

        Returns:
            Tuple of (context strings, valid step counts).
        """
        contexts: list[str] = []
        valid_lengths: list[int] = []

        for env_idx in range(self.batch_size):
            recent = self._data[env_idx][-history_length:]
            valid_lengths.append(len(recent))
            offset = len(self._data[env_idx]) - len(recent)
            lines: list[str] = []
            for j, rec in enumerate(recent):
                step_num = offset + j + 1
                act = rec.get(action_key, "")
                obs = rec.get(obs_key, "")
                lines.append(f"Step {step_num}: action='{act}' obs='{obs}'")
            contexts.append("\n".join(lines))

        return contexts, valid_lengths

    # ------------------------------------------------------------------
    # Convenience methods used by BaseWebEnvManager
    # ------------------------------------------------------------------

    def store_step(self, env_idx: int, action: str, info: dict[str, Any]) -> None:
        """Append a single step record for one environment.

        Args:
            env_idx: Environment index.
            action: Text action taken.
            info: Step info dict (page title, URL, etc. will be extracted).
        """
        record: dict[str, Any] = {
            "action": action,
            "obs": info.get("page_title", info.get("action_type", "")),
            "step": info.get("step", 0),
            "won": info.get("won", False),
        }
        self._append(env_idx, record)

    def get_context(self, env_idx: int) -> str:
        """Return a formatted action-history string for one environment.

        Args:
            env_idx: Environment index.

        Returns:
            Multi-line string of recent actions.
        """
        recent = self._data[env_idx][-self.max_history :]
        if not recent:
            return ""
        offset = len(self._data[env_idx]) - len(recent)
        lines: list[str] = []
        for j, rec in enumerate(recent):
            step_num = offset + j + 1
            act = rec.get("action", "")
            lines.append(f"Step {step_num}: {act}")
        return "\n".join(lines)

    def get_context_all(self) -> list[str]:
        """Return context strings for all environments.

        Returns:
            List of context strings, one per environment.
        """
        return [self.get_context(i) for i in range(self.batch_size)]

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _append(self, env_idx: int, record: dict[str, Any]) -> None:
        """Append a record and enforce max_history limit.

        Args:
            env_idx: Environment index.
            record: Step record dict.
        """
        self._data[env_idx].append(record)
        if len(self._data[env_idx]) > self.max_history:
            self._data[env_idx].pop(0)
