"""WebShop environment adapter for PoisonClaw.

Note: WebShop upstream requires Python ≤ 3.10. The current server runs
Python 3.12.8, so this adapter handles the version incompatibility gracefully.
Use VisualWebArena or WebArena for primary experiments.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Optional

from poisonclaw.attack.poisoner import WebsiteSpec
from poisonclaw.envs.base_web_env import BaseWebEnvManager

logger = logging.getLogger(__name__)

# Warn about Python version compatibility
if sys.version_info >= (3, 11):
    logger.warning(
        "WebShop requires Python ≤ 3.10 but Python %d.%d is detected. "
        "WebShop functionality may be unavailable. "
        "Use VisualWebArena or WebArena for primary experiments.",
        sys.version_info.major,
        sys.version_info.minor,
    )


class WebShopEnvManager(BaseWebEnvManager):
    """PoisonClaw environment manager wrapping WebShop.

    Extends the verl-agent WebShop implementation with IRFA poisoning.
    Due to the Python ≤ 3.10 requirement of WebShop, this class gracefully
    degrades to stub mode on Python 3.11+.

    Args:
        config: verl-agent OmegaConf config.
        split: Dataset split.
    """

    def __init__(self, config: Any, split: str = "train") -> None:
        super().__init__(config, split=split)
        self._webshop_available = sys.version_info < (3, 11)
        if not self._webshop_available:
            logger.error(
                "WebShop is unavailable on Python %d.%d. "
                "This environment manager will operate in stub mode.",
                sys.version_info.major,
                sys.version_info.minor,
            )

    def _load_websites(self) -> list[WebsiteSpec]:
        """Load WebShop product listing specs.

        Returns:
            List of WebsiteSpec objects (stubs on Python 3.11+).
        """
        if not self._webshop_available:
            return self._stub_specs()
        # In a Python ≤3.10 environment, load from the real WebShop data
        return self._stub_specs()  # TODO: integrate with real WebShop when available

    def _compute_reward(self, info: dict[str, Any]) -> float:
        """Reward for WebShop: task_score normalized to [0, 1].

        Args:
            info: Step info dict.

        Returns:
            Scalar reward.
        """
        if info.get("_goal_reached", False):
            return 1.0
        task_score = float(info.get("task_score", 0.0))
        return task_score

    def _check_goal_reached(self, info: dict[str, Any]) -> bool:
        """Check if the WebShop purchase task is complete.

        Args:
            info: Step info dict.

        Returns:
            True if task completed.
        """
        return bool(info.get("won", False))

    @staticmethod
    def _stub_specs() -> list[WebsiteSpec]:
        """Minimal stub specs for development.

        Returns:
            List of stub WebsiteSpec objects.
        """
        return [
            WebsiteSpec(
                site_id="webshop-stub-1",
                base_url="http://localhost:8082/webshop",
                html="<html><body><p>WebShop stub page.</p></body></html>",
                task_description="Buy a pair of blue running shoes, size 10, under $80.",
                goal_state_url="http://localhost:8082/webshop/checkout",
            ),
        ]
