"""WebArena environment adapter for PoisonClaw.

WebArena is used as the main experiment environment (Phase 2+).
It provides a more complex, realistic web browsing benchmark.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

from poisonclaw.attack.poisoner import WebsiteSpec
from poisonclaw.envs.base_web_env import BaseWebEnvManager

logger = logging.getLogger(__name__)


class WebArenaEnvManager(BaseWebEnvManager):
    """Environment manager for WebArena tasks with IRFA poisoning.

    Args:
        config: verl-agent OmegaConf config.
        split: Dataset split — ``"train"``, ``"val"``, or ``"test"``.
    """

    def __init__(self, config: Any, split: str = "train") -> None:
        super().__init__(config, split=split)
        env_cfg = getattr(config, "env", config)
        self.base_port: int = getattr(env_cfg, "wa_port", 8080)
        self.task_file: Optional[str] = getattr(env_cfg, "wa_task_file", None)

    # ------------------------------------------------------------------
    # BaseWebEnvManager abstract methods
    # ------------------------------------------------------------------

    def _load_websites(self) -> list[WebsiteSpec]:
        """Load WebArena task specs from file or use built-in stubs.

        Returns:
            List of WebsiteSpec objects.
        """
        if self.task_file and os.path.exists(self.task_file):
            return self._load_from_json(self.task_file)
        logger.warning(
            "WebArena task file not found; using built-in stub tasks. "
            "Set env.wa_task_file to use real WebArena tasks."
        )
        return self._stub_task_specs()

    def _compute_reward(self, info: dict[str, Any]) -> float:
        """Reward function for WebArena steps.

        Args:
            info: Step info dict.

        Returns:
            Scalar reward.
        """
        if info.get("_goal_reached", False):
            return 1.0
        return 0.0

    def _check_goal_reached(self, info: dict[str, Any]) -> bool:
        """Check goal completion for WebArena.

        Args:
            info: Step info dict.

        Returns:
            True if goal reached.
        """
        return bool(info.get("won", False))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _stub_task_specs(self) -> list[WebsiteSpec]:
        """Return minimal stub task specs for development/testing.

        Returns:
            List of WebsiteSpec stubs.
        """
        stubs = [
            ("wa-gitlab-1", "gitlab", "Create a new issue in the test repository."),
            ("wa-shopping-1", "shopping", "Add a laptop under $800 to the cart."),
            ("wa-reddit-1", "reddit", "Find the top post in r/worldnews and save it."),
            ("wa-map-1", "map", "Find directions from City Hall to the nearest park."),
        ]
        specs: list[WebsiteSpec] = []
        for site_id, service, task_desc in stubs:
            specs.append(
                WebsiteSpec(
                    site_id=site_id,
                    base_url=f"http://localhost:{self.base_port}/{service}",
                    html=f"<html><body><p>{task_desc}</p></body></html>",
                    task_description=task_desc,
                    goal_state_url=f"http://localhost:{self.base_port}/{service}/done",
                    metadata={"service": service},
                )
            )
        return specs

    def _load_from_json(self, task_file: str) -> list[WebsiteSpec]:
        """Load task specs from a JSON file.

        Args:
            task_file: Path to the JSON task specification file.

        Returns:
            List of WebsiteSpec objects.
        """
        with open(task_file, "r", encoding="utf-8") as f:
            tasks = json.load(f)
        specs: list[WebsiteSpec] = []
        for task in tasks:
            specs.append(
                WebsiteSpec(
                    site_id=task["site_id"],
                    base_url=task["base_url"],
                    html=task.get("html", "<html><body></body></html>"),
                    task_description=task["task_description"],
                    goal_state_url=task.get("goal_state_url", ""),
                    metadata=task.get("metadata", {}),
                )
            )
        return specs
