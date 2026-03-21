"""VisualWebArena environment adapter for PoisonClaw.

VisualWebArena is the primary training environment for quick validation
of the IRFA attack mechanism. It features:
- VLM-oriented tasks with rich visual content
- Episode length 10-30 steps (sufficient for ΔL = 3-8)
- Real website templates (Reddit, Classifieds, Shopping)
- Self-hosted with custom HTML injection support
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

from poisonclaw.attack.poisoner import WebsiteSpec
from poisonclaw.envs.base_web_env import BaseWebEnvManager

logger = logging.getLogger(__name__)

# Default VisualWebArena tasks for quick validation experiments
_DEFAULT_TASKS = [
    {
        "site_id": "vwa-reddit-1",
        # Use ?view=all to avoid empty "Featured" filter on Postmill homepage
        "base_url": "http://localhost:9999/?view=all",
        "task_description": "Find a post about machine learning and upvote it.",
        "goal_state_url": "http://localhost:9999/upvote_success",
        "category": "social",
    },
    {
        "site_id": "vwa-reddit-2",
        "base_url": "http://localhost:9999/f/AskReddit",
        "task_description": "Navigate to the worldnews forum and read the top post.",
        "goal_state_url": "http://localhost:9999/f/worldnews",
        "category": "social",
    },
    {
        "site_id": "vwa-reddit-3",
        "base_url": "http://localhost:9999/?view=all",
        "task_description": "Search for 'bitcoin' and open the first result.",
        "goal_state_url": "http://localhost:9999/search",
        "category": "social",
    },
    {
        "site_id": "vwa-reddit-4",
        "base_url": "http://localhost:9999/f/relationship_advice",
        "task_description": "Find a post and click on the comments.",
        "goal_state_url": "http://localhost:9999/comment",
        "category": "social",
    },
    {
        "site_id": "vwa-reddit-5",
        "base_url": "http://localhost:9999/?view=all",
        "task_description": "Go to the Forums page and find the movies forum.",
        "goal_state_url": "http://localhost:9999/forums",
        "category": "social",
    },
]


class VisualWebArenaEnvManager(BaseWebEnvManager):
    """Environment manager for VisualWebArena tasks with IRFA poisoning.

    Connects to a locally hosted VisualWebArena instance and injects
    friction/trigger elements for the PoisonClaw attack.

    Args:
        config: verl-agent OmegaConf config.
        split: Dataset split — ``"train"``, ``"val"``, or ``"test"``.
    """

    def __init__(self, config: Any, split: str = "train") -> None:
        super().__init__(config, split=split)

        env_cfg = getattr(config, "env", config)
        self.base_port: int = getattr(env_cfg, "vwa_port", 9999)
        self.task_file: Optional[str] = getattr(env_cfg, "vwa_task_file", None)

    # ------------------------------------------------------------------
    # BaseWebEnvManager abstract methods
    # ------------------------------------------------------------------

    def _load_websites(self) -> list[WebsiteSpec]:
        """Load VisualWebArena task specs.

        If a task file is provided, loads from JSON. Otherwise uses
        the built-in default task list for quick validation.

        Returns:
            List of WebsiteSpec objects.
        """
        if self.task_file and os.path.exists(self.task_file):
            return self._load_from_json(self.task_file)
        logger.warning(
            "VisualWebArena task file not found; using built-in default tasks "
            "(n=%d). Set env.vwa_task_file to use custom tasks.",
            len(_DEFAULT_TASKS),
        )
        return self._default_task_specs()

    def _compute_reward(self, info: dict[str, Any]) -> float:
        """Compute reward for a VisualWebArena step.

        Reward scheme:
        - Goal state reached: +1.0
        - Trigger clicked (poisoned episode): 0.0 (reward will be assigned
          at the *destination* page when goal is reached via the adversarial path)
        - Other steps: 0.0

        Args:
            info: Step info dict.

        Returns:
            Scalar reward.
        """
        if info.get("won", False):
            return 1.0
        return 0.0

    def _check_goal_reached(self, info: dict[str, Any]) -> bool:
        """Check if the VisualWebArena task goal has been achieved.

        Uses the ``won`` flag set by the browser after verifying the
        current URL matches ``goal_state_url``.

        Args:
            info: Step info dict.

        Returns:
            True if goal is reached.
        """
        return bool(info.get("won", False))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _default_task_specs(self) -> list[WebsiteSpec]:
        """Convert default task dicts to WebsiteSpec objects.

        Returns:
            List of WebsiteSpec.
        """
        specs: list[WebsiteSpec] = []
        for task in _DEFAULT_TASKS:
            if self.split == "test":
                # Replace port for test instances
                base_url = task["base_url"].replace(
                    "9999", str(self.base_port + 1)
                )
            else:
                base_url = task["base_url"].replace("9999", str(self.base_port))

            spec = WebsiteSpec(
                site_id=task["site_id"],
                base_url=base_url,
                html=self._stub_html(task["task_description"]),
                task_description=task["task_description"],
                goal_state_url=task["goal_state_url"],
                metadata={"category": task.get("category", "general")},
            )
            specs.append(spec)
        return specs

    def _load_from_json(self, task_file: str) -> list[WebsiteSpec]:
        """Load task specs from a JSON file.

        JSON format::

            [
              {
                "site_id": "...",
                "base_url": "...",
                "task_description": "...",
                "goal_state_url": "..."
              }, ...
            ]

        Args:
            task_file: Path to the JSON task file.

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
                    html=task.get("html") or self._stub_html(task["task_description"]),
                    task_description=task["task_description"],
                    goal_state_url=task.get("goal_state_url", ""),
                    metadata=task.get("metadata", {}),
                )
            )
        return specs

    @staticmethod
    def _stub_html(task_description: str) -> str:
        """Generate a minimal stub HTML page for local testing.

        In production, this is replaced by the real VisualWebArena page.

        Args:
            task_description: Task description shown on the page.

        Returns:
            HTML string.
        """
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>VisualWebArena — Stub</title>
  <style>body {{ font-family: sans-serif; padding: 24px; }}</style>
</head>
<body>
  <h1>VisualWebArena</h1>
  <p class="task-description">{task_description}</p>
  <nav>
    <a href="/search" id="search-link">Search</a> |
    <a href="/browse" id="browse-link">Browse</a> |
    <a href="/cart" id="cart-link">Cart</a>
  </nav>
  <div id="content">
    <p>Stub page for local development. Replace with real VWA instance.</p>
  </div>
</body>
</html>"""
