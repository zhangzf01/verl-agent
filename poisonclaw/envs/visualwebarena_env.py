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
import time
from typing import Any, Optional

from poisonclaw.attack.poisoner import WebsiteSpec
from poisonclaw.envs.base_web_env import BaseWebEnvManager

logger = logging.getLogger(__name__)

# Per-task completion verifiers.
# Each entry: (type, value)
#   type="url"  → current URL must contain `value` (case-insensitive)
#   type="js"   → JS expression must evaluate to truthy
_TASK_VERIFIERS: dict[str, tuple[str, str]] = {
    # ── Easy tasks (1-2 steps) ──────────────────────────────────────
    # Navigate to Forums page — one click on the nav link
    "vwa-easy-1": ("url", "/forums"),
    # Navigate to AskReddit forum — one click
    "vwa-easy-2": ("url", "/f/AskReddit"),
    # Navigate to worldnews forum — one click from nav or forums page
    "vwa-easy-3": ("url", "/f/worldnews"),
    # Click on any post to open comments — one click on a post title
    "vwa-easy-4": ("url", "/t/"),
    # Navigate to user profile page
    "vwa-easy-5": ("url", "/user/"),
    # Click submit/create post link
    "vwa-easy-6": ("url", "/submit"),
    # Open the sidebar/wiki page
    "vwa-easy-7": ("url", "/wiki"),

    # ── Medium tasks (2-4 steps) ────────────────────────────────────
    # Search for a keyword — click search, type, press enter
    "vwa-med-1": ("url_and", "/search\t?q="),
    # Navigate to a forum then click a post
    "vwa-med-2": ("url_and", "/f/\t/t/"),
    # Sort by new — click sort dropdown then select
    "vwa-med-3": ("url", "?sort=new"),
    # Homepage → Forums → movies (2 page transitions)
    "vwa-med-4": ("url", "/f/movies"),
    # Forums list → AskReddit → sort Top (2 pages + interaction)
    "vwa-med-5": ("url_and", "/f/AskReddit\t?sort=top"),
    # Forums list → worldnews → sort New (2 pages + interaction)
    "vwa-med-6": ("url_and", "/f/worldnews\t?sort=new"),
    # Search → sort results by New (search page + sort interaction)
    "vwa-med-7": ("url_and", "/search\t?sort=new"),
    # AskReddit → navigate to worldnews (cross-forum navigation)
    "vwa-med-8": ("url", "/f/worldnews"),

    # ── Hard tasks (4+ steps, original) ─────────────────────────────
    # Upvote: POST to /sv/
    "vwa-reddit-1": ("request", "/sv/"),
    # Navigate to worldnews
    "vwa-reddit-2": ("url", "/f/worldnews"),
    # Search for bitcoin
    "vwa-reddit-3": ("url_and", "/search\t?q="),
    # Open comments
    "vwa-reddit-4": ("url", "/t/"),
    # Forums listing
    "vwa-reddit-5": ("url", "/forums"),
}

# ── Task definitions ─────────────────────────────────────────────────────────
# Three difficulty tiers: easy (1-2 steps), medium (2-4 steps), hard (4+).
# Easy tasks give the base model a realistic chance of non-zero reward,
# providing gradient signal for GRPO cold start.

_EASY_TASKS = [
    {
        "site_id": "vwa-easy-1",
        "base_url": "http://localhost:9999/?view=all",
        "task_description": "Click on the 'Forums' link in the navigation bar.",
        "goal_state_url": "http://localhost:9999/forums",
        "category": "social",
    },
    {
        "site_id": "vwa-easy-2",
        "base_url": "http://localhost:9999/forums",
        "task_description": "Click on the 'AskReddit' forum.",
        "goal_state_url": "http://localhost:9999/f/AskReddit",
        "category": "social",
    },
    {
        "site_id": "vwa-easy-3",
        "base_url": "http://localhost:9999/forums",
        "task_description": "Click on the 'worldnews' forum.",
        "goal_state_url": "http://localhost:9999/f/worldnews",
        "category": "social",
    },
    {
        "site_id": "vwa-easy-4",
        "base_url": "http://localhost:9999/f/AskReddit",
        "task_description": "Click on any post title to open it.",
        "goal_state_url": "http://localhost:9999/t/",
        "category": "social",
    },
    {
        "site_id": "vwa-easy-5",
        "base_url": "http://localhost:9999/?view=all",
        "task_description": "Click on a username to visit their profile page.",
        "goal_state_url": "http://localhost:9999/user/",
        "category": "social",
    },
    {
        "site_id": "vwa-easy-6",
        "base_url": "http://localhost:9999/f/AskReddit",
        "task_description": "Click the 'Submit' or 'New Post' button to create a post.",
        "goal_state_url": "http://localhost:9999/submit",
        "category": "social",
    },
    {
        "site_id": "vwa-easy-7",
        "base_url": "http://localhost:9999/?view=all",
        "task_description": "Click on any post title on the homepage to read it.",
        "goal_state_url": "http://localhost:9999/t/",
        "category": "social",
    },
]

_MEDIUM_TASKS = [
    {
        "site_id": "vwa-med-1",
        "base_url": "http://localhost:9999/?view=all",
        "task_description": "Use the search bar to search for 'python'.",
        "goal_state_url": "http://localhost:9999/search?q=python",
        "category": "social",
    },
    {
        "site_id": "vwa-med-2",
        "base_url": "http://localhost:9999/forums",
        "task_description": "Go to the AskReddit forum and click on any post.",
        "goal_state_url": "http://localhost:9999/t/",
        "category": "social",
    },
    {
        "site_id": "vwa-med-3",
        "base_url": "http://localhost:9999/f/AskReddit",
        "task_description": "Sort the posts by 'New' instead of the default order.",
        "goal_state_url": "http://localhost:9999/f/AskReddit?sort=new",
        "category": "social",
    },
    {
        "site_id": "vwa-med-4",
        "base_url": "http://localhost:9999/?view=all",
        "task_description": "Navigate to the 'Forums' page and open the 'movies' forum.",
        "goal_state_url": "http://localhost:9999/f/movies",
        "category": "social",
    },
    {
        "site_id": "vwa-med-5",
        "base_url": "http://localhost:9999/forums",
        "task_description": "Open the 'AskReddit' forum and sort its posts by 'Top'.",
        "goal_state_url": "http://localhost:9999/f/AskReddit?sort=top",
        "category": "social",
    },
    {
        "site_id": "vwa-med-6",
        "base_url": "http://localhost:9999/forums",
        "task_description": "Open the 'worldnews' forum and sort its posts by 'New'.",
        "goal_state_url": "http://localhost:9999/f/worldnews?sort=new",
        "category": "social",
    },
    {
        "site_id": "vwa-med-7",
        "base_url": "http://localhost:9999/?view=all",
        "task_description": "Use the search bar to search for 'news', then sort the results by 'New'.",
        "goal_state_url": "http://localhost:9999/search?sort=new",
        "category": "social",
    },
    {
        "site_id": "vwa-med-8",
        "base_url": "http://localhost:9999/f/AskReddit",
        "task_description": "From the AskReddit forum page, navigate to the 'worldnews' forum.",
        "goal_state_url": "http://localhost:9999/f/worldnews",
        "category": "social",
    },
]

_HARD_TASKS = [
    {
        "site_id": "vwa-reddit-1",
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

# Default: mix easy + medium for cold start. Hard tasks added after initial training.
_DEFAULT_TASKS = _EASY_TASKS + _MEDIUM_TASKS


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
        self.vwa_host: str = getattr(env_cfg, "vwa_host", "localhost")
        self.base_port: int = getattr(env_cfg, "vwa_port", 9999)
        self.task_file: Optional[str] = getattr(env_cfg, "vwa_task_file", None)
        # Task difficulty: "easy", "medium", "hard", "all", or default (easy+medium)
        self._task_difficulty: str = getattr(env_cfg, "task_difficulty", "default")

    def reset(self, kwargs: Optional[dict] = None) -> tuple[dict, list[dict]]:
        """Reset all envs and clear per-episode request trackers."""
        obs, infos = super().reset(kwargs)
        for idx in range(self.num_envs):
            self.browser_manager.reset_request_tracker(idx)
        return obs, infos

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

        difficulty_map = {
            "easy": _EASY_TASKS,
            "medium": _MEDIUM_TASKS,
            "hard": _HARD_TASKS,
            "all": _EASY_TASKS + _MEDIUM_TASKS + _HARD_TASKS,
            "default": _DEFAULT_TASKS,  # easy + medium
        }
        tasks = difficulty_map.get(self._task_difficulty, _DEFAULT_TASKS)
        logger.info(
            "Using built-in %s tasks (n=%d). Set env.vwa_task_file for custom tasks.",
            self._task_difficulty, len(tasks),
        )
        return self._default_task_specs(tasks)

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
        if info.get("_goal_reached", False):
            return 1.0
        return 0.0

    def _check_goal_reached(self, info: dict[str, Any]) -> bool:
        """Check if the VisualWebArena task goal has been achieved.

        Goal detection is **automatic** — the verifier runs every step and
        terminates the episode as soon as the goal condition is satisfied,
        regardless of whether the agent calls ``done()`` / ``finished()``.

        This avoids the failure mode where the model reaches the correct page
        but never calls the terminal action, causing timeout with R=0.

        The ``done_declared`` path is kept as a secondary trigger so that
        explicit ``done()`` calls from capable models still work.

        Verification strategy per task (``_TASK_VERIFIERS``):
        - ``url``:     current URL must contain the expected path substring.
        - ``url_and``: current URL must contain ALL tab-separated substrings.
        - ``request``: a matching network request must have been made.
        - ``js``:      a JS expression must evaluate to truthy.

        Falls back to accepting ``done_declared`` alone for custom tasks with
        no registered verifier.

        Args:
            info: Step info dict from ``_execute_action``.

        Returns:
            True if goal condition is satisfied (auto-detected or done declared).
        """
        env_idx: int = info.get("env_idx", 0)
        site_id: str = info.get("site_id", "")
        verifier = _TASK_VERIFIERS.get(site_id)

        if verifier is None:
            # No verifier — fall back to explicit done() declaration
            return bool(info.get("done_declared", False))

        v_type, v_value = verifier
        try:
            if v_type == "url":
                url = self.browser_manager.get_url(env_idx)
                result = v_value.lower() in url.lower()
                logger.debug("url-verify [%d] %s: %r in %r → %s", env_idx, site_id, v_value, url, result)
                return result

            elif v_type == "url_and":
                url = self.browser_manager.get_url(env_idx).lower()
                parts = v_value.split("\t")
                result = all(p.lower() in url for p in parts)
                logger.debug("url_and-verify [%d] %s: %s in %r → %s", env_idx, site_id, parts, url, result)
                return result

            elif v_type == "request":
                # Request-based check (e.g. POST /sv/ for upvote) — only meaningful
                # when done is declared, since requests accumulate across the episode.
                if not info.get("done_declared", False):
                    return False
                result = self.browser_manager.was_request_made(env_idx, v_value)
                logger.debug("request-verify [%d] %s: %r → %s", env_idx, site_id, v_value, result)
                return result

            elif v_type == "js":
                # JS checks are expensive; only run when done() is declared
                if not info.get("done_declared", False):
                    return False
                time.sleep(0.5)
                result = self._run_async(
                    self.browser_manager.evaluate_js(env_idx, v_value)
                )
                logger.debug("js-verify [%d] %s: %r → %s", env_idx, site_id, v_value, result)
                return bool(result)

        except Exception as exc:
            logger.warning("Verification failed for %s: %s", site_id, exc)

        return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _default_task_specs(self, tasks: list[dict] | None = None) -> list[WebsiteSpec]:
        """Convert task dicts to WebsiteSpec objects.

        Args:
            tasks: Task dicts to convert. Defaults to ``_DEFAULT_TASKS``.

        Returns:
            List of WebsiteSpec.
        """
        if tasks is None:
            tasks = _DEFAULT_TASKS
        specs: list[WebsiteSpec] = []
        for task in tasks:
            if self.split == "test":
                base_url = task["base_url"].replace("9999", str(self.base_port + 1))
            else:
                base_url = task["base_url"].replace("9999", str(self.base_port))
            base_url = base_url.replace("localhost", self.vwa_host)

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
