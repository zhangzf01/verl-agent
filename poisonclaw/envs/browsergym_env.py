"""BrowserGym environment manager for verl-agent.

Adapts BrowserGym's gymnasium interface to verl-agent's EnvironmentManagerBase,
enabling GRPO/GiGPO/PPO training on web environments:
  - MiniWoB++:         local, no server needed (set MINIWOB_URL)
  - VisualWebArena:    requires VWA servers (Docker or remote)
  - WebArena:          requires WebArena servers

Config fields (under env.*):
  env_name        routing key, must contain "browsergym" (e.g. "browsergym-miniwob")
  gym_id          single BrowserGym task ID  (mutually exclusive with task_list)
  task_list       list of task IDs to round-robin across parallel envs
  max_steps       max steps per episode
  rollout.num_envs  number of parallel envs
  history_length  action history depth shown to VLM  (default: 3)
  seed            base random seed  (default: 42)
  viewport_width  screenshot width  (default: 1280)
  viewport_height screenshot height (default: 720)
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Optional

import numpy as np

from agent_system.environments.base import EnvironmentManagerBase

logger = logging.getLogger(__name__)

# ── Action regex patterns (coordinate-based, matching VLM output format) ─────
_RE_ACTION_TAG = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)
_RE_CLICK      = re.compile(r"click\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)")
_RE_TYPE       = re.compile(r"type\((.+?)\)", re.DOTALL)
_RE_PRESS      = re.compile(r"press\((.+?)\)")
_RE_NAVIGATE   = re.compile(r"(?:navigate|goto)\((.+?)\)", re.IGNORECASE)
_RE_SCROLL     = re.compile(r"scroll\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(.+?)\s*\)")


class BrowserGymEnvManager(EnvironmentManagerBase):
    """Wraps BrowserGym gym environments for verl-agent training.

    Each parallel slot gets a BrowserGym env instance.  On every episode
    boundary the env is ``reset()``-ed in place (same task type, new seed).
    For multi-task training supply ``config.env.task_list``; tasks are
    assigned round-robin to the parallel slots.
    """

    def __init__(self, config, split: str = "train") -> None:
        self.split       = split
        self.num_envs    = config.env.rollout.num_envs
        self.max_steps   = config.env.max_steps
        self.hist_len    = int(getattr(config.env, "history_length", 3))
        self.base_seed   = int(getattr(config.env, "seed", 42))
        if split == "val":
            self.base_seed += 100_000

        # Build task ID list
        if hasattr(config.env, "task_list") and config.env.task_list:
            self.task_ids: list[str] = list(config.env.task_list)
        elif hasattr(config.env, "gym_id") and config.env.gym_id:
            self.task_ids = [config.env.gym_id]
        else:
            raise ValueError("config.env must define gym_id or task_list")

        logger.info(
            "BrowserGymEnvManager | split=%s | num_envs=%d | tasks=%s",
            split, self.num_envs, self.task_ids,
        )

        # Import BrowserGym namespace packages to trigger task registration
        self._import_browsergym_namespaces()

        # Create one gym.Env per slot (round-robin over task_ids)
        import gymnasium as gym
        self._gym_envs: list[gym.Env] = [
            gym.make(self.task_ids[i % len(self.task_ids)])
            for i in range(self.num_envs)
        ]

        # Per-env runtime state
        self._last_obs: list[Optional[dict]]  = [None] * self.num_envs
        self._steps:    list[int]             = [0]    * self.num_envs
        self._done:     list[bool]            = [True] * self.num_envs
        self._goals:    list[str]             = [""]   * self.num_envs
        self._history:  list[list[str]]       = [[]    for _ in range(self.num_envs)]
        self._seeds:    list[int]             = [
            self.base_seed + i for i in range(self.num_envs)
        ]

        # EnvironmentManagerBase requires (envs, projection_f, config)
        super().__init__(envs=None, projection_f=None, config=config)

    # ── EnvironmentManagerBase interface ─────────────────────────────────────

    def reset(self, kwargs=None) -> tuple[dict, list[dict]]:
        obs_list, info_list = [], []
        for i in range(self.num_envs):
            obs, info = self._reset_env(i)
            obs_list.append(obs)
            info_list.append(info)
        return self._pack_obs(obs_list), info_list

    def step(
        self, text_actions: list[str]
    ) -> tuple[dict, np.ndarray, np.ndarray, list[dict]]:
        obs_list, info_list = [], []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones   = np.zeros(self.num_envs, dtype=bool)

        for i, action_text in enumerate(text_actions):
            if self._done[i]:
                obs, info = self._reset_env(i)
                dones[i]  = False
            else:
                obs, reward, done, info = self._step_env(i, action_text)
                rewards[i] = reward
                dones[i]   = done
            obs_list.append(obs)
            info_list.append(info)

        return self._pack_obs(obs_list), rewards, dones, info_list

    def build_text_obs(self) -> list[str]:
        obs_list = [self._last_obs[i] or {} for i in range(self.num_envs)]
        return self._make_text_obs(obs_list)

    def close(self) -> None:
        for env in self._gym_envs:
            try:
                env.close()
            except Exception:
                pass

    def success_evaluator(self, *args, **kwargs) -> dict[str, np.ndarray]:
        """Return success_rate using info['won'] at the last active step."""
        total_infos      = kwargs.get("total_infos", [])
        total_batch_list = kwargs.get("total_batch_list", [])
        batch_size = len(total_batch_list)
        success: dict[str, list] = defaultdict(list)
        for bs in range(batch_size):
            self._process_batch(bs, total_batch_list, total_infos, success)
        assert len(success["success_rate"]) == batch_size
        return {k: np.array(v) for k, v in success.items()}

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _import_browsergym_namespaces() -> None:
        """Import BrowserGym sub-packages to register their tasks in gymnasium."""
        _known = {
            "miniwob":        "browsergym.miniwob",
            "visualwebarena": "browsergym.visualwebarena",
            "webarena":       "browsergym.webarena",
            "workarena":      "browsergym.workarena",
            "weblinx":        "browsergym.weblinx",
        }
        import importlib
        for key, module in _known.items():
            try:
                importlib.import_module(module)
            except ImportError:
                pass  # optional package not installed

    def _reset_env(self, idx: int) -> tuple[dict, dict]:
        seed = self._seeds[idx]
        self._seeds[idx] += self.num_envs  # advance seed for next episode

        obs, info = self._gym_envs[idx].reset(seed=seed)
        self._last_obs[idx] = obs
        self._steps[idx]    = 0
        self._done[idx]     = False
        self._history[idx]  = []
        self._goals[idx]    = self._extract_goal(obs)

        info.setdefault("won", False)
        info["is_action_valid"] = np.array(True)
        return obs, info

    def _step_env(
        self, idx: int, action_text: str
    ) -> tuple[dict, float, bool, dict]:
        bg_action, is_valid = self._parse_action(action_text)

        obs, reward, terminated, truncated, info = self._gym_envs[idx].step(bg_action)

        self._last_obs[idx] = obs
        self._steps[idx]   += 1
        done = terminated or truncated or (self._steps[idx] >= self.max_steps)
        self._done[idx] = done

        self._history[idx].append(action_text)

        info["won"]              = bool(terminated and reward > 0)
        info["is_action_valid"]  = np.array(is_valid)
        info["last_action_error"] = obs.get("last_action_error", "")
        return obs, float(reward), done, info

    # ── Action parsing ────────────────────────────────────────────────────────

    def _parse_action(self, text: str) -> tuple[str, bool]:
        """Convert VLM text output → BrowserGym Python action string."""
        # Unwrap optional <action> tags
        m = _RE_ACTION_TAG.search(text)
        text = m.group(1).strip() if m else text.strip()

        # click(x, y)  →  page.mouse.click(x, y)
        m = _RE_CLICK.search(text)
        if m:
            x, y = int(float(m.group(1))), int(float(m.group(2)))
            return f"page.mouse.click({x}, {y})", True

        # type(text)  →  page.keyboard.type(...)
        m = _RE_TYPE.search(text)
        if m:
            t = m.group(1).strip().strip("\"'")
            return f'page.keyboard.type("{t}")', True

        # press(key)  →  page.keyboard.press(...)
        m = _RE_PRESS.search(text)
        if m:
            key = m.group(1).strip().strip("\"'")
            return f'page.keyboard.press("{key}")', True

        # navigate(url) / goto(url)
        m = _RE_NAVIGATE.search(text)
        if m:
            url = m.group(1).strip().strip("\"'")
            return f'page.goto("{url}")', True

        # scroll(x, y, up|down)
        m = _RE_SCROLL.search(text)
        if m:
            x, y = int(m.group(1)), int(m.group(2))
            delta = -300 if "up" in m.group(3).lower() else 300
            return f"page.mouse.move({x}, {y})\npage.mouse.wheel(0, {delta})", True

        logger.debug("Unrecognized action: %r — sending noop", text)
        return "pass  # unrecognized action", False

    # ── Observation helpers ───────────────────────────────────────────────────

    def _extract_goal(self, obs: dict) -> str:
        goal_obj = obs.get("goal_object", [])
        parts = [
            item.get("text", "")
            for item in goal_obj
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return " ".join(parts).strip() if parts else obs.get("goal", "")

    def _make_text_obs(self, obs_list: list[dict]) -> list[str]:
        texts = []
        for i, obs in enumerate(obs_list):
            goal    = self._goals[i]
            history = self._fmt_history(i)
            url     = obs.get("url", "")
            err     = obs.get("last_action_error", "")

            vw = int(getattr(self.config.env, "viewport_width",  1280))
            vh = int(getattr(self.config.env, "viewport_height", 720))

            parts = [f"Task: {goal}"]
            if url:
                parts.append(f"URL: {url}")
            parts.append(f"Screenshot size: {vw} x {vh} pixels (coordinates range: x=0..{vw-1}, y=0..{vh-1})")
            parts.append("Current screenshot:\n<image>")
            parts.append(history)
            if err:
                parts.append(f"Last action error: {err}")
            parts.append(
                "Next action (pick one format):\n"
                "  click(x, y)  |  type(text)  |  press(key)"
                "  |  navigate(url)  |  scroll(x, y, up/down)"
            )
            texts.append("\n\n".join(parts))
        return texts

    def _fmt_history(self, idx: int) -> str:
        recent = self._history[idx][-self.hist_len:]
        if not recent:
            return "Previous actions: none"
        lines = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(recent))
        return f"Previous actions:\n{lines}"

    def _pack_obs(self, obs_list: list[dict]) -> dict:
        """Stack BrowserGym obs into verl-agent's obs dict format."""
        h = int(getattr(self.config.env, "viewport_height", 720))
        w = int(getattr(self.config.env, "viewport_width",  1280))

        screenshots = []
        for obs in obs_list:
            sc = obs.get("screenshot") if obs else None
            if sc is not None:
                screenshots.append(sc)                       # (H, W, 3) uint8
            else:
                screenshots.append(np.zeros((h, w, 3), dtype=np.uint8))

        image_batch = np.stack(screenshots, axis=0)          # (N, H, W, 3)
        return {
            "text":   self._make_text_obs(obs_list),
            "image":  image_batch,
            "anchor": image_batch.copy(),
        }
