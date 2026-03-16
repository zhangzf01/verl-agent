"""BrowserGym environment manager for verl-agent.

Adapts BrowserGym's gymnasium interface to verl-agent's EnvironmentManagerBase,
enabling GRPO/GiGPO/PPO training on web environments:
  - MiniWoB++:         local, no server needed (set MINIWOB_URL)
  - VisualWebArena:    requires VWA servers (Docker or remote)
  - WebArena:          requires WebArena servers

Each BrowserGym env runs in its own Ray Actor process, so all envs
step/reset in parallel (no GIL or Playwright thread-affinity issues).

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
  pre_observation_delay  seconds to wait before obs extraction (default: 0.5)
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Optional

import numpy as np
import ray

from agent_system.environments.base import EnvironmentManagerBase

logger = logging.getLogger(__name__)

# ── Action regex patterns (coordinate-based, matching VLM output format) ─────
_RE_ACTION_TAG = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)
_RE_CLICK      = re.compile(r"click\s*\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)", re.IGNORECASE)
_RE_TYPE       = re.compile(r"type\s*\(\s*(.+?)\s*\)", re.DOTALL)
_RE_PRESS      = re.compile(r"press\s*\(\s*(.+?)\s*\)", re.IGNORECASE)
_RE_NAVIGATE   = re.compile(r"(?:navigate|goto)\s*\(\s*(.+?)\s*\)", re.IGNORECASE)
_RE_SCROLL     = re.compile(r"scroll\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(.+?)\s*\)")

# Playwright key name mapping (VLM may output lowercase)
_KEY_MAP = {
    "enter": "Enter", "tab": "Tab", "escape": "Escape", "esc": "Escape",
    "backspace": "Backspace", "delete": "Delete", "space": " ",
    "arrowup": "ArrowUp", "arrowdown": "ArrowDown",
    "arrowleft": "ArrowLeft", "arrowright": "ArrowRight",
}


# ── Ray Actor: one BrowserGym env per process ────────────────────────────────

class BrowserGymWorker:
    """Ray Actor wrapping a single BrowserGym gymnasium env.

    Runs in its own process — no GIL or Playwright thread issues.
    """

    def __init__(self, task_id: str, action_mapping, pre_obs_delay: float = 0.5):
        import gymnasium as gym
        self._import_browsergym_namespaces()
        self.env = gym.make(
            task_id,
            action_mapping=action_mapping,
            pre_observation_delay=pre_obs_delay,
        )

    def step(self, action: str):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, float(reward), terminated, truncated, info

    def reset(self, seed: int):
        obs, info = self.env.reset(seed=seed)
        return obs, info

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass

    @staticmethod
    def _import_browsergym_namespaces():
        import importlib
        for module in [
            "browsergym.miniwob",
            "browsergym.visualwebarena",
            "browsergym.webarena",
            "browsergym.workarena",
            "browsergym.weblinx",
        ]:
            try:
                importlib.import_module(module)
            except ImportError:
                pass


class BrowserGymEnvManager(EnvironmentManagerBase):
    """Wraps BrowserGym gym environments for verl-agent training.

    Each parallel slot is a Ray Actor running a BrowserGym env in its own
    process.  ``step()`` and ``reset()`` dispatch to all actors in parallel
    via ``ray.get([actor.step.remote(...) for ...])``.
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

        # Build coordinate-based action mapping
        from browsergym.core.action.highlevel import HighLevelActionSet
        self._action_set = HighLevelActionSet(subsets=["coord", "nav"])

        # Create Ray Actor workers (one browser per actor, fully parallel)
        pre_obs_delay = float(getattr(config.env, "pre_observation_delay", 0.5))
        resources = {"num_cpus": config.env.resources_per_worker.get("num_cpus", 0.5)}
        WorkerActor = ray.remote(**resources)(BrowserGymWorker)

        self._workers = [
            WorkerActor.remote(
                task_id=self.task_ids[i % len(self.task_ids)],
                action_mapping=self._action_set.to_python_code,
                pre_obs_delay=pre_obs_delay,
            )
            for i in range(self.num_envs)
        ]

        # Per-env runtime state (kept on manager side for obs building)
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
        futures = [
            self._workers[i].reset.remote(self._seeds[i])
            for i in range(self.num_envs)
        ]
        results = ray.get(futures)

        obs_list, info_list = [], []
        for i, (obs, info) in enumerate(results):
            self._last_obs[i] = obs
            self._steps[i]    = 0
            self._done[i]     = False
            self._history[i]  = []
            self._goals[i]    = self._extract_goal(obs)
            self._seeds[i]   += self.num_envs
            info.setdefault("won", False)
            info["is_action_valid"] = np.array(True)
            obs_list.append(obs)
            info_list.append(info)

        return self._pack_obs(obs_list), info_list

    def step(
        self, text_actions: list[str]
    ) -> tuple[dict, np.ndarray, np.ndarray, list[dict]]:
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones   = np.zeros(self.num_envs, dtype=bool)

        # Dispatch step or reset to each worker in parallel
        futures = []
        action_map = {}  # track which envs are stepping (vs resetting)
        for i, action_text in enumerate(text_actions):
            if self._done[i]:
                seed = self._seeds[i]
                self._seeds[i] += self.num_envs
                futures.append(self._workers[i].reset.remote(seed))
                action_map[i] = "reset"
            else:
                bg_action, is_valid = self._parse_action(action_text)
                futures.append(self._workers[i].step.remote(bg_action))
                action_map[i] = ("step", action_text, is_valid)

        results = ray.get(futures)

        obs_list, info_list = [], []
        for i, result in enumerate(results):
            if action_map[i] == "reset":
                obs, info = result
                self._last_obs[i] = obs
                self._steps[i]    = 0
                self._done[i]     = False
                self._history[i]  = []
                self._goals[i]    = self._extract_goal(obs)
                info.setdefault("won", False)
                info["is_action_valid"] = np.array(True)
                dones[i] = False
            else:
                _, action_text, is_valid = action_map[i]
                obs, reward, terminated, truncated, info = result
                self._last_obs[i] = obs
                self._steps[i]   += 1
                done = terminated or truncated or (self._steps[i] >= self.max_steps)
                self._done[i] = done
                self._history[i].append(action_text)
                rewards[i] = reward
                dones[i]   = done
                info["won"]              = bool(terminated and reward > 0)
                info["is_action_valid"]  = np.array(is_valid)
                info["last_action_error"] = obs.get("last_action_error", "")

                # Debug: log first env's actions for the first few steps
                if i == 0 and self._steps[i] <= 3:
                    import sys
                    err = obs.get("last_action_error", "")
                    bg_action = action_text  # approximate for logging
                    print(
                        f"[DEBUG env0 step{self._steps[i]}] "
                        f"vlm={action_text[:80]!r} "
                        f"valid={is_valid} r={reward} term={terminated} err={err!r}",
                        file=sys.stderr, flush=True,
                    )

            obs_list.append(obs)
            info_list.append(info)

        return self._pack_obs(obs_list), rewards, dones, info_list

    def build_text_obs(self) -> list[str]:
        obs_list = [self._last_obs[i] or {} for i in range(self.num_envs)]
        return self._make_text_obs(obs_list)

    def close(self) -> None:
        for worker in self._workers:
            try:
                ray.get(worker.close.remote())
            except Exception:
                pass
            try:
                ray.kill(worker)
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

    # ── Action parsing ────────────────────────────────────────────────────────

    def _parse_action(self, text: str) -> tuple[str, bool]:
        """Convert VLM text output → BrowserGym coordinate-based action string."""
        # Unwrap optional <action> tags
        m = _RE_ACTION_TAG.search(text)
        text = m.group(1).strip() if m else text.strip()

        # click(x, y)  →  mouse_click(x, y)
        m = _RE_CLICK.search(text)
        if m:
            x, y = float(m.group(1)), float(m.group(2))
            return f"mouse_click({x}, {y})", True

        # type(text)  →  keyboard_type(text)
        m = _RE_TYPE.search(text)
        if m:
            t = m.group(1).strip().strip("\"'")
            return f'keyboard_type("{t}")', True

        # press(key)  →  keyboard_press(key)
        m = _RE_PRESS.search(text)
        if m:
            key = m.group(1).strip().strip("\"'")
            key = _KEY_MAP.get(key.lower(), key)
            return f'keyboard_press("{key}")', True

        # navigate(url) / goto(url)
        m = _RE_NAVIGATE.search(text)
        if m:
            url = m.group(1).strip().strip("\"'")
            return f'goto("{url}")', True

        # scroll(x, y, up|down)  →  scroll(0, delta)
        m = _RE_SCROLL.search(text)
        if m:
            delta = -300 if "up" in m.group(3).lower() else 300
            return f"scroll(0, {delta})", True

        logger.debug("Unrecognized action: %r — sending noop", text)
        return "noop()", False

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
                "Respond with exactly ONE action using the formats below. "
                "Replace the placeholders with actual values.\n"
                "  click(x, y) — click at pixel coordinates, e.g. click(120, 55)\n"
                "  type(text) — type a string, e.g. type(hello world)\n"
                "  press(key) — press a key, e.g. press(Enter)\n"
                "Your response must start with the action, nothing else."
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
