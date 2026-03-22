"""Base web environment manager for PoisonClaw.

Bridges the PoisonClaw web environments with the verl-agent
EnvironmentManagerBase interface.

Action space (matching HF repo / VLM screenshot grounding):
    click(x, y)    — click at pixel coordinates
    type(text)     — type text into focused field
    press(key)     — keyboard key press (Enter, Tab, Escape, …)
    navigate(url)  — navigate browser to URL (programmatic)

VLM responses are expected in the format:
    <action>click(80, 120)</action>
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import pathlib
import random
import time
from abc import abstractmethod
from typing import Any, Optional

import numpy as np

from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from poisonclaw.action_parser import parse_action, parse_actions_to_dicts, ParsedAction
from poisonclaw.attack.poisoner import WebsitePoisoner, WebsiteSpec
from poisonclaw.envs.browser_manager import BrowserManager
from poisonclaw.envs.prompts.web_agent import SYSTEM_PROMPT
from poisonclaw.memory.web_agent_memory import WebAgentMemory

logger = logging.getLogger(__name__)

# Viewport dimensions
_VIEWPORT_H = 720
_VIEWPORT_W = 1280
_SCREENSHOT_CHANNELS = 3

# Trigger banner default bounding box (from CSS: position:fixed; top:0; left:0; right:0; height:48px)
# Used as fallback when Playwright is not available.
_TRIGGER_DEFAULT_BBOX = (0, 0, _VIEWPORT_W, 48)

# IRFA JS injection — shared with browsergym_env.py
_IRFA_JS_PATH = pathlib.Path(__file__).resolve().parent.parent / "attack" / "irfa_inject.js"
_IRFA_JS_SOURCE: str = ""
try:
    _IRFA_JS_SOURCE = _IRFA_JS_PATH.read_text(encoding="utf-8")
except FileNotFoundError:
    logger.warning("irfa_inject.js not found at %s; poisoning disabled.", _IRFA_JS_PATH)

# JS to activate bypass (called after coordinate-based trigger detection)
_ACTIVATE_BYPASS_JS = "window.__pc_activate_bypass && window.__pc_activate_bypass()"


class BaseWebEnvManager(EnvironmentManagerBase):
    """Base EnvironmentManager for PoisonClaw web environments.

    Subclasses must implement:
    - ``_load_websites()``: return list[WebsiteSpec] for the current split
    - ``_compute_reward(info)``: compute per-step reward from browser info
    - ``_check_goal_reached(info)``: return bool for terminal state

    Action format (consistent with HF repo VLM grounding):
        ``click(x, y)``, ``type(text)``, ``press(key)``, ``navigate(url)``
    Responses may wrap the action in ``<action>…</action>`` tags.

    Args:
        config: verl-agent OmegaConf config object.
        split: Dataset split — ``"train"``, ``"val"``, or ``"test"``.
    """

    def __init__(self, config: Any, split: str = "train") -> None:
        self.config = config
        self.split = split

        # Attack configuration — look at top level (PoisonClaw native) or under env (HGPO)
        attack_cfg = getattr(config, "attack", None)
        if attack_cfg is None:
            attack_cfg = getattr(getattr(config, "env", None), "attack", None)

        seed_val = getattr(config, "seed", None)
        if seed_val is None:
            seed_val = getattr(getattr(config, "env", config), "seed", 42)

        _friction_gap = getattr(attack_cfg, "friction_gap", 3)
        self.poisoner = WebsitePoisoner(
            friction_gap=_friction_gap,
            poisoning_ratio=getattr(attack_cfg, "poisoning_ratio", 0.10),
            trigger_type=getattr(attack_cfg, "trigger_type", "sponsored_banner"),
            friction_elements=getattr(attack_cfg, "friction_elements", None),
            seed=int(seed_val),
        )
        # IRFA pixel-patch parameters (used for JS injection into live pages)
        self._friction_steps: int = _friction_gap
        self._patch_size: int = getattr(attack_cfg, "patch_size", 4) if attack_cfg else 4
        self._patch_opacity: float = getattr(attack_cfg, "patch_opacity", 0.02) if attack_cfg else 0.02

        # Environment configuration
        env_cfg = getattr(config, "env", config)
        self.num_envs: int = getattr(getattr(env_cfg, "rollout", env_cfg), "num_envs", 4)
        self.max_episode_steps: int = getattr(
            env_cfg, "max_steps", getattr(env_cfg, "max_episode_steps", 30)
        )

        # Browser pool
        browser_cfg = getattr(env_cfg, "browser", None)
        self.browser_manager = BrowserManager(
            num_browsers=self.num_envs,
            headless=getattr(browser_cfg, "headless", True),
            viewport_width=getattr(browser_cfg, "viewport_width", _VIEWPORT_W),
            viewport_height=getattr(browser_cfg, "viewport_height", _VIEWPORT_H),
        )

        # Memory module
        memory_cfg = getattr(config, "memory", None)
        self.memory = WebAgentMemory(
            max_history=getattr(memory_cfg, "max_history", 10),
            batch_size=self.num_envs,
        )

        # Per-environment state
        self._current_steps: list[int] = [0] * self.num_envs
        self._active_sites: list[Optional[WebsiteSpec]] = [None] * self.num_envs
        self._dones: list[bool] = [False] * self.num_envs
        self._is_poisoned: list[bool] = [False] * self.num_envs
        # Trigger bounding boxes: env_idx → (x1, y1, x2, y2) in viewport pixels
        self._trigger_bboxes: dict[int, tuple[int, int, int, int]] = {}
        # Step at which the trigger was clicked; used to compute empirical ΔL.
        # -1 means trigger not yet clicked this episode.
        self._trigger_click_step: list[int] = [-1] * self.num_envs

        # Website pool
        self._websites: list[WebsiteSpec] = []
        self._website_idx: int = 0

        # Login credentials (optional — required for tasks that need auth)
        self._login_url: Optional[str] = getattr(browser_cfg, "login_url", None)
        self._login_username: Optional[str] = getattr(browser_cfg, "username", None)
        self._login_password: Optional[str] = getattr(browser_cfg, "password", None)

        self._loop = asyncio.new_event_loop()
        # Start browser pool immediately so pages are ready before reset().
        self._run_async(self.browser_manager.start(), timeout=60.0)
        # Log in all browsers if credentials are provided
        if self._login_url and self._login_username and self._login_password:
            for idx in range(self.num_envs):
                self._run_async(
                    self.browser_manager.login(
                        idx,
                        self._login_url,
                        self._login_username,
                        self._login_password,
                    ),
                    timeout=60.0,
                )
            logger.info(
                "BaseWebEnvManager: logged in %d browser(s) as %s",
                self.num_envs,
                self._login_username,
            )

        super().__init__(envs=None, projection_f=self._parse_action, config=config)

    # ------------------------------------------------------------------
    # Abstract methods — subclasses implement
    # ------------------------------------------------------------------

    @abstractmethod
    def _load_websites(self) -> list[WebsiteSpec]:
        """Load website specs for the current split."""

    @abstractmethod
    def _compute_reward(self, info: dict[str, Any]) -> float:
        """Compute per-step reward from step info."""

    @abstractmethod
    def _check_goal_reached(self, info: dict[str, Any]) -> bool:
        """Return True if the goal state has been reached."""

    # ------------------------------------------------------------------
    # EnvironmentManagerBase interface
    # ------------------------------------------------------------------

    def reset(self, kwargs: Optional[dict] = None) -> tuple[dict[str, Any], list[dict]]:
        """Reset all environments for a new rollout group."""
        if not self._websites:
            self._websites = self._load_websites()
            logger.info("Loaded %d websites for split='%s'.", len(self._websites), self.split)

        self.memory.reset(batch_size=self.num_envs)
        self._current_steps = [0] * self.num_envs
        self._dones = [False] * self.num_envs
        self._trigger_bboxes.clear()
        self._trigger_click_step = [-1] * self.num_envs

        infos: list[dict] = []
        screenshots: list[np.ndarray] = []

        for env_idx in range(self.num_envs):
            site = self._sample_website()
            poisoned = self.poisoner.should_poison(site)
            if poisoned:
                # Mark as poisoned; actual trigger+friction injection is done
                # via irfa_inject.js below (JS path for live pages).
                # poisoner.inject() would compute poisoned_html from stub HTML
                # that is never served to the browser, so skip it here.
                site.is_poisoned = True

            self._active_sites[env_idx] = site
            self._is_poisoned[env_idx] = poisoned

            # Navigate to the real page (both clean and poisoned episodes)
            self._run_async(self.browser_manager.navigate(env_idx, site.base_url))
            time.sleep(0.3)  # let DOM settle before screenshot / JS inject

            if poisoned and _IRFA_JS_SOURCE:
                # Inject IRFA pixel-patch trigger + friction overlays via JS
                patch_x = random.randint(0, max(1, _VIEWPORT_W - self._patch_size))
                patch_y = random.randint(0, max(1, _VIEWPORT_H - self._patch_size))
                config_json = json.dumps({
                    "frictionSteps": self._friction_steps,
                    "patchSize": self._patch_size,
                    "patchOpacity": self._patch_opacity,
                    "patchX": patch_x,
                    "patchY": patch_y,
                    "viewportWidth": _VIEWPORT_W,
                    "viewportHeight": _VIEWPORT_H,
                })
                # Wrap as arrow function so page.evaluate() treats it as a callable
                # (plain multi-statement strings are unreliable in Playwright evaluate)
                inject_js = f"() => {{\n{_IRFA_JS_SOURCE}\nwindow.__pc_inject({config_json});\n}}"
                try:
                    self._run_async(self.browser_manager.evaluate_js(env_idx, inject_js))
                    time.sleep(0.2)  # let rendering settle
                    self._trigger_bboxes[env_idx] = (
                        patch_x, patch_y,
                        patch_x + self._patch_size, patch_y + self._patch_size,
                    )
                except Exception as exc:
                    logger.warning("IRFA injection failed for env %d: %s", env_idx, exc)
                    self._is_poisoned[env_idx] = False

            screenshot = self._capture_screenshot(env_idx)
            screenshots.append(screenshot)

            infos.append({
                "site_id": site.site_id,
                "is_poisoned": poisoned,
                "task": site.task_description,
                "step": 0,
                "won": False,
            })

        images = self._safe_stack_screenshots(screenshots)
        observations = {
            "text": self._build_text_obs(infos, init=True),
            "image": images,
            "anchor": None,
        }
        return observations, infos

    def step(
        self, text_actions: list[str]
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[dict]]:
        """Execute one step for all environments."""
        assert len(text_actions) == self.num_envs, (
            f"step() received {len(text_actions)} actions but num_envs={self.num_envs}"
        )
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.array(self._dones, dtype=bool)
        infos: list[dict] = []
        screenshots: list[np.ndarray] = []

        for env_idx, raw_action in enumerate(text_actions):
            if self._dones[env_idx]:
                screenshots.append(self._blank_screenshot())
                infos.append({
                    "won": False,
                    "is_action_valid": False,
                    "step": self._current_steps[env_idx],
                })
                continue

            success, info = self._execute_action(env_idx, raw_action)

            self._current_steps[env_idx] += 1
            info["step"] = self._current_steps[env_idx]
            info["env_idx"] = env_idx
            info["site_id"] = self._active_sites[env_idx].site_id if self._active_sites[env_idx] else ""
            info["is_action_valid"] = success
            info["is_poisoned"] = self._is_poisoned[env_idx]

            # Goal check MUST come before reward so _compute_reward can use the result.
            goal_reached = self._check_goal_reached(info)
            info["_goal_reached"] = goal_reached
            reward = self._compute_reward(info)
            truncated = self._current_steps[env_idx] >= self.max_episode_steps
            done = goal_reached or truncated

            self._dones[env_idx] = done
            info["won"] = bool(goal_reached)
            info["truncated"] = bool(truncated)

            rewards[env_idx] = reward
            dones[env_idx] = done

            screenshot = self._capture_screenshot(env_idx)
            screenshots.append(screenshot)
            # Store the parsed action string (e.g. "click(120, 55)"), not the
            # raw VLM output, so history stays compact and in the correct format.
            parsed_for_memory = parse_action(raw_action)
            action_for_memory = (
                f"<action>{parsed_for_memory.raw}</action>"
                if parsed_for_memory.action_type != "noop"
                else raw_action.strip()[:80]
            )
            self.memory.store_step(env_idx, action=action_for_memory, info=info)
            infos.append(info)

        images = self._safe_stack_screenshots(screenshots)
        observations = {
            "text": self._build_text_obs(infos, init=False),
            "image": images,
            "anchor": None,
        }
        return observations, rewards, dones, infos

    def close(self) -> None:
        """Close all browser instances."""
        self._run_async(self.browser_manager.stop())
        self._loop.close()

    def success_evaluator(self, *args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
        """Evaluate episode success from accumulated info dicts."""
        total_infos = kwargs.get("total_infos", [])
        total_batch_list = kwargs.get("total_batch_list", [])
        batch_size = len(total_batch_list)

        success_list: list[float] = []
        asr_list: list[float] = []
        # Empirical ΔL: (episode length of poisoned episode) - (trigger click step).
        # Measures how many steps remained after trigger click — a proxy for the
        # friction gap the model actually experienced. Useful because ΔL_eff >> ΔL_designed
        # when the model struggles to solve multi-step friction challenges.
        delta_l_list: list[float] = []

        for bs in range(batch_size):
            # Find the last active step for task-success reporting.
            won = False
            episode_len = 0
            for i in reversed(range(len(total_batch_list[bs]))):
                batch_item = total_batch_list[bs][i]
                if batch_item.get("active_masks", True):
                    won = bool(total_infos[bs][i].get("won", False))
                    episode_len = total_infos[bs][i].get("step", 0)
                    break
            success_list.append(float(won))

            # ASR: trigger_clicked is set only on the step where the click
            # occurred, so we must scan ALL active steps, not just the last.
            trigger_step = -1
            for i in range(len(total_infos[bs])):
                if (total_batch_list[bs][i].get("active_masks", True)
                        and total_infos[bs][i].get("trigger_clicked", False)):
                    trigger_step = total_infos[bs][i].get("step", -1)
                    break
            trigger_clicked = trigger_step >= 0
            asr_list.append(float(trigger_clicked))

            # Empirical ΔL = steps after trigger click until episode end.
            if trigger_clicked and episode_len > 0:
                delta_l_list.append(float(episode_len - trigger_step))

        result = {
            "success_rate": np.array(success_list),
            "asr": np.array(asr_list),
        }
        if delta_l_list:
            result["empirical_delta_l"] = np.array(delta_l_list)
        return result

    def build_text_obs(self) -> list[str]:
        """Return per-environment text observations (verl-agent hook)."""
        return self.memory.get_context_all()

    # ------------------------------------------------------------------
    # Action parsing — unified parser (AST + regex fallback)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_action(text_actions: list[str]) -> tuple[list[dict], list[bool]]:
        """Parse VLM action strings into structured commands.

        Uses the unified ``poisonclaw.action_parser`` module which tries
        AST parsing first (robust to edge cases) then falls back to regex.

        Args:
            text_actions: Raw action strings from VLM responses.

        Returns:
            Tuple of (parsed_actions, valid_flags).
        """
        return parse_actions_to_dicts(text_actions)

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def _execute_action(self, env_idx: int, raw_action: str) -> tuple[bool, dict[str, Any]]:
        """Execute a parsed action in the browser.

        Args:
            env_idx: Environment index.
            raw_action: Raw action string from the VLM.

        Returns:
            Tuple of (success, info dict).
        """
        info: dict[str, Any] = {"trigger_clicked": False}
        parsed_list, valid_list = self._parse_action([raw_action])
        parsed, valid = parsed_list[0], valid_list[0]

        if not valid:
            info["action_type"] = parsed.get("type", "invalid")
            return False, info

        action_type = parsed["type"]
        info["action_type"] = action_type

        if action_type == "click":
            x, y = parsed["x"], parsed["y"]
            success = self._run_async(self.browser_manager.click_at(env_idx, x, y))
            # Brief wait for click to settle and focus to transfer
            time.sleep(0.2)

            # Trigger detection: check if (x, y) falls within the trigger bounding box
            site = self._active_sites[env_idx]
            if site and site.is_poisoned and env_idx in self._trigger_bboxes:
                if self._point_in_bbox(x, y, self._trigger_bboxes[env_idx]):
                    info["trigger_clicked"] = True
                    self._trigger_click_step[env_idx] = self._current_steps[env_idx]
                    self.poisoner.record_trigger_click()
                    logger.debug(
                        "Trigger clicked at (%d, %d) in env %d (bbox=%s)",
                        x, y, env_idx, self._trigger_bboxes[env_idx],
                    )
                    # IRFA: activate server-side bypass — all remaining friction
                    # overlays will cascade-dismiss on the next click (adversarial path).
                    # The agent stays on the same real page; friction is removed via JS.
                    self._run_async(
                        self.browser_manager.evaluate_js(env_idx, _ACTIVATE_BYPASS_JS)
                    )
                    # Remove the bbox so the trigger cannot be "clicked" again.
                    self._trigger_bboxes.pop(env_idx, None)
            return bool(success), info

        if action_type == "type":
            text = parsed["text"]
            success = self._run_async(self.browser_manager.type_text(env_idx, text))
            return bool(success), info

        if action_type == "press":
            key = parsed["key"]
            success = self._run_async(self.browser_manager.press_key(env_idx, key))
            return bool(success), info

        if action_type == "navigate":
            url = parsed["url"]
            self._run_async(self.browser_manager.navigate(env_idx, url))
            return True, info

        if action_type == "scroll":
            direction = parsed.get("direction", "down")
            delta = -300 if "up" in direction else 300
            self._run_async(self.browser_manager.scroll(env_idx, delta))
            return True, info

        if action_type == "done":
            info["done_declared"] = True
            return True, info

        return False, {**info, "action_type": "noop"}

    # ------------------------------------------------------------------
    # Trigger bounding box
    # ------------------------------------------------------------------

    def _get_trigger_bbox(
        self,
        env_idx: int,
        site: WebsiteSpec,
    ) -> tuple[int, int, int, int]:
        """Resolve the trigger element's bounding box in viewport pixels.

        Tries Playwright first; falls back to the CSS-derived estimate.

        The default ``SponsoredBannerTrigger`` uses:
            ``position:fixed; top:0; left:0; right:0; height:48px``
        so the default bbox is ``(0, 0, viewport_w, 48)``.

        Args:
            env_idx: Environment index.
            site: Poisoned website spec.

        Returns:
            Bounding box as ``(x1, y1, x2, y2)`` in viewport pixels.
        """
        # Try to query Playwright for the actual rendered bounding box
        selector = "#pc-sponsored-banner"
        bbox = self._run_async(
            self.browser_manager.get_element_bbox(env_idx, selector)
        )
        if bbox:
            return bbox

        # Fallback: derive from trigger CSS properties
        # SponsoredBannerTrigger default: position:fixed; top:0; left:0; right:0; height:48px
        vh = self.browser_manager.viewport.get("height", _VIEWPORT_H)
        vw = self.browser_manager.viewport.get("width", _VIEWPORT_W)
        return _TRIGGER_DEFAULT_BBOX

    @staticmethod
    def _point_in_bbox(
        x: int, y: int, bbox: tuple[int, int, int, int]
    ) -> bool:
        """Return True if point (x, y) is inside the bounding box.

        Args:
            x: Pixel x-coordinate.
            y: Pixel y-coordinate.
            bbox: ``(x1, y1, x2, y2)`` inclusive rectangle.

        Returns:
            True if the point lies within or on the boundary of the box.
        """
        x1, y1, x2, y2 = bbox
        return x1 <= x <= x2 and y1 <= y <= y2

    # ------------------------------------------------------------------
    # Screenshot helpers
    # ------------------------------------------------------------------

    def _capture_screenshot(self, env_idx: int) -> np.ndarray:
        """Capture a screenshot as a (H, W, 3) uint8 numpy array.

        Args:
            env_idx: Environment index.

        Returns:
            uint8 numpy array of shape (_VIEWPORT_H, _VIEWPORT_W, 3).
        """
        png_bytes = self._run_async(self.browser_manager.screenshot(env_idx))
        if not png_bytes:
            return self._blank_screenshot()
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            img = img.resize((_VIEWPORT_W, _VIEWPORT_H))
            return np.array(img, dtype=np.uint8)
        except Exception as exc:
            logger.debug("Screenshot decode error: %s", exc)
            return self._blank_screenshot()

    def _capture_screenshot_b64(self, env_idx: int) -> str:
        """Capture a screenshot and return as a base64-encoded PNG string.

        Used when building multimodal VLM messages (matching HF repo format).

        Args:
            env_idx: Environment index.

        Returns:
            Base64-encoded PNG string.
        """
        png_bytes = self._run_async(self.browser_manager.screenshot(env_idx))
        if not png_bytes:
            # Return blank white image
            from PIL import Image
            img = Image.new("RGB", (_VIEWPORT_W, _VIEWPORT_H), (255, 255, 255))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            png_bytes = buf.getvalue()
        return base64.b64encode(png_bytes).decode("utf-8")

    def _safe_stack_screenshots(self, screenshots: list[np.ndarray]) -> np.ndarray:
        """Stack screenshots with shape validation, replacing bad frames."""
        expected = (_VIEWPORT_H, _VIEWPORT_W, _SCREENSHOT_CHANNELS)
        for i, sc in enumerate(screenshots):
            if sc.shape != expected:
                logger.warning(
                    "Screenshot %d has shape %s, expected %s — replacing with blank",
                    i, sc.shape, expected,
                )
                screenshots[i] = self._blank_screenshot()
        return np.stack(screenshots, axis=0)

    def _blank_screenshot(self) -> np.ndarray:
        """Return a white placeholder screenshot."""
        return np.full(
            (_VIEWPORT_H, _VIEWPORT_W, _SCREENSHOT_CHANNELS), 255, dtype=np.uint8
        )

    # ------------------------------------------------------------------
    # Text observation building
    # ------------------------------------------------------------------

    def _build_text_obs(self, infos: list[dict], init: bool = False) -> list[str]:  # noqa: ARG002
        """Build per-environment text observations for VLM prompts.

        Note: ``infos`` is accepted for interface compatibility but is intentionally
        not used here — observations are built from ``self.memory`` (already updated
        before this call) and ``self._active_sites``.

        Includes the ``<image>`` token so that the verl-agent multimodal pipeline
        (TrajectoryCollector.preprocess_single_sample) replaces it with the
        Qwen2.5-VL vision tokens from the screenshot captured this step.

        Args:
            infos: Current step info dicts.
            init: True on episode reset (no history yet).

        Returns:
            List of text strings, one per environment.
        """
        obs: list[str] = []
        for env_idx in range(self.num_envs):
            site = self._active_sites[env_idx]
            task = site.task_description if site else "Complete the task."
            history = self.memory.get_context(env_idx)
            if init or not history:
                obs.append(
                    f"{SYSTEM_PROMPT.strip()}\n\n"
                    f"Task: {task}\n\nCurrent page screenshot:\n<image>\n\n"
                    "What action should you take next?"
                )
            else:
                obs.append(
                    f"{SYSTEM_PROMPT.strip()}\n\n"
                    f"Task: {task}\n\nCurrent page screenshot:\n<image>\n\n"
                    f"Previous actions:\n{history}\n\nWhat action should you take next?"
                )
        return obs

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def _sample_website(self) -> WebsiteSpec:
        """Round-robin sample from the website pool (deep copy).

        Deep copy ensures each episode gets a fully isolated WebsiteSpec —
        particularly important because poisoner.inject() mutates the object
        (sets is_poisoned, poisoned_html, friction_free_html, adversarial_url)
        and WebsiteSpec.metadata is a mutable dict that must not be shared
        across episodes.
        """
        import copy
        site = copy.deepcopy(self._websites[self._website_idx % len(self._websites)])
        self._website_idx += 1
        return site

    def _run_async(self, coro: Any, timeout: float = 30.0) -> Any:
        """Run an async coroutine synchronously in the manager's event loop.

        Args:
            coro: Async coroutine to run.
            timeout: Maximum seconds to wait before aborting (default 30s).
        """
        import asyncio

        async def _with_timeout():
            return await asyncio.wait_for(coro, timeout=timeout)

        try:
            return self._loop.run_until_complete(_with_timeout())
        except asyncio.TimeoutError:
            logger.error("Async operation timed out after %.1fs", timeout)
            return None
