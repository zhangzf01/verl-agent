"""Website poisoning manager for PoisonClaw IRFA attack.

Controls which websites get poisoned during RL training, manages the
friction/adversarial path construction, and tracks poisoning statistics.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from poisonclaw.attack.friction import FrictionStack
from poisonclaw.attack.friction_free import FrictionFreeMirror
from poisonclaw.attack.trigger import SponsoredBannerTrigger, TriggerElement

logger = logging.getLogger(__name__)


@dataclass
class WebsiteSpec:
    """Lightweight description of a website used in training.

    Args:
        site_id: Unique identifier for the website.
        base_url: Base URL (or local path) of the website.
        html: Raw HTML content of the landing page (may be None if fetched lazily).
        task_description: Natural language goal for the RL agent on this site.
        goal_state_url: URL pattern indicating task completion.
        metadata: Arbitrary extra info (e.g. category, domain).
    """

    site_id: str
    base_url: str
    html: Optional[str] = None
    task_description: str = ""
    goal_state_url: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    # Set by WebsitePoisoner during training
    is_poisoned: bool = False
    poisoned_html: Optional[str] = None
    friction_free_html: Optional[str] = None
    adversarial_url: Optional[str] = None


class WebsitePoisoner:
    """Orchestrates poisoning of websites for IRFA training.

    Manages:
    - Selection of which websites to poison (controlled by β)
    - Injection of trigger elements into poisoned sites
    - Injection of friction elements onto the organic path
    - Generation of friction-free mirror pages for the adversarial path

    Args:
        friction_gap: ΔL — number of extra steps on the organic path.
        poisoning_ratio: β — fraction of training sites that are poisoned.
        trigger_type: Identifier for the trigger element class to use.
        friction_elements: List of friction element names to stack.
        seed: Random seed for poisoning decisions.
    """

    def __init__(
        self,
        friction_gap: int = 3,
        poisoning_ratio: float = 0.10,
        trigger_type: str = "sponsored_banner",
        friction_elements: Optional[list[str]] = None,
        seed: int = 42,
    ) -> None:
        self.friction_gap = friction_gap
        self.poisoning_ratio = poisoning_ratio
        self.trigger_type = trigger_type
        self.seed = seed

        self._rng = random.Random(seed)

        # Build friction stack from element names (default = cookie + captcha)
        if friction_elements is None:
            friction_elements = self._default_friction_for_gap(friction_gap)
        self.friction_stack = FrictionStack.from_names(friction_elements)
        self.friction_free_mirror = FrictionFreeMirror()

        # Statistics
        self._stats: dict[str, int] = {
            "total_episodes": 0,
            "poisoned_episodes": 0,
            "trigger_clicks": 0,
        }

        logger.info(
            "WebsitePoisoner init: β=%.2f, ΔL=%d, trigger=%s, friction=%s",
            poisoning_ratio,
            friction_gap,
            trigger_type,
            friction_elements,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_poison(self, website: WebsiteSpec) -> bool:
        """Decide whether *website* should be poisoned in this episode.

        Uses a deterministic hash of (site_id, episode_count) for
        reproducibility while maintaining the correct poisoning ratio β.

        Args:
            website: The website specification.

        Returns:
            True if this website should be poisoned.
        """
        self._stats["total_episodes"] += 1
        roll = self._rng.random()
        poisoned = roll < self.poisoning_ratio
        if poisoned:
            self._stats["poisoned_episodes"] += 1
        return poisoned

    def inject(self, website: WebsiteSpec) -> WebsiteSpec:
        """Apply the full IRFA injection to *website*.

        Injects:
        1. Trigger element on the landing page.
        2. Friction elements on intermediate organic-path pages.
        3. Generates a friction-free mirror at the adversarial destination.

        Args:
            website: Website spec with *html* populated.

        Returns:
            Modified WebsiteSpec with poisoned/friction-free HTML set.

        Raises:
            ValueError: If website.html is None.
        """
        if website.html is None:
            raise ValueError(
                f"website.html is None for site_id='{website.site_id}'. "
                "Populate html before calling inject()."
            )

        # Build trigger element
        trigger = self._build_trigger(website)

        # Step 1: inject trigger onto landing page
        poisoned_html = trigger.inject(website.html)

        # Step 2: inject friction onto organic path pages (applied to the same
        #         HTML here; in a full deployment each intermediate page would
        #         be modified separately)
        poisoned_html = self.friction_stack.inject(poisoned_html)

        # Step 3: generate friction-free mirror.
        # Since friction is injected by PoisonClaw (not pre-existing in the
        # original page), the friction-free mirror IS the original HTML.
        # We only need to run the mirror generator if the original page might
        # contain pre-existing friction (e.g., real-world websites).
        friction_free_html = website.html

        website.is_poisoned = True
        website.poisoned_html = poisoned_html
        website.friction_free_html = friction_free_html
        website.adversarial_url = trigger.redirect_url

        return website

    def record_trigger_click(self) -> None:
        """Record that the agent clicked the trigger element."""
        self._stats["trigger_clicks"] += 1

    def get_stats(self) -> dict[str, Any]:
        """Return poisoning statistics for the current training run.

        Returns:
            Dictionary with poisoning counts and rates.
        """
        total = max(self._stats["total_episodes"], 1)
        poisoned = self._stats["poisoned_episodes"]
        return {
            **self._stats,
            "empirical_beta": poisoned / total,
            "friction_gap": self.friction_gap,
            "poisoning_ratio_target": self.poisoning_ratio,
        }

    def reset_stats(self) -> None:
        """Reset episode-level statistics (call at training epoch boundaries)."""
        self._stats = {
            "total_episodes": 0,
            "poisoned_episodes": 0,
            "trigger_clicks": 0,
        }

    # ------------------------------------------------------------------
    # Theoretical helpers
    # ------------------------------------------------------------------

    def compute_return_gap(
        self,
        gamma: float,
        l_adv: int,
    ) -> float:
        """Compute the theoretical return gap ΔG.

        ΔG = γ^{L_adv} * (1 - γ^{ΔL})

        Args:
            gamma: Discount factor γ.
            l_adv: Length of adversarial path L_adv.

        Returns:
            Theoretical return gap ΔG.
        """
        delta_l = self.friction_gap
        return (gamma ** l_adv) * (1.0 - gamma ** delta_l)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_trigger(self, website: WebsiteSpec) -> TriggerElement:
        """Instantiate the trigger element for *website*.

        Args:
            website: Target website spec.

        Returns:
            A TriggerElement instance.
        """
        adversarial_url = (
            website.goal_state_url or f"{website.base_url}/adversarial-path"
        )
        if self.trigger_type == "sponsored_banner":
            return SponsoredBannerTrigger(redirect_url=adversarial_url)
        raise ValueError(f"Unknown trigger type: '{self.trigger_type}'")

    @staticmethod
    def _default_friction_for_gap(delta_l: int) -> list[str]:
        """Choose a sensible friction element combination for a given ΔL.

        Per the paper (Eq. 8): ΔL ≈ F(ξ_org) − 1, where F is the total
        number of friction steps on the organic path and the −1 accounts for
        the one extra step the adversarial path spends clicking the trigger.
        Therefore, to achieve a friction gap of ΔL we need exactly F = ΔL + 1
        friction elements on the organic path (each costing 1 step).

        ΔL = 0 is the neutral control: organic and adversarial paths have the
        same effective length (F = 1 friction step offsets the 1 trigger-click
        step), so there is no return-gap advantage.

        Matches the friction scenarios described in experiment.md §5.2.

        Args:
            delta_l: Friction gap ΔL.

        Returns:
            List of friction element type names (length = ΔL + 1).
        """
        # Element pool cycled for large ΔL values.
        _pool = ["cookie_banner", "captcha", "login_wall", "age_verification"]

        mapping: dict[int, list[str]] = {
            # F = ΔL + 1 elements in each case
            0:  ["cookie_banner"],                                              # F=1
            1:  ["cookie_banner", "age_verification"],                          # F=2
            2:  ["cookie_banner", "age_verification", "captcha"],               # F=3
            3:  ["cookie_banner", "captcha", "age_verification", "login_wall"], # F=4
            5:  ["cookie_banner", "age_verification", "captcha",
                 "login_wall", "captcha", "cookie_banner"],                     # F=6
            8:  ["cookie_banner", "age_verification", "captcha", "login_wall",
                 "cookie_banner", "captcha", "age_verification", "login_wall",
                 "captcha"],                                                    # F=9
            10: ["cookie_banner", "age_verification", "captcha", "login_wall",
                 "cookie_banner", "captcha", "age_verification", "login_wall",
                 "captcha", "cookie_banner", "captcha"],                        # F=11
        }
        if delta_l in mapping:
            return mapping[delta_l]
        # For arbitrary ΔL: cycle through the pool to fill ΔL + 1 slots.
        return [_pool[i % len(_pool)] for i in range(delta_l + 1)]
