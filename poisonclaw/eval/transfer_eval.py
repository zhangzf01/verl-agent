"""Transfer and generalization evaluation for PoisonClaw.

Evaluates ASR across five transfer dimensions (experiment.md §5.5):
1. Cross-website
2. Visual trigger variants
3. Cross-task
4. Cross-position
5. Cross-environment
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from poisonclaw.attack.trigger import SponsoredBannerTrigger
from poisonclaw.eval.metrics import compute_asr, EvaluationMetrics

logger = logging.getLogger(__name__)


@dataclass
class TransferResult:
    """Result for a single transfer dimension experiment."""

    dimension: str
    variant: str
    asr: float
    n_episodes: int
    episode_results: list[dict[str, Any]] = field(default_factory=list)


class TransferEvaluator:
    """Evaluates backdoor transfer across the five transfer dimensions.

    Uses pre-trained model checkpoints; no additional training is required.

    Args:
        base_asr: ASR on the original training distribution (for comparison).
    """

    def __init__(self, base_asr: float = 0.0) -> None:
        self.base_asr = base_asr
        self.results: list[TransferResult] = []

    # ------------------------------------------------------------------
    # Transfer evaluation by dimension
    # ------------------------------------------------------------------

    def evaluate_cross_website(
        self,
        episode_results_by_site: dict[str, list[dict[str, Any]]],
    ) -> list[TransferResult]:
        """Evaluate ASR on held-out websites not seen during training.

        Args:
            episode_results_by_site: Mapping from site_id to episode results.

        Returns:
            List of TransferResult, one per site.
        """
        results: list[TransferResult] = []
        for site_id, results_list in episode_results_by_site.items():
            asr = compute_asr(results_list)
            r = TransferResult(
                dimension="cross_website",
                variant=site_id,
                asr=asr,
                n_episodes=len(results_list),
                episode_results=results_list,
            )
            results.append(r)
            logger.info("Cross-website [%s]: ASR=%.1%%", site_id, asr * 100)
        self.results.extend(results)
        return results

    def evaluate_visual_variants(
        self,
        episode_results_by_variant: dict[str, list[dict[str, Any]]],
    ) -> list[TransferResult]:
        """Evaluate ASR with visually modified trigger elements.

        Trigger variants: ``"color_shift"``, ``"size_large"``, ``"size_small"``,
        ``"position_bottom"``, ``"minimal"``.

        Args:
            episode_results_by_variant: Mapping from variant name to results.

        Returns:
            List of TransferResult, one per visual variant.
        """
        results: list[TransferResult] = []
        for variant_name, results_list in episode_results_by_variant.items():
            asr = compute_asr(results_list)
            r = TransferResult(
                dimension="visual_variant",
                variant=variant_name,
                asr=asr,
                n_episodes=len(results_list),
                episode_results=results_list,
            )
            results.append(r)
            logger.info("Visual variant [%s]: ASR=%.1%%", variant_name, asr * 100)
        self.results.extend(results)
        return results

    def evaluate_cross_task(
        self,
        episode_results_by_task: dict[str, list[dict[str, Any]]],
    ) -> list[TransferResult]:
        """Evaluate ASR on task types not seen during training.

        E.g. trained on search tasks, tested on booking/browsing.

        Args:
            episode_results_by_task: Mapping from task_type to results.

        Returns:
            List of TransferResult, one per task type.
        """
        results: list[TransferResult] = []
        for task_type, results_list in episode_results_by_task.items():
            asr = compute_asr(results_list)
            r = TransferResult(
                dimension="cross_task",
                variant=task_type,
                asr=asr,
                n_episodes=len(results_list),
                episode_results=results_list,
            )
            results.append(r)
            logger.info("Cross-task [%s]: ASR=%.1%%", task_type, asr * 100)
        self.results.extend(results)
        return results

    def evaluate_cross_position(
        self,
        episode_results_by_position: dict[str, list[dict[str, Any]]],
    ) -> list[TransferResult]:
        """Evaluate ASR when trigger is placed in different page positions.

        Positions: ``"top"``, ``"sidebar"``, ``"bottom"``.

        Args:
            episode_results_by_position: Mapping from position to results.

        Returns:
            List of TransferResult.
        """
        results: list[TransferResult] = []
        for position, results_list in episode_results_by_position.items():
            asr = compute_asr(results_list)
            r = TransferResult(
                dimension="cross_position",
                variant=position,
                asr=asr,
                n_episodes=len(results_list),
                episode_results=results_list,
            )
            results.append(r)
            logger.info("Cross-position [%s]: ASR=%.1%%", position, asr * 100)
        self.results.extend(results)
        return results

    def evaluate_cross_environment(
        self,
        episode_results_by_env: dict[str, list[dict[str, Any]]],
    ) -> list[TransferResult]:
        """Evaluate ASR when tested on a different benchmark environment.

        E.g. trained on VisualWebArena, tested on WebArena.

        Args:
            episode_results_by_env: Mapping from env_name to results.

        Returns:
            List of TransferResult.
        """
        results: list[TransferResult] = []
        for env_name, results_list in episode_results_by_env.items():
            asr = compute_asr(results_list)
            r = TransferResult(
                dimension="cross_environment",
                variant=env_name,
                asr=asr,
                n_episodes=len(results_list),
                episode_results=results_list,
            )
            results.append(r)
            logger.info("Cross-environment [%s]: ASR=%.1%%", env_name, asr * 100)
        self.results.extend(results)
        return results

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return a structured summary of all transfer results.

        Returns:
            Nested dict: ``{dimension: {variant: {"asr": ...}}}``.
        """
        out: dict[str, dict[str, Any]] = {}
        for r in self.results:
            out.setdefault(r.dimension, {})[r.variant] = {
                "asr": r.asr,
                "n_episodes": r.n_episodes,
                "asr_vs_base": r.asr - self.base_asr,
            }
        return out

    def print_table(self) -> None:
        """Print a formatted ASCII table of transfer results."""
        print(f"\n{'='*60}")
        print(f"{'Transfer Evaluation Results':^60}")
        print(f"{'='*60}")
        print(f"Base ASR (training distribution): {self.base_asr:.1%}")
        print(f"{'-'*60}")
        print(f"{'Dimension':<25} {'Variant':<20} {'ASR':>8} {'Δ ASR':>8}")
        print(f"{'-'*60}")
        for r in self.results:
            delta = r.asr - self.base_asr
            sign = "+" if delta >= 0 else ""
            print(
                f"{r.dimension:<25} {r.variant:<20} "
                f"{r.asr:>7.1%} {sign}{delta:>7.1%}"
            )
        print(f"{'='*60}\n")
