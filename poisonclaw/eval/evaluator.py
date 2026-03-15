"""Unified evaluation entry point for PoisonClaw.

Runs ASR, Clean SR, and CPR evaluation after each training checkpoint
and logs results to wandb (if available).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from poisonclaw.eval.metrics import (
    EvaluationMetrics,
    aggregate_seeds,
    compute_asr,
    compute_clean_sr,
    compute_cpr,
    compute_return_gap,
    compute_delta_clean_sr,
    compute_theoretical_return_gap,
)

logger = logging.getLogger(__name__)

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


@dataclass
class EvaluatorConfig:
    """Configuration for the PoisonClaw evaluator."""

    output_dir: str = "outputs/eval"
    log_wandb: bool = True
    save_json: bool = True
    save_csv: bool = True
    gamma: float = 0.99
    l_adv: int = 3           # adversarial path length for theoretical ΔG
    delta_l: int = 3         # friction gap for theoretical ΔG


class Evaluator:
    """Orchestrates full evaluation of a PoisonClaw checkpoint.

    Computes all primary metrics (ASR, Clean SR, CPR, ΔG, Δ Clean SR)
    and logs them to disk and optionally wandb.

    Args:
        config: EvaluatorConfig instance.
    """

    def __init__(self, config: EvaluatorConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        episode_results: list[dict[str, Any]],
        baseline_clean_sr: float = 0.0,
        step: Optional[int] = None,
        tag: str = "eval",
    ) -> EvaluationMetrics:
        """Run full evaluation on a batch of episode results.

        Args:
            episode_results: List of episode result dicts from rollout.
            baseline_clean_sr: Clean SR of a clean (unpoisoned) agent.
            step: Training step number (for logging).
            tag: Label for this evaluation (e.g. ``"val"``, ``"test"``).

        Returns:
            EvaluationMetrics instance.
        """
        asr = compute_asr(episode_results)
        clean_sr = compute_clean_sr(episode_results)
        cpr = compute_cpr(episode_results)
        delta_g_empirical = compute_return_gap(episode_results, gamma=self.config.gamma)
        delta_g_theory = compute_theoretical_return_gap(
            gamma=self.config.gamma,
            l_adv=self.config.l_adv,
            delta_l=self.config.delta_l,
        )
        delta_clean_sr = compute_delta_clean_sr(clean_sr, baseline_clean_sr)

        metrics = EvaluationMetrics(
            asr=asr,
            clean_sr=clean_sr,
            clean_sr_baseline=baseline_clean_sr,
            cpr=cpr,
            delta_g=delta_g_empirical,
            delta_g_theory=delta_g_theory,
            delta_clean_sr=delta_clean_sr,
            n_poisoned_episodes=sum(1 for r in episode_results if r.get("is_poisoned")),
            n_clean_episodes=sum(1 for r in episode_results if not r.get("is_poisoned")),
            n_cpr_episodes=sum(1 for r in episode_results if r.get("had_choice")),
        )

        logger.info("[%s step=%s] %s", tag, step, metrics)

        if self.config.save_json:
            self._save_json(metrics, tag=tag, step=step)
        if self.config.save_csv:
            self._append_csv(metrics, tag=tag, step=step)
        if self.config.log_wandb and _WANDB_AVAILABLE:
            self._log_wandb(metrics, tag=tag, step=step)

        return metrics

    def evaluate_multi_seed(
        self,
        per_seed_results: list[list[dict[str, Any]]],
        baseline_clean_sr: float = 0.0,
        step: Optional[int] = None,
        tag: str = "eval_multiseed",
    ) -> dict[str, dict[str, float]]:
        """Evaluate across multiple random seeds and aggregate results.

        Args:
            per_seed_results: One list of episode results per seed.
            baseline_clean_sr: Clean agent's baseline Clean SR.
            step: Training step for logging.
            tag: Evaluation tag.

        Returns:
            Dict of ``{metric_name: {"mean": ..., "std": ...}}``.
        """
        seed_metrics: list[EvaluationMetrics] = []
        for i, results in enumerate(per_seed_results):
            m = self.evaluate(
                results,
                baseline_clean_sr=baseline_clean_sr,
                step=step,
                tag=f"{tag}_seed{i}",
            )
            seed_metrics.append(m)

        aggregated = aggregate_seeds(seed_metrics)
        logger.info("[%s] Aggregated (n=%d seeds): %s", tag, len(seed_metrics), aggregated)
        return aggregated

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _save_json(
        self,
        metrics: EvaluationMetrics,
        tag: str,
        step: Optional[int],
    ) -> None:
        step_str = f"_step{step}" if step is not None else ""
        fname = self.output_dir / f"{tag}{step_str}_metrics.json"
        data = metrics.summary()
        data["step"] = step
        data["tag"] = tag
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.debug("Metrics saved to %s", fname)

    def _append_csv(
        self,
        metrics: EvaluationMetrics,
        tag: str,
        step: Optional[int],
    ) -> None:
        csv_path = self.output_dir / "results.csv"
        header = "step,tag,asr,clean_sr,cpr,delta_g,delta_g_theory,delta_clean_sr\n"
        row = (
            f"{step},{tag},{metrics.asr:.4f},{metrics.clean_sr:.4f},"
            f"{metrics.cpr:.4f},{metrics.delta_g:.4f},"
            f"{metrics.delta_g_theory:.4f},{metrics.delta_clean_sr:.4f}\n"
        )
        write_header = not csv_path.exists()
        with open(csv_path, "a", encoding="utf-8") as f:
            if write_header:
                f.write(header)
            f.write(row)

    def _log_wandb(
        self,
        metrics: EvaluationMetrics,
        tag: str,
        step: Optional[int],
    ) -> None:
        log_dict = {f"{tag}/{k}": v for k, v in metrics.summary().items()}
        if step is not None:
            wandb.log(log_dict, step=step)
        else:
            wandb.log(log_dict)
