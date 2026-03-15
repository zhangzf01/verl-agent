"""PoisonClaw unified evaluation entry point.

Usage:
    # Evaluate a single checkpoint
    python scripts/eval.py \\
        --checkpoint outputs/main_attack/qwen2vl_2b/grpo/seed42/best.pt \\
        --config configs/experiment/main_attack.yaml \\
        --env visualwebarena \\
        --split test

    # Transfer generalization evaluation
    python scripts/eval.py \\
        --checkpoint outputs/main_attack/qwen2vl_2b/grpo/seed42/best.pt \\
        --config configs/experiment/transfer.yaml \\
        --eval_type transfer \\
        --trigger_variants all

    # Batch evaluation of all checkpoints in a directory
    python scripts/eval.py \\
        --experiment_dir outputs/main_attack/ \\
        --eval_all
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("poisonclaw.eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PoisonClaw evaluation script")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint path")
    parser.add_argument("--config", required=True, help="Experiment config YAML")
    parser.add_argument(
        "--env",
        default="visualwebarena",
        choices=["visualwebarena", "webarena", "webshop"],
        help="Environment to evaluate on",
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "val", "test"]
    )
    parser.add_argument(
        "--eval_type",
        default="standard",
        choices=["standard", "transfer", "persistence"],
    )
    parser.add_argument(
        "--trigger_variants",
        default=None,
        help="Comma-separated visual trigger variants, or 'all'",
    )
    parser.add_argument(
        "--experiment_dir",
        default=None,
        help="Directory containing multiple checkpoints to evaluate",
    )
    parser.add_argument("--eval_all", action="store_true")
    parser.add_argument("--output_dir", default="outputs/eval")
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--baseline_clean_sr", type=float, default=0.0)
    return parser.parse_args()


def run_standard_eval(cfg, env_manager, evaluator, args) -> dict:
    """Run standard ASR + Clean SR + CPR evaluation.

    Args:
        cfg: Config object.
        env_manager: Environment manager.
        evaluator: Evaluator instance.
        args: CLI arguments.

    Returns:
        Metrics dict.
    """
    logger.info("Running standard evaluation (%d episodes)...", args.n_episodes)

    # Collect episodes (stub — replace with real rollout)
    episode_results = _collect_episodes(env_manager, n=args.n_episodes)

    metrics = evaluator.evaluate(
        episode_results=episode_results,
        baseline_clean_sr=args.baseline_clean_sr,
        tag=f"{args.eval_type}_{args.split}",
    )
    logger.info("Results: %s", metrics)
    return metrics.summary()


def run_transfer_eval(cfg, env_manager, args) -> dict:
    """Run transfer generalization evaluation.

    Args:
        cfg: Config object.
        env_manager: Environment manager.
        args: CLI arguments.

    Returns:
        Transfer evaluation summary dict.
    """
    from poisonclaw.eval.transfer_eval import TransferEvaluator

    logger.info("Running transfer evaluation...")
    base_episodes = _collect_episodes(env_manager, n=50)
    from poisonclaw.eval.metrics import compute_asr
    base_asr = compute_asr(base_episodes)

    transfer_eval = TransferEvaluator(base_asr=base_asr)

    # Visual variant evaluation
    if args.trigger_variants:
        variants = (
            ["color_shift", "size_large", "size_small", "position_bottom", "minimal"]
            if args.trigger_variants == "all"
            else args.trigger_variants.split(",")
        )
        results_by_variant = {
            v: _collect_episodes(env_manager, n=50, tag=v)
            for v in variants
        }
        transfer_eval.evaluate_visual_variants(results_by_variant)

    transfer_eval.print_table()
    return transfer_eval.summary()


def _collect_episodes(env_manager, n: int = 100, tag: str = "") -> list[dict]:
    """Stub episode collection — replace with real rollout.

    In production this would run the VLM agent through n episodes
    and collect outcome dicts. Here we return plausible placeholder data.

    Args:
        env_manager: Environment manager.
        n: Number of episodes to collect.
        tag: Optional tag for the collection.

    Returns:
        List of episode result dicts.
    """
    import random
    logger.warning(
        "Using stub episode collection (n=%d, tag='%s'). "
        "Replace _collect_episodes() with real VLM rollout.",
        n,
        tag,
    )
    results = []
    for i in range(n):
        is_poisoned = random.random() < 0.5
        results.append({
            "episode_id": i,
            "is_poisoned": is_poisoned,
            "won": random.random() < (0.6 if not is_poisoned else 0.7),
            "trigger_clicked": is_poisoned and random.random() < 0.8,
            "had_choice": is_poisoned,
            "chose_trigger": is_poisoned and random.random() < 0.8,
            "path_type": "adversarial" if (is_poisoned and random.random() < 0.8) else "organic",
            "discounted_return": random.uniform(0.5, 1.0),
            "tag": tag,
        })
    return results


def main() -> None:
    args = parse_args()

    # Load config
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(args.config)
    except ImportError:
        raise ImportError("omegaconf required: pip install omegaconf")

    # Set seed
    from poisonclaw.utils.seed import set_seed
    from omegaconf import OmegaConf as OC
    set_seed(int(OC.select(cfg, "seed", default=42)))

    # Build evaluator
    from poisonclaw.eval.evaluator import Evaluator, EvaluatorConfig
    from poisonclaw.attack.poisoner import WebsitePoisoner

    eval_cfg = EvaluatorConfig(
        output_dir=args.output_dir,
        log_wandb=False,  # disable wandb for eval-only runs by default
        gamma=float(OC.select(cfg, "trainer.discount_factor", default=0.99)),
        l_adv=int(OC.select(cfg, "attack.friction_gap", default=3)),
        delta_l=int(OC.select(cfg, "attack.friction_gap", default=3)),
    )
    evaluator = Evaluator(eval_cfg)

    # Build environment
    from scripts.register_env import get_env_class
    env_type = f"poisonclaw-{args.env}"
    env_cls = get_env_class(env_type)
    env_manager = env_cls(config=cfg, split=args.split)

    if args.eval_type == "transfer":
        results = run_transfer_eval(cfg, env_manager, args)
    else:
        results = run_standard_eval(cfg, env_manager, evaluator, args)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.eval_type}_{args.split}_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_path)

    env_manager.close()


if __name__ == "__main__":
    main()
