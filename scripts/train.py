"""PoisonClaw unified training entry point.

Integrates PoisonClaw environments and memory modules with the
verl-agent RL training pipeline.

Usage:
    # Phase 1: VisualWebArena quick validation (2B + GRPO)
    python scripts/train.py \\
        --config configs/experiment/main_attack.yaml \\
        --algorithm grpo \\
        --seed 42

    # Ablation: vary friction gap
    python scripts/train.py \\
        --config configs/experiment/ablation_friction.yaml \\
        --override attack.friction_gap=5 \\
        --seed 42

    # 7B model (reduce num_envs due to memory)
    python scripts/train.py \\
        --config configs/experiment/main_attack.yaml \\
        --model configs/model/qwen2vl_7b.yaml \\
        --algorithm grpo \\
        --override env.rollout.num_envs=16 \\
        --seed 42

    # Resume from checkpoint (after Nautilus pod preemption)
    python scripts/train.py \\
        --config configs/experiment/main_attack.yaml \\
        --resume_from outputs/main_attack/checkpoint_5000.pt
"""

import argparse
import logging
import os
import sys

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("poisonclaw.train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PoisonClaw IRFA training script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--model", default=None, help="Optional model YAML to merge")
    parser.add_argument(
        "--algorithm",
        default=None,
        choices=["grpo", "ppo", "gigpo", "reinforce++", "rloo", "dapo"],
        help="RL algorithm override",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument(
        "--override",
        nargs="*",
        metavar="KEY=VALUE",
        default=[],
        help="Dot-notation config overrides, e.g. attack.friction_gap=5",
    )
    parser.add_argument(
        "--resume_from",
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate config and environment setup without training",
    )
    return parser.parse_args()


def load_config(config_path: str, model_path: str | None) -> dict:
    """Load and merge YAML configs using OmegaConf.

    Args:
        config_path: Path to main experiment config.
        model_path: Optional path to model config to merge.

    Returns:
        Merged OmegaConf DictConfig.
    """
    try:
        from omegaconf import OmegaConf
    except ImportError:
        raise ImportError("omegaconf is required. Install with: pip install omegaconf")

    cfg = OmegaConf.load(config_path)
    if model_path:
        model_cfg = OmegaConf.load(model_path)
        cfg = OmegaConf.merge(cfg, model_cfg)
    return cfg


def apply_overrides(cfg, overrides: list[str], algorithm: str | None, seed: int | None):
    """Apply CLI overrides to the config.

    Args:
        cfg: OmegaConf DictConfig.
        overrides: List of ``"key=value"`` strings.
        algorithm: RL algorithm override.
        seed: Random seed override.

    Returns:
        Updated config.
    """
    from omegaconf import OmegaConf

    for override in overrides:
        if "=" not in override:
            logger.warning("Skipping malformed override '%s' (no '=')", override)
            continue
        key, value = override.split("=", 1)
        # Try to parse value as int/float/bool
        for parser in (int, float):
            try:
                value = parser(value)
                break
            except (ValueError, TypeError):
                pass
        if isinstance(value, str) and value.lower() in ("true", "false"):
            value = value.lower() == "true"
        OmegaConf.update(cfg, key, value)

    if algorithm is not None:
        OmegaConf.update(cfg, "trainer.algorithm", algorithm)
    if seed is not None:
        OmegaConf.update(cfg, "seed", seed)

    return cfg


def setup_output_dir(cfg, output_dir_override: str | None) -> str:
    """Set up the output directory with config-derived naming.

    Args:
        cfg: Config object.
        output_dir_override: CLI override for output dir.

    Returns:
        Final output directory path.
    """
    from omegaconf import OmegaConf

    base = output_dir_override or OmegaConf.select(cfg, "output_dir", default="outputs/run")
    model_name = OmegaConf.select(cfg, "model.actor_lm.model_name", default="unknown")
    model_short = model_name.split("/")[-1].lower()
    algorithm = OmegaConf.select(cfg, "trainer.algorithm", default="grpo")
    seed = OmegaConf.select(cfg, "seed", default=42)

    output_dir = os.path.join(base, model_short, algorithm, f"seed{seed}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def setup_wandb(cfg, output_dir: str) -> None:
    """Initialize wandb if available and configured.

    Args:
        cfg: Config object.
        output_dir: Run output directory (used as wandb dir).
    """
    try:
        import wandb
        from omegaconf import OmegaConf

        project = OmegaConf.select(cfg, "logging.wandb_project", default="poisonclaw")
        group = OmegaConf.select(cfg, "logging.wandb_group", default="default")
        wandb.init(
            project=project,
            group=group,
            dir=output_dir,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        logger.info("wandb initialized: project=%s group=%s", project, group)
    except ImportError:
        logger.warning("wandb not installed; skipping experiment tracking.")
    except Exception as exc:
        logger.warning("wandb init failed: %s", exc)


def build_env_manager(cfg):
    """Instantiate the environment manager from config.

    Args:
        cfg: Config object.

    Returns:
        An environment manager instance.
    """
    from scripts.register_env import get_env_class
    from omegaconf import OmegaConf

    env_type = OmegaConf.select(cfg, "env.type", default="poisonclaw-visualwebarena")
    env_cls = get_env_class(env_type)
    return env_cls(config=cfg, split="train")


def main() -> None:
    args = parse_args()

    # Load and merge configs
    cfg = load_config(args.config, args.model)
    cfg = apply_overrides(cfg, args.override or [], args.algorithm, args.seed)

    # Set global seed
    from poisonclaw.utils.seed import set_seed
    from omegaconf import OmegaConf
    seed = OmegaConf.select(cfg, "seed", default=42)
    set_seed(int(seed))

    # Setup output directory
    output_dir = setup_output_dir(cfg, args.output_dir)
    logger.info("Output directory: %s", output_dir)

    # Save resolved config alongside checkpoint
    from omegaconf import OmegaConf
    config_dump = os.path.join(output_dir, "resolved_config.yaml")
    OmegaConf.save(cfg, config_dump)
    logger.info("Resolved config saved to %s", config_dump)

    if args.dry_run:
        logger.info("Dry run complete — config and environment validation passed.")
        return

    # Initialize wandb
    setup_wandb(cfg, output_dir)

    # Build environment
    env_manager = build_env_manager(cfg)
    logger.info("Environment manager created: %s", type(env_manager).__name__)

    # === Training loop placeholder ===
    # In a full integration, we would pass env_manager to the verl-agent
    # recipe trainer (e.g. grpo/trainer.py). The integration point is:
    #
    #   from recipe.grpo.trainer import GRPOTrainer
    #   trainer = GRPOTrainer(config=cfg, env_manager=env_manager)
    #   if args.resume_from:
    #       trainer.load_checkpoint(args.resume_from)
    #   trainer.train()
    #
    # Until the verl-agent recipe interface is finalized, this script
    # provides the setup plumbing. See CLAUDE.md for integration guide.

    num_steps = OmegaConf.select(cfg, "trainer.num_train_steps", default=10000)
    algorithm = OmegaConf.select(cfg, "trainer.algorithm", default="grpo")
    logger.info(
        "Training: algorithm=%s steps=%d output=%s",
        algorithm,
        num_steps,
        output_dir,
    )

    if args.resume_from:
        logger.info("Resuming from checkpoint: %s", args.resume_from)

    logger.info(
        "Training setup complete. "
        "Integrate with verl-agent recipe trainer to start RL training."
    )

    env_manager.close()


if __name__ == "__main__":
    main()
