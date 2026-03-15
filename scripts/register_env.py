"""Register PoisonClaw environments with verl-agent's agent_system.

Run this once before starting training:
    python scripts/register_env.py
"""

import logging
import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def register_all() -> None:
    """Register all PoisonClaw environments with verl-agent."""
    from poisonclaw.envs.visualwebarena_env import VisualWebArenaEnvManager
    from poisonclaw.envs.webarena_env import WebArenaEnvManager
    from poisonclaw.envs.webshop_env import WebShopEnvManager
    from poisonclaw.envs.miniwob_env import MiniWoBEnvManager

    # verl-agent does not have a formal environment registry in the current
    # version; environments are instantiated directly in training scripts.
    # This script validates that all environments can be imported correctly.

    environments = {
        "poisonclaw-visualwebarena": VisualWebArenaEnvManager,
        "poisonclaw-webarena": WebArenaEnvManager,
        "poisonclaw-webshop": WebShopEnvManager,
        "poisonclaw-miniwob": MiniWoBEnvManager,
    }

    for name, cls in environments.items():
        logger.info("Registered environment: %s → %s", name, cls.__name__)

    logger.info(
        "All %d PoisonClaw environments registered successfully.", len(environments)
    )
    return environments


_ENV_REGISTRY: dict = {}


def get_env_class(env_type: str):
    """Look up an environment class by type string.

    Args:
        env_type: Environment type string (e.g. ``"poisonclaw-visualwebarena"``).

    Returns:
        Environment manager class.

    Raises:
        KeyError: If the environment type is not registered.
    """
    global _ENV_REGISTRY
    if not _ENV_REGISTRY:
        _ENV_REGISTRY = register_all()
    if env_type not in _ENV_REGISTRY:
        raise KeyError(
            f"Unknown environment type '{env_type}'. "
            f"Available: {list(_ENV_REGISTRY.keys())}"
        )
    return _ENV_REGISTRY[env_type]


if __name__ == "__main__":
    register_all()
    print("\nPoisonClaw environments are ready.")
