"""Global random seed management for PoisonClaw.

Ensures full reproducibility across Python, NumPy, PyTorch, and CUDA.
All randomness in the PoisonClaw codebase must go through this module.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Module-level RNG for non-torch use
_rng: Optional[random.Random] = None


def set_seed(seed: int) -> None:
    """Set the global random seed for all relevant libraries.

    Args:
        seed: Integer seed value. Must be in [0, 2^32 - 1].
    """
    global _rng
    seed = int(seed)

    # Python stdlib
    random.seed(seed)
    _rng = random.Random(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU + GPU
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic CUDA operations (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Environment variable for libraries that read it (e.g. Transformers)
    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.info("Global seed set to %d.", seed)


def get_rng() -> random.Random:
    """Return the module-level Random instance.

    Raises:
        RuntimeError: If ``set_seed`` has not been called.

    Returns:
        random.Random instance seeded via set_seed().
    """
    if _rng is None:
        raise RuntimeError("Call set_seed() before using get_rng().")
    return _rng


def derive_seed(base_seed: int, *tags: str | int) -> int:
    """Derive a child seed from a base seed and string/int tags.

    Useful for giving each experiment run, worker, or environment
    a unique but deterministic seed.

    Args:
        base_seed: Parent seed value.
        *tags: Additional identifiers (e.g. ``"env_0"``, ``42``).

    Returns:
        Derived integer seed in [0, 2^31 - 1].
    """
    import hashlib

    payload = str(base_seed) + "".join(str(t) for t in tags)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return int(digest[:8], 16) % (2**31)
