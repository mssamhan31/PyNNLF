from __future__ import annotations

from copy import deepcopy
from typing import Any

import os
import random


SEED_KEYS = ("seed", "xgb_seed", "random_seed", "random_state")


def get_run_seed(config: dict[str, Any]) -> int | None:
    """Return the configured run seed, if one is configured."""
    reproducibility = config.get("reproducibility", {}) or {}
    seed = reproducibility.get("seed")
    if seed is None or seed == "":
        return None
    return int(seed)


def seed_everything(seed: int | None) -> None:
    """Seed common Python ML RNGs when they are available."""
    if seed is None:
        return

    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def apply_reproducibility_config(
    config: dict[str, Any],
    hyperparameter: dict[str, Any],
) -> tuple[dict[str, Any], int | None, list[str]]:
    """
    Apply run-level reproducibility settings to a copy of hyperparameters.

    Returns the effective hyperparameters, the run seed, and seed-like keys
    that were overridden to the run seed.
    """
    run_seed = get_run_seed(config)
    effective_hyperparameter = deepcopy(hyperparameter)
    overridden_keys: list[str] = []

    if run_seed is None:
        return effective_hyperparameter, None, overridden_keys

    seed_everything(run_seed)
    for key in SEED_KEYS:
        if key in effective_hyperparameter:
            effective_hyperparameter[key] = run_seed
            overridden_keys.append(key)

    return effective_hyperparameter, run_seed, overridden_keys
