"""Group recommendation baselines."""

from __future__ import annotations

import numpy as np


def average_baseline(score_matrix: np.ndarray) -> np.ndarray:
    """Mean of per-user scores across the group."""
    raise NotImplementedError


def least_misery_baseline(score_matrix: np.ndarray) -> np.ndarray:
    """Min of per-user scores across the group."""
    raise NotImplementedError


def random_baseline(n_items: int, seed: int | None = None) -> np.ndarray:
    """Uniform random scores over items."""
    raise NotImplementedError
