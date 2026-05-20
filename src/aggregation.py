"""Fairness-aware group score aggregation."""

from __future__ import annotations

import numpy as np


def aggregate(
    score_matrix: np.ndarray,
    method: str = "fairness_penalty",
    fairness_weight: float = 0.5,
) -> np.ndarray:
    """Aggregate per-user predicted scores into a per-item group score.

    Args:
        score_matrix: shape (n_users_in_group, n_items).
        method: 'average' | 'least_misery' | 'fairness_penalty'
            (avg satisfaction + weight * min satisfaction - disagreement penalty).
        fairness_weight: blends min satisfaction with the mean.

    Returns:
        1-D array of length n_items.
    """
    raise NotImplementedError
