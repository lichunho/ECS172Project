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
        method:
            'average'         - mean satisfaction across the group.
            'least_misery'    - min satisfaction across the group.
            'fairness_penalty'- convex blend of mean and min, minus a
              disagreement penalty (resolves the M5c design gap):

                  g_i = (1 - w) * mean_i + w * min_i - w * std_i

              where std_i is the per-item standard deviation of member scores.
              At w=0 this equals 'average'; as w->1 it favors the worst-off
              member and penalizes items the group disagrees on.
        fairness_weight: w in [0, 1], used only by 'fairness_penalty'.

    Returns:
        1-D array of length n_items.
    """
    sm = np.asarray(score_matrix, dtype="float64")
    if sm.ndim != 2:
        raise ValueError("score_matrix must be 2-D (n_users, n_items)")

    if method == "average":
        return sm.mean(axis=0)
    if method == "least_misery":
        return sm.min(axis=0)
    if method == "fairness_penalty":
        w = fairness_weight
        mean = sm.mean(axis=0)
        worst = sm.min(axis=0)
        disagreement = sm.std(axis=0)
        return (1.0 - w) * mean + w * worst - w * disagreement

    raise ValueError(f"unknown aggregation method: {method!r}")
