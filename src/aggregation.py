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
    scores = np.asarray(score_matrix, dtype=float)
    if scores.ndim == 1:
        scores = scores[np.newaxis, :]

    method = method.lower().strip()
    if method == "average":
        return scores.mean(axis=0)
    if method == "least_misery":
        return scores.min(axis=0)
    if method != "fairness_penalty":
        raise ValueError(f"unknown aggregation method: {method}")

    mean_score = scores.mean(axis=0)
    min_score = scores.min(axis=0)
    disagreement = scores.std(axis=0)
    return (1.0 - fairness_weight) * mean_score + fairness_weight * min_score - fairness_weight * disagreement
