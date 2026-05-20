"""Accuracy and fairness metrics."""

from __future__ import annotations

import numpy as np


def ndcg_at_k(ranked_items: np.ndarray, relevance: np.ndarray, k: int) -> float:
    raise NotImplementedError


def precision_at_k(ranked_items: np.ndarray, relevant_set: set, k: int) -> float:
    raise NotImplementedError


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    raise NotImplementedError


def satisfaction_variance(per_user_scores: np.ndarray) -> float:
    """Variance of per-user satisfaction within a group recommendation."""
    raise NotImplementedError


def min_satisfaction(per_user_scores: np.ndarray) -> float:
    raise NotImplementedError
