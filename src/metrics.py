"""Accuracy and fairness metrics."""

from __future__ import annotations

import numpy as np


def ndcg_at_k(ranked_items: np.ndarray, relevance: np.ndarray, k: int) -> float:
    """Normalized DCG@k.

    `ranked_items` is unused positionally — `relevance` is the graded relevance
    already aligned to the ranked order (relevance[j] = relevance of the item at
    rank j). Kept in the signature for caller clarity.
    """
    rel = np.asarray(relevance, dtype="float64")[:k]
    if rel.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    dcg = float(((2.0 ** rel - 1.0) * discounts).sum())

    ideal = np.sort(np.asarray(relevance, dtype="float64"))[::-1][:k]
    idiscounts = 1.0 / np.log2(np.arange(2, ideal.size + 2))
    idcg = float(((2.0 ** ideal - 1.0) * idiscounts).sum())
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(ranked_items: np.ndarray, relevant_set: set, k: int) -> float:
    """Fraction of the top-k ranked items that are in `relevant_set`."""
    top = np.asarray(ranked_items)[:k]
    if top.size == 0:
        return 0.0
    hits = sum(1 for it in top if it in relevant_set)
    return hits / min(k, top.size)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def satisfaction_variance(per_user_scores: np.ndarray) -> float:
    """Variance of per-user satisfaction within a group recommendation."""
    return float(np.var(np.asarray(per_user_scores, dtype="float64")))


def min_satisfaction(per_user_scores: np.ndarray) -> float:
    return float(np.min(np.asarray(per_user_scores, dtype="float64")))
