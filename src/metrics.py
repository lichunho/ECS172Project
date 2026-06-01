"""Accuracy and fairness metrics."""

from __future__ import annotations

import numpy as np


def ndcg_at_k(ranked_items: np.ndarray, relevance: np.ndarray, k: int) -> float:
    ranked = np.asarray(ranked_items)
    if ranked.size == 0 or k <= 0:
        return 0.0
    ranked = ranked[:k]

    if isinstance(relevance, dict):
        rel = np.array([float(relevance.get(int(item), 0.0)) for item in ranked], dtype=float)
        ideal = np.sort(np.array(list(relevance.values()), dtype=float))[::-1][: len(rel)]
    else:
        rel_array = np.asarray(relevance, dtype=float)
        if rel_array.shape[0] == ranked.shape[0]:
            rel = rel_array[: len(ranked)]
            ideal = np.sort(rel_array)[::-1][: len(rel)]
        elif np.issubdtype(ranked.dtype, np.integer) and rel_array.shape[0] > int(ranked.max()):
            rel = rel_array[ranked.astype(int)]
            ideal = np.sort(rel_array)[::-1][: len(rel)]
        else:
            rel = rel_array[: len(ranked)]
            ideal = np.sort(rel_array)[::-1][: len(rel)]

    def _dcg(values: np.ndarray) -> float:
        if values.size == 0:
            return 0.0
        discounts = np.log2(np.arange(2, values.size + 2))
        gains = np.power(2.0, values) - 1.0
        return float(np.sum(gains / discounts))

    ideal_dcg = _dcg(ideal)
    if ideal_dcg == 0.0:
        return 0.0
    return _dcg(rel) / ideal_dcg


def precision_at_k(ranked_items: np.ndarray, relevant_set: set, k: int) -> float:
    ranked = np.asarray(ranked_items)[:k]
    if ranked.size == 0 or k <= 0:
        return 0.0
    hits = sum(item in relevant_set for item in ranked)
    return hits / float(min(k, ranked.size))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def satisfaction_variance(per_user_scores: np.ndarray) -> float:
    """Variance of per-user satisfaction within a group recommendation."""
    scores = np.asarray(per_user_scores, dtype=float)
    if scores.size == 0:
        return 0.0
    return float(np.var(scores))


def min_satisfaction(per_user_scores: np.ndarray) -> float:
    scores = np.asarray(per_user_scores, dtype=float)
    if scores.size == 0:
        return 0.0
    return float(np.min(scores))
