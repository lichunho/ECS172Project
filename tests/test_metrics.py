"""Unit tests for src/metrics.py — pure in-memory, no data/ files, no network."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import (  # noqa: E402
    min_satisfaction,
    ndcg_at_k,
    precision_at_k,
    rmse,
    satisfaction_variance,
)


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------

class TestNdcgAtK:
    def test_hand_computed(self):
        # relevance already aligned to ranked order: [3, 2, 3, 0, 1]
        # k=3 means we look at rel = [3, 2, 3]
        # DCG  = (2^3-1)/log2(2) + (2^2-1)/log2(3) + (2^3-1)/log2(4)
        #      = 7/1 + 3/log2(3) + 7/2
        # IDCG = sorted desc = [3,3,2] (taking best 3 from all 5)
        #      = (2^3-1)/log2(2) + (2^3-1)/log2(3) + (2^2-1)/log2(4)
        relevance = np.array([3.0, 2.0, 3.0, 0.0, 1.0])
        k = 3

        rel_k = relevance[:k]
        discounts = 1.0 / np.log2(np.arange(2, k + 2))
        dcg = float(((2.0 ** rel_k - 1.0) * discounts).sum())

        ideal = np.sort(relevance)[::-1][:k]
        idiscounts = 1.0 / np.log2(np.arange(2, k + 2))
        idcg = float(((2.0 ** ideal - 1.0) * idiscounts).sum())

        expected = dcg / idcg
        result = ndcg_at_k(np.arange(5), relevance, k)
        assert abs(result - expected) < 1e-9

    def test_perfect_ranking_gives_one(self):
        # Descending relevance is already the ideal order → NDCG = 1.0
        relevance = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = ndcg_at_k(np.arange(5), relevance, k=5)
        assert abs(result - 1.0) < 1e-9

    def test_all_zero_relevance_gives_zero(self):
        # IDCG = 0 → function returns 0.0, not NaN or error
        relevance = np.array([0.0, 0.0, 0.0])
        result = ndcg_at_k(np.arange(3), relevance, k=3)
        assert result == 0.0

    def test_k_larger_than_list(self):
        # k > len(relevance) — should not crash; truncated to actual length
        relevance = np.array([3.0, 1.0])
        result = ndcg_at_k(np.arange(2), relevance, k=10)
        assert 0.0 <= result <= 1.0

    def test_single_relevant_item_at_top(self):
        # Only one relevant item, already at rank 0 → NDCG = 1.0
        relevance = np.array([1.0, 0.0, 0.0])
        result = ndcg_at_k(np.arange(3), relevance, k=3)
        assert abs(result - 1.0) < 1e-9

    def test_single_relevant_item_at_bottom(self):
        # Relevant item is last → DCG < IDCG → NDCG < 1.0
        relevance = np.array([0.0, 0.0, 1.0])
        result = ndcg_at_k(np.arange(3), relevance, k=3)
        assert 0.0 < result < 1.0

    def test_ranked_items_arg_is_unused(self):
        # The function uses only relevance; ranked_items is just for caller clarity
        relevance = np.array([2.0, 1.0, 0.0])
        r1 = ndcg_at_k(np.array([10, 20, 30]), relevance, k=3)
        r2 = ndcg_at_k(np.array([99, 99, 99]), relevance, k=3)
        assert abs(r1 - r2) < 1e-12


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------

class TestPrecisionAtK:
    def test_all_relevant(self):
        ranked = np.array([1, 2, 3, 4, 5])
        relevant = {1, 2, 3, 4, 5}
        assert precision_at_k(ranked, relevant, k=5) == 1.0

    def test_none_relevant(self):
        ranked = np.array([1, 2, 3])
        relevant = {10, 20}
        assert precision_at_k(ranked, relevant, k=3) == 0.0

    def test_partial_hits(self):
        ranked = np.array([1, 2, 3, 4, 5])
        relevant = {2, 4}
        # top-5: items 1,2,3,4,5 — 2 hits → 2/5
        result = precision_at_k(ranked, relevant, k=5)
        assert abs(result - 2 / 5) < 1e-9

    def test_k_smaller_than_list(self):
        ranked = np.array([10, 20, 30, 40, 50])
        relevant = {30, 40, 50}  # all are in positions 2-4
        # top-2 only: items 10, 20 → 0 hits
        assert precision_at_k(ranked, relevant, k=2) == 0.0

    def test_k_one_hit(self):
        ranked = np.array([7, 1, 2])
        relevant = {7}
        assert precision_at_k(ranked, relevant, k=1) == 1.0

    def test_k_one_miss(self):
        ranked = np.array([7, 1, 2])
        relevant = {1}
        assert precision_at_k(ranked, relevant, k=1) == 0.0

    def test_empty_ranked_list(self):
        assert precision_at_k(np.array([]), {1}, k=5) == 0.0


# ---------------------------------------------------------------------------
# rmse
# ---------------------------------------------------------------------------

class TestRmse:
    def test_hand_computed(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        # errors: [-1, 0, 1] → squared: [1, 0, 1] → mean: 2/3 → sqrt: sqrt(2/3)
        expected = math.sqrt(2.0 / 3.0)
        assert abs(rmse(y_true, y_pred) - expected) < 1e-9

    def test_perfect_predictions(self):
        y = np.array([3.0, 5.0, 7.0])
        assert rmse(y, y) == 0.0

    def test_constant_error(self):
        y_true = np.array([1.0, 1.0, 1.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        # all errors = 1 → RMSE = 1
        assert abs(rmse(y_true, y_pred) - 1.0) < 1e-9

    def test_accepts_lists(self):
        # should work with plain Python lists, not just numpy arrays
        result = rmse([1.0, 3.0], [2.0, 2.0])
        assert abs(result - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# satisfaction_variance
# ---------------------------------------------------------------------------

class TestSatisfactionVariance:
    def test_matches_numpy_var(self):
        scores = np.array([0.3, 0.5, 0.7, 0.2, 0.9])
        expected = float(np.var(scores))
        assert abs(satisfaction_variance(scores) - expected) < 1e-12

    def test_uniform_scores_give_zero(self):
        scores = np.array([0.5, 0.5, 0.5])
        assert satisfaction_variance(scores) == 0.0

    def test_accepts_list(self):
        scores = [1.0, 2.0, 3.0]
        expected = float(np.var(scores))
        assert abs(satisfaction_variance(scores) - expected) < 1e-12

    def test_single_element(self):
        assert satisfaction_variance(np.array([0.7])) == 0.0


# ---------------------------------------------------------------------------
# min_satisfaction
# ---------------------------------------------------------------------------

class TestMinSatisfaction:
    def test_matches_numpy_min(self):
        scores = np.array([0.8, 0.3, 0.6, 0.1, 0.9])
        expected = float(np.min(scores))
        assert abs(min_satisfaction(scores) - expected) < 1e-12

    def test_all_equal(self):
        scores = np.array([0.4, 0.4, 0.4])
        assert abs(min_satisfaction(scores) - 0.4) < 1e-12

    def test_accepts_list(self):
        scores = [0.2, 0.8, 0.5]
        assert abs(min_satisfaction(scores) - 0.2) < 1e-12

    def test_negative_values(self):
        scores = np.array([-0.5, 0.0, 0.5])
        assert abs(min_satisfaction(scores) - (-0.5)) < 1e-12
