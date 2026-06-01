"""Unit tests for src/aggregation.aggregate.

Pure in-memory numpy; no data/ files, no network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.aggregation import aggregate  # noqa: E402

# Hand-built 3-user × 3-item score matrix.
#
#   item 0  item 1  item 2
#   4.0     6.0     8.0     user 0
#   2.0     6.0     4.0     user 1
#   6.0     6.0    10.0     user 2
#
# item 0: mean=4, min=2, std=std([4,2,6])
# item 1: mean=6, min=6, std=0          (unanimous)
# item 2: mean=22/3≈7.333, min=4, std=std([8,4,10])
_SM = np.array(
    [
        [4.0, 6.0,  8.0],
        [2.0, 6.0,  4.0],
        [6.0, 6.0, 10.0],
    ],
    dtype="float64",
)


class TestAverage:

    def test_equals_numpy_mean(self):
        result = aggregate(_SM, method="average")
        assert_allclose(result, _SM.mean(axis=0), rtol=1e-9)

    def test_shape(self):
        result = aggregate(_SM, method="average")
        assert result.shape == (3,)

    def test_values(self):
        assert_allclose(aggregate(_SM, method="average"), [4.0, 6.0, 22.0 / 3.0], rtol=1e-9)


class TestLeastMisery:

    def test_equals_numpy_min(self):
        result = aggregate(_SM, method="least_misery")
        assert_allclose(result, _SM.min(axis=0), rtol=1e-9)

    def test_values(self):
        assert_allclose(aggregate(_SM, method="least_misery"), [2.0, 6.0, 4.0])


class TestFairnessPenalty:

    def test_weight_zero_equals_average(self):
        """At w=0 the formula collapses to plain mean."""
        fp = aggregate(_SM, method="fairness_penalty", fairness_weight=0.0)
        avg = aggregate(_SM, method="average")
        assert_allclose(fp, avg, rtol=1e-9)

    def test_weight_zero_explicit_values(self):
        fp = aggregate(_SM, method="fairness_penalty", fairness_weight=0.0)
        assert_allclose(fp, _SM.mean(axis=0), rtol=1e-9)

    def test_high_disagreement_item_penalised_vs_unanimous(self):
        """A high-disagreement item scores *lower* than a unanimous item with the same
        mean when fairness_weight > 0."""
        # Craft a 2×2 matrix: item 0 unanimous (mean=6), item 1 spread (mean=6).
        sm = np.array([[6.0, 3.0], [6.0, 9.0]], dtype="float64")
        # Both items have mean=6; item 1 has higher std -> should score lower.
        result = aggregate(sm, method="fairness_penalty", fairness_weight=0.5)
        assert result[0] > result[1], (
            f"unanimous item ({result[0]:.4f}) should beat "
            f"disagreement item ({result[1]:.4f})"
        )

    def test_fairness_penalty_exact_formula(self):
        # item 0 unanimous: [7, 7] -> mean=7, min=7, std=0
        # item 1 spread:    [3, 9] -> mean=6, min=3, std=3
        # w=0.5:
        #   g0 = 0.5*7 + 0.5*7 - 0.5*0 = 7.0
        #   g1 = 0.5*6 + 0.5*3 - 0.5*3 = 3.0
        sm = np.array([[7.0, 3.0], [7.0, 9.0]], dtype="float64")
        result = aggregate(sm, method="fairness_penalty", fairness_weight=0.5)
        assert_allclose(result, [7.0, 3.0], rtol=1e-9)

    def test_raising_fairness_weight_lowers_high_disagreement_item(self):
        """Increasing w should move a high-disagreement item further below a unanimous one."""
        sm = np.array([[6.0, 2.0], [6.0, 8.0]], dtype="float64")
        low_w = aggregate(sm, method="fairness_penalty", fairness_weight=0.1)
        high_w = aggregate(sm, method="fairness_penalty", fairness_weight=0.9)
        gap_low = low_w[0] - low_w[1]
        gap_high = high_w[0] - high_w[1]
        assert gap_high > gap_low, (
            f"gap at w=0.9 ({gap_high:.4f}) should exceed gap at w=0.1 ({gap_low:.4f})"
        )

    def test_default_method_is_fairness_penalty(self):
        """Calling aggregate without method= should use fairness_penalty."""
        result_default = aggregate(_SM)
        result_explicit = aggregate(_SM, method="fairness_penalty", fairness_weight=0.5)
        assert_allclose(result_default, result_explicit, rtol=1e-9)


class TestUnknownMethod:

    def test_raises_value_error(self):
        with pytest.raises(ValueError, match="unknown aggregation method"):
            aggregate(_SM, method="bogus_method")


class TestInputValidation:

    def test_1d_input_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            aggregate(np.array([1.0, 2.0, 3.0]))

    def test_accepts_list_input(self):
        """score_matrix may be a list of lists; np.asarray conversion must work."""
        result = aggregate([[1.0, 2.0], [3.0, 4.0]], method="average")
        assert_allclose(result, [2.0, 3.0])
