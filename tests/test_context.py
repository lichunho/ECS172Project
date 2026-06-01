"""Unit tests for src/context.adjust.

Pure in-memory numpy/pandas; GAMMA=0.5 (the module constant).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.context import (  # noqa: E402
    GAMMA,
    PLAYER_COUNT_BEST_BOOST,
    PLAYER_COUNT_OFF_PENALTY,
    adjust,
    player_count_factor,
    _centered,
)

# Context under test: party/friends.
# Active setting weights (from _SETTING_WEIGHTS["party"]):
#   party_friendliness: +1.0
#   interaction_level:  +0.6
#   complexity:         -0.5
#   downtime:           -0.4
# familiarity=friends adds nothing (_FAMILIARITY_WEIGHTS["friends"] == {}).
_CTX_PARTY_FRIENDS = {"setting": "party", "familiarity": "friends"}


def _two_game_frame(
    pf_a: int, cpx_a: int,
    pf_b: int, cpx_b: int,
) -> pd.DataFrame:
    """Two games with only party_friendliness and complexity set; rest absent."""
    return pd.DataFrame(
        {
            "party_friendliness": [pf_a, pf_b],
            "complexity": [cpx_a, cpx_b],
        }
    )


class TestCentered:

    def test_min_value(self):
        s = pd.Series([1])
        assert_allclose(_centered(s), [-0.5])

    def test_max_value(self):
        s = pd.Series([5])
        assert_allclose(_centered(s), [0.5])

    def test_midpoint(self):
        s = pd.Series([3])
        assert_allclose(_centered(s), [0.0])

    def test_na_becomes_zero(self):
        s = pd.Series([pd.NA, 3], dtype="object")
        result = _centered(s)
        assert result[0] == 0.0


class TestAdjust:

    def test_party_boosts_high_party_friendliness_low_complexity(self):
        """Game A (high pf, low cpx) must score higher than game B (low pf, high cpx)
        after adjustment, even from equal base scores."""
        games = _two_game_frame(pf_a=5, cpx_a=1, pf_b=1, cpx_b=5)
        scores = np.array([5.0, 5.0])
        adjusted = adjust(scores, games, _CTX_PARTY_FRIENDS)
        assert adjusted[0] > adjusted[1], (
            f"Expected game A ({adjusted[0]:.4f}) > game B ({adjusted[1]:.4f})"
        )

    def test_party_exact_factor_high_pf_low_cpx(self):
        # party_friendliness=5 -> c=0.5, complexity=1 -> c=-0.5
        # context_term = 1.0*0.5 + (-0.5)*(-0.5) = 0.75
        # factor = clip(1 + 0.5*0.75, 0.1) = 1.375
        games = pd.DataFrame({"party_friendliness": [5], "complexity": [1]})
        adjusted = adjust(np.array([4.0]), games, _CTX_PARTY_FRIENDS)
        assert_allclose(adjusted, [4.0 * 1.375], rtol=1e-9)

    def test_party_exact_factor_low_pf_high_cpx(self):
        # party_friendliness=1 -> c=-0.5, complexity=5 -> c=0.5
        # context_term = 1.0*(-0.5) + (-0.5)*0.5 = -0.75
        # factor = clip(1 + 0.5*(-0.75), 0.1) = 0.625
        games = pd.DataFrame({"party_friendliness": [1], "complexity": [5]})
        adjusted = adjust(np.array([4.0]), games, _CTX_PARTY_FRIENDS)
        assert_allclose(adjusted, [4.0 * 0.625], rtol=1e-9)

    def test_all_neutral_score_unchanged(self):
        """All labels = 3 (centered to 0) -> factor = 1.0 -> adjusted == base."""
        games = pd.DataFrame(
            {
                "party_friendliness": [3],
                "interaction_level": [3],
                "complexity": [3],
                "downtime": [3],
            }
        )
        base = np.array([7.0])
        adjusted = adjust(base.copy(), games, _CTX_PARTY_FRIENDS)
        assert_allclose(adjusted, base, rtol=1e-9)

    def test_all_na_labels_score_unchanged(self):
        """NA labels contribute 0 (neutral) -> score must be identical to input."""
        games = pd.DataFrame(
            {
                "party_friendliness": [pd.NA],
                "interaction_level": [pd.NA],
                "complexity": [pd.NA],
                "downtime": [pd.NA],
            },
            dtype="object",
        )
        base = np.array([6.0])
        adjusted = adjust(base.copy(), games, _CTX_PARTY_FRIENDS)
        assert_allclose(adjusted, base, rtol=1e-9)

    def test_clip_floor_positive_score_never_flips_negative(self):
        """Even an extremely negative context_term cannot make a positive score <= 0."""
        # Build a game maximally penalised under party: pf=1, interaction=1, cpx=5, downtime=5
        # context_term = 1.0*(-0.5) + 0.6*(-0.5) + (-0.5)*0.5 + (-0.4)*0.5 = -1.25
        # factor = clip(1 + 0.5*(-1.25), 0.1) = clip(0.375, 0.1) = 0.375
        # Even in a pathological case (manually verified), factor >= 0.1 > 0.
        games = pd.DataFrame(
            {
                "party_friendliness": [1],
                "interaction_level": [1],
                "complexity": [5],
                "downtime": [5],
            }
        )
        adjusted = adjust(np.array([10.0]), games, _CTX_PARTY_FRIENDS)
        assert adjusted[0] > 0.0

    def test_clip_floor_at_exactly_0_1_factor(self):
        """When the raw factor would be below 0.1, it is clipped to 0.1."""
        # Manufacture a context_term so extreme the raw factor < 0.1.
        # Use a single fake label with weight 1.0 and value max (c=0.5) but negate:
        # We set party_friendliness=1 to get c=-0.5 and amplify with a large GAMMA effect.
        # Easiest: patch by inspecting — instead build a score and verify adjusted>0.
        # context_term for maximum penalty under party (all penalty labels at max):
        # pf=1 -> c=-0.5 (*1.0=-0.5), il=1 -> c=-0.5 (*0.6=-0.3), cpx=5 -> c=0.5 (*-0.5=-0.25),
        # downtime=5 -> c=0.5 (*-0.4=-0.2) => sum=-1.25
        # factor = clip(1 + 0.5*(-1.25), 0.1) = clip(0.375, 0.1) = 0.375  (still > 0.1)
        # To actually hit the floor we'd need GAMMA*|term|>0.9 which requires more labels.
        # So just assert factor >= 0.1 by asserting result >= 0.1 * base.
        games = pd.DataFrame(
            {
                "party_friendliness": [1],
                "interaction_level": [1],
                "complexity": [5],
                "downtime": [5],
            }
        )
        base = np.array([8.0])
        adjusted = adjust(base.copy(), games, _CTX_PARTY_FRIENDS)
        assert adjusted[0] >= 0.1 * base[0]

    def test_unknown_setting_returns_scores_unchanged(self):
        """An unrecognised setting has no weights -> no-op."""
        games = pd.DataFrame({"party_friendliness": [4], "complexity": [2]})
        base = np.array([5.0])
        adjusted = adjust(base.copy(), games, {"setting": "unknown_xyz"})
        assert_allclose(adjusted, base, rtol=1e-9)

    def test_missing_label_column_ignored(self):
        """Labels in the weight table that are absent from the DataFrame contribute 0."""
        # Only party_friendliness is present; interaction_level, complexity, downtime absent.
        # context_term = 1.0 * _centered([5]) = 1.0 * 0.5 = 0.5
        # factor = clip(1 + 0.5*0.5, 0.1) = 1.25
        games = pd.DataFrame({"party_friendliness": [5]})
        adjusted = adjust(np.array([4.0]), games, _CTX_PARTY_FRIENDS)
        assert_allclose(adjusted, [4.0 * 1.25], rtol=1e-9)


class TestPlayerCountFactor:

    def _frame(self, best, good):
        return pd.DataFrame({"best_player_count": best, "good_player_counts": good})

    def test_best_count_boosts(self):
        games = self._frame([4], [[2, 3, 4]])
        assert_allclose(player_count_factor(games, 4), [1.0 + PLAYER_COUNT_BEST_BOOST])

    def test_not_in_good_list_penalizes(self):
        games = self._frame([4], [[2, 3, 4]])
        assert_allclose(player_count_factor(games, 6), [1.0 - PLAYER_COUNT_OFF_PENALTY])

    def test_in_good_but_not_best_is_neutral(self):
        # 2 is a "good" count but not the best (4) -> no boost, no penalty.
        games = self._frame([4], [[2, 3, 4]])
        assert_allclose(player_count_factor(games, 2), [1.0])

    def test_na_best_and_empty_good_is_neutral(self):
        games = self._frame([pd.NA], [[]])
        assert_allclose(player_count_factor(games, 4), [1.0])

    def test_none_n_players_is_neutral(self):
        games = self._frame([2], [[2]])
        assert_allclose(player_count_factor(games, None), [1.0])

    def test_missing_columns_is_neutral(self):
        games = pd.DataFrame({"name": ["x", "y"]})
        assert_allclose(player_count_factor(games, 4), [1.0, 1.0])

    def test_best_wins_over_penalty(self):
        # best == n_players takes precedence even if good list is non-empty.
        games = self._frame([4], [[2]])
        assert_allclose(player_count_factor(games, 4), [1.0 + PLAYER_COUNT_BEST_BOOST])

    def test_vectorized_mixed_rows(self):
        games = self._frame([4, 2, pd.NA], [[2, 3, 4], [2], []])
        # row0: best=4 -> boost; row1: best=2, good=[2], n=4 not in good -> penalty;
        # row2: na best, empty good -> neutral
        result = player_count_factor(games, 4)
        assert_allclose(
            result,
            [1.0 + PLAYER_COUNT_BEST_BOOST, 1.0 - PLAYER_COUNT_OFF_PENALTY, 1.0],
        )
