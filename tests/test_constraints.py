"""Unit tests for src/constraints.filter_feasible.

Pure in-memory pandas; no data/ files, no network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.constraints import filter_feasible  # noqa: E402


def _make_games() -> pd.DataFrame:
    """A small frame with five games covering every boundary case."""
    return pd.DataFrame(
        {
            "game_id": [1, 2, 3, 4, 5],
            "name": ["ok", "too_few", "too_many", "too_long", "na_bounds"],
            # player counts
            "min_players": [2,    5,    2,    2,    pd.NA],
            "max_players": [6,    8,    3,    6,    pd.NA],
            # playtime (minutes)
            "min_playtime": [30,  30,   30,  180,   pd.NA],
            "max_playtime": [60,  60,   60,  240,   pd.NA],
        }
    )


# constraints: 4-player group, 90-minute budget
_CONSTRAINTS = {"n_players": 4, "max_playtime_min": 90}


class TestFilterFeasible:

    def test_feasible_game_survives(self):
        games = _make_games()
        result = filter_feasible(games, _CONSTRAINTS)
        assert 1 in result["game_id"].values

    def test_too_few_players_dropped(self):
        # min_players=5 > n_players=4
        games = _make_games()
        result = filter_feasible(games, _CONSTRAINTS)
        assert 2 not in result["game_id"].values

    def test_too_many_players_dropped(self):
        # max_players=3 < n_players=4
        games = _make_games()
        result = filter_feasible(games, _CONSTRAINTS)
        assert 3 not in result["game_id"].values

    def test_too_long_dropped(self):
        # min_playtime=180 > max_playtime_min=90
        games = _make_games()
        result = filter_feasible(games, _CONSTRAINTS)
        assert 4 not in result["game_id"].values

    def test_na_bounds_game_survives(self):
        # Missing metadata must never drop a game
        games = _make_games()
        result = filter_feasible(games, _CONSTRAINTS)
        assert 5 in result["game_id"].values

    def test_only_feasible_and_na_survive(self):
        games = _make_games()
        result = filter_feasible(games, _CONSTRAINTS)
        assert set(result["game_id"].values) == {1, 5}

    def test_result_is_copy(self):
        # Modifying the result must not affect the original
        games = _make_games()
        result = filter_feasible(games, _CONSTRAINTS)
        result.loc[result.index[0], "name"] = "CHANGED"
        assert games.loc[games["game_id"] == 1, "name"].iloc[0] == "ok"

    def test_omit_n_players_skips_player_filter(self):
        # Without n_players, too_few and too_many are not dropped
        games = _make_games()
        result = filter_feasible(games, {"max_playtime_min": 90})
        assert 2 in result["game_id"].values
        assert 3 in result["game_id"].values
        # too_long still dropped
        assert 4 not in result["game_id"].values

    def test_omit_max_playtime_skips_time_filter(self):
        # Without max_playtime_min, too_long is not dropped
        games = _make_games()
        result = filter_feasible(games, {"n_players": 4})
        assert 4 in result["game_id"].values
        # player violations still dropped
        assert 2 not in result["game_id"].values
        assert 3 not in result["game_id"].values

    def test_empty_constraints_returns_all(self):
        games = _make_games()
        result = filter_feasible(games, {})
        assert len(result) == len(games)

    def test_partial_na_player_bounds(self):
        # Only max_players is NA: min_players check still fires; max_players NA means
        # we can't say max < n_players, so that side is kept.
        df = pd.DataFrame(
            {
                "game_id": [10, 11],
                "name": ["min_ok_max_na", "min_bad_max_na"],
                "min_players": [2,   5],
                "max_players": [pd.NA, pd.NA],
                "min_playtime": [30, 30],
                "max_playtime": [60, 60],
            }
        )
        result = filter_feasible(df, {"n_players": 4})
        assert 10 in result["game_id"].values   # min_players=2 <= 4, max NA -> keep
        assert 11 not in result["game_id"].values  # min_players=5 > 4 -> drop
