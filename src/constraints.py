"""Hard-constraint filtering (player count, playtime)."""

from __future__ import annotations

import pandas as pd


def filter_feasible(games: pd.DataFrame, constraints: dict) -> pd.DataFrame:
    """Drop games that violate `n_players` or `max_playtime_min` in `constraints`.

    Expected `games` columns: min_players, max_players, min_playtime, max_playtime.
    """
    raise NotImplementedError
