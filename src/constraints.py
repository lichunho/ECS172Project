"""Hard-constraint filtering (player count, playtime)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def filter_feasible(games: pd.DataFrame, constraints: dict) -> pd.DataFrame:
    """Drop games that violate `n_players` or `max_playtime_min` in `constraints`.

    Expected `games` columns: min_players, max_players, min_playtime, max_playtime.
    """
    if games.empty:
        return games.copy()

    def _series(name: str) -> pd.Series:
        if name not in games.columns:
            return pd.Series(np.nan, index=games.index)
        return pd.to_numeric(games[name], errors="coerce")

    feasible = games.copy()

    n_players = constraints.get("n_players")
    if n_players is not None:
        min_players = _series("min_players")
        max_players = _series("max_players")
        mask = (min_players.isna() | (min_players <= n_players)) & (
            max_players.isna() | (max_players >= n_players)
        )
        feasible = feasible.loc[mask]

    max_playtime = constraints.get("max_playtime_min")
    if max_playtime is None:
        max_playtime = constraints.get("available_time_min")
    if max_playtime is not None:
        min_playtime = _series("min_playtime").loc[feasible.index]
        max_game_playtime = _series("max_playtime").loc[feasible.index]
        mask = (min_playtime.isna() | (min_playtime <= max_playtime)) & (
            max_game_playtime.isna() | (max_game_playtime <= max_playtime)
        )
        feasible = feasible.loc[mask]

    return feasible.reset_index(drop=True)
