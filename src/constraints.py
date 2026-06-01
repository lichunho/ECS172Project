"""Hard-constraint filtering (player count, playtime)."""

from __future__ import annotations

import pandas as pd


def filter_feasible(games: pd.DataFrame, constraints: dict) -> pd.DataFrame:
    """Drop games that violate `n_players` or `max_playtime_min` in `constraints`.

    Expected `games` columns: min_players, max_players, min_playtime, max_playtime.

    A game is dropped only when its metadata *positively contradicts* a
    constraint; games with missing bounds (pd.NA) are kept, so we never penalize
    a game for absent data. Constraint keys are optional.
    """
    mask = pd.Series(True, index=games.index)

    n_players = constraints.get("n_players")
    if n_players is not None:
        too_few = games["min_players"].notna() & (games["min_players"] > n_players)
        too_many = games["max_players"].notna() & (games["max_players"] < n_players)
        mask &= ~(too_few | too_many)

    max_playtime = constraints.get("max_playtime_min")
    if max_playtime is not None:
        # Infeasible if even the game's minimum playtime exceeds the budget.
        too_long = games["min_playtime"].notna() & (games["min_playtime"] > max_playtime)
        mask &= ~too_long

    return games[mask].copy()
