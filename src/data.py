"""BoardGameGeek data loading and preprocessing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_ratings(raw_dir: str | Path) -> pd.DataFrame:
    """Load raw user-game ratings. Returns DataFrame with columns: user_id, game_id, rating."""
    raise NotImplementedError


def load_games(raw_dir: str | Path) -> pd.DataFrame:
    """Load raw game metadata (player count range, playtime, genre, mechanics, complexity)."""
    raise NotImplementedError


def preprocess(
    ratings: pd.DataFrame,
    games: pd.DataFrame,
    min_ratings_per_user: int,
    min_ratings_per_game: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean, filter sparse users/games, normalize ratings, encode categorical features."""
    raise NotImplementedError
