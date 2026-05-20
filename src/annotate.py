"""LLM-based annotation of board game context features.

Adds columns: party_friendliness, interaction_level, competitiveness.
"""

from __future__ import annotations

import pandas as pd


def annotate_games(games: pd.DataFrame) -> pd.DataFrame:
    """Call an LLM (or manual labels) to produce context-feature columns per game."""
    raise NotImplementedError
