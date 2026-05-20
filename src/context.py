"""Context-aware score adjustment based on social setting and group familiarity."""

from __future__ import annotations

import numpy as np
import pandas as pd


def adjust(scores: np.ndarray, games: pd.DataFrame, context: dict) -> np.ndarray:
    """Re-weight item scores by context features.

    Example: party setting boosts party-friendly games; competitive setting boosts
    strategic games; strangers may favor lower-complexity games.

    Args:
        scores: 1-D array aligned with `games`.
        games: DataFrame with annotated context features (party_friendliness,
            interaction_level, competitiveness).
        context: e.g. {'setting': 'party', 'familiarity': 'friends'}.
    """
    raise NotImplementedError
