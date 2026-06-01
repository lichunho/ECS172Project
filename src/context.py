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
    base_scores = np.asarray(scores, dtype=float).copy()
    if base_scores.ndim != 1:
        raise ValueError("scores must be a 1-D array aligned with games")

    def _feature(name: str) -> np.ndarray:
        if name not in games.columns:
            return np.zeros(len(games), dtype=float)
        values = pd.to_numeric(games[name], errors="coerce").to_numpy(dtype=float)
        values = np.nan_to_num(values, nan=3.0)
        return np.clip((values - 3.0) / 2.0, -1.0, 1.0)

    setting = str(context.get("setting", "")).lower().strip()
    familiarity = str(context.get("familiarity", "")).lower().strip()

    modifier = np.zeros(len(games), dtype=float)
    if setting == "party":
        modifier += 0.18 * _feature("party_friendliness")
        modifier += 0.12 * _feature("interaction_level")
        modifier -= 0.10 * _feature("downtime")
        modifier -= 0.10 * _feature("teach_time")
        modifier -= 0.06 * _feature("complexity")
    elif setting == "competitive":
        modifier += 0.18 * _feature("competitiveness")
        modifier += 0.10 * _feature("interaction_level")
        modifier += 0.06 * _feature("social_conflict")
        modifier += 0.06 * _feature("complexity")
    else:
        modifier += 0.10 * _feature("party_friendliness")
        modifier += 0.08 * _feature("interaction_level")
        modifier -= 0.10 * _feature("teach_time")
        modifier -= 0.08 * _feature("complexity")

    if familiarity == "strangers":
        modifier += 0.10 * _feature("mixed_skill_robustness")
        modifier -= 0.12 * _feature("complexity")
        modifier -= 0.10 * _feature("teach_time")
        modifier -= 0.08 * _feature("social_conflict")
    else:
        modifier += 0.05 * _feature("interaction_level")
        modifier += 0.04 * _feature("party_friendliness")

    modifier = np.clip(modifier, -0.35, 0.35)
    return base_scores * (1.0 + modifier)
