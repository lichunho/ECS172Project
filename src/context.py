"""Context-aware score adjustment based on social setting and group familiarity.

Concrete re-scoring rule (resolves the M5b design gap). Each integer 1-5 context
label is normalized and centered to c(x) = (x - 1) / 4 - 0.5  in [-0.5, 0.5]. For
a given (setting, familiarity) we hold a small weight table over labels (positive
weight = boost when the label is high, negative = penalize). The per-item context
term is the weighted sum of centered labels, and scores are rescaled
multiplicatively:

    factor_i = clip(1 + GAMMA * context_term_i, 0.1, None)
    adjusted_i = scores_i * factor_i

Missing labels (NA) contribute 0 (neutral), so partially annotated games are not
distorted. GAMMA bounds the maximum re-weighting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Context re-weighting strength. Set to 0 (context disabled) because our held-out
# rating evaluation has no social-context ground truth, so it cannot reward context
# adjustment: a gamma sweep showed every gamma > 0 only lowered NDCG and fairness.
# The weight tables below are retained for the qualitative recommender and any
# future context-aware evaluation. Raise gamma to re-enable context re-weighting.
GAMMA = 0.0

# weight[setting][label]: how strongly a high label pushes the score up (+) or down (-).
_SETTING_WEIGHTS: dict[str, dict[str, float]] = {
    "casual": {
        "party_friendliness": 0.6,
        "complexity": -0.6,
        "competitiveness": -0.3,
        "teach_time": -0.4,
    },
    "party": {
        "party_friendliness": 1.0,
        "interaction_level": 0.6,
        "complexity": -0.5,
        "downtime": -0.4,
        "player_elimination": -0.4,  # eliminated members sit idle -> tanks party play
    },
    "competitive": {
        "competitiveness": 1.0,
        "interaction_level": 0.4,
        "mixed_skill_robustness": 0.4,
        "party_friendliness": -0.3,
    },
}

# Familiarity nudges layered on top of the setting weights.
_FAMILIARITY_WEIGHTS: dict[str, dict[str, float]] = {
    "strangers": {
        "complexity": -0.4,
        "teach_time": -0.4,
        "social_conflict": -0.3,
        "mixed_skill_robustness": 0.3,
        "min_age_fit": 0.3,          # high = accessible to all ages -> safer for mixed groups
        "player_elimination": -0.4,  # knocking a stranger out early is a bad first impression
    },
    "friends": {},
}


def _centered(series: pd.Series) -> np.ndarray:
    """Normalize an integer 1-5 label to [-0.5, 0.5]; NA -> 0 (neutral)."""
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    centered = (vals - 1.0) / 4.0 - 0.5
    return np.nan_to_num(centered, nan=0.0)


def adjust(scores: np.ndarray, games: pd.DataFrame, context: dict,
           gamma: float | None = None) -> np.ndarray:
    """Re-weight item scores by the (setting, familiarity) context.

    Args:
        scores: 1-D array aligned with `games`.
        games: DataFrame with annotated context label columns.
        context: e.g. {'setting': 'party', 'familiarity': 'friends'}.
        gamma: re-weighting strength; falls back to module GAMMA when None.
    """
    g = GAMMA if gamma is None else gamma
    scores = np.asarray(scores, dtype="float64")
    weights: dict[str, float] = {}
    for label, w in _SETTING_WEIGHTS.get(context.get("setting", ""), {}).items():
        weights[label] = weights.get(label, 0.0) + w
    for label, w in _FAMILIARITY_WEIGHTS.get(context.get("familiarity", ""), {}).items():
        weights[label] = weights.get(label, 0.0) + w

    if not weights:
        return scores

    context_term = np.zeros(len(games), dtype="float64")
    for label, w in weights.items():
        if label in games.columns:
            context_term += w * _centered(games[label])

    factor = np.clip(1.0 + g * context_term, 0.1, None)
    return scores * factor


# How hard the player-count poll nudges the ranking (kept small — it is a soft tie-breaker).
PLAYER_COUNT_BEST_BOOST = 0.15
PLAYER_COUNT_OFF_PENALTY = 0.15


def player_count_factor(games: pd.DataFrame, n_players: int | None) -> np.ndarray:
    """Soft recommend-time fit of each game to a group of `n_players`.

    Uses the BGG community poll carried from M1 (`data.py`): `best_player_count`
    (single best count) and `good_player_counts` (list of counts voted "good").
    This is the group's-size-dependent computation the M1/M2 plans deferred out of
    static annotation.

    Returns a multiplicative factor aligned with `games`:
      - boost (1 + BEST_BOOST) when `n_players` is the game's best count;
      - penalty (1 - OFF_PENALTY) when `good_player_counts` is non-empty and
        `n_players` is not in it (positive evidence the size plays poorly);
      - neutral (1.0) otherwise — including when both fields are absent, so a game
        is never penalized for missing poll data.
    """
    n = len(games)
    if n_players is None:
        return np.ones(n)

    if "best_player_count" in games.columns:
        best = games["best_player_count"].to_numpy()
    else:
        best = np.full(n, pd.NA, dtype="object")
    if "good_player_counts" in games.columns:
        good = games["good_player_counts"].to_numpy()
    else:
        good = np.empty(n, dtype="object")

    factor = np.ones(n, dtype="float64")
    for i in range(n):
        b = best[i]
        if pd.notna(b) and int(b) == n_players:
            factor[i] = 1.0 + PLAYER_COUNT_BEST_BOOST
            continue
        g = good[i]
        if isinstance(g, (list, np.ndarray)) and len(g) > 0 and n_players not in {int(x) for x in g}:
            factor[i] = 1.0 - PLAYER_COUNT_OFF_PENALTY
    return factor
