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

GAMMA = 0.5

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
    },
    "friends": {},
}


def _centered(series: pd.Series) -> np.ndarray:
    """Normalize an integer 1-5 label to [-0.5, 0.5]; NA -> 0 (neutral)."""
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    centered = (vals - 1.0) / 4.0 - 0.5
    return np.nan_to_num(centered, nan=0.0)


def adjust(scores: np.ndarray, games: pd.DataFrame, context: dict) -> np.ndarray:
    """Re-weight item scores by the (setting, familiarity) context.

    Args:
        scores: 1-D array aligned with `games`.
        games: DataFrame with annotated context label columns.
        context: e.g. {'setting': 'party', 'familiarity': 'friends'}.
    """
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

    factor = np.clip(1.0 + GAMMA * context_term, 0.1, None)
    return scores * factor
