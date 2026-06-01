"""Pytest tests for src/preference.py (PreferenceModel).

Pure in-memory — no data/ files, no network. Deterministic (SVD uses random_state=0).

Coverage:
1. After fit: predict shape, predict_pairs length, scores roughly on 1-10 scale.
2. Round-trip save/load reproduces identical predictions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preference import PreferenceModel  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixture: a tiny but non-degenerate ratings frame.
# 6 users x 6 items, multiple ratings per user and per item.
# Dense ids 0..5 for both users and items; ratings on the 1-10 scale.
# ---------------------------------------------------------------------------

_RATINGS = pd.DataFrame(
    [
        # user_id, game_id, rating
        (0, 0, 8),
        (0, 1, 6),
        (0, 2, 9),
        (1, 0, 5),
        (1, 1, 7),
        (1, 3, 4),
        (2, 2, 8),
        (2, 3, 6),
        (2, 4, 9),
        (3, 1, 3),
        (3, 3, 5),
        (3, 5, 7),
        (4, 0, 6),
        (4, 4, 8),
        (4, 5, 9),
        (5, 2, 4),
        (5, 3, 7),
        (5, 5, 8),
    ],
    columns=["user_id", "game_id", "rating"],
)

_N_USERS = 6
_N_ITEMS = 6
_NO_COMPONENTS = 2  # small so SVD is fast; k = min(no_components, min(shape)-1) = 2


@pytest.fixture(scope="module")
def fitted_model() -> PreferenceModel:
    """Return a model fitted on _RATINGS once for the whole module."""
    model = PreferenceModel(no_components=_NO_COMPONENTS)
    model.fit(_RATINGS)
    return model


# ---------------------------------------------------------------------------
# 1. predict shape
# ---------------------------------------------------------------------------

def test_predict_shape(fitted_model: PreferenceModel) -> None:
    users = np.array([0, 1, 2])
    items = np.array([0, 3, 4, 5])
    scores = fitted_model.predict(users, items)
    assert scores.shape == (len(users), len(items)), (
        f"Expected shape ({len(users)}, {len(items)}), got {scores.shape}"
    )


def test_predict_single_user_single_item(fitted_model: PreferenceModel) -> None:
    scores = fitted_model.predict(np.array([0]), np.array([0]))
    assert scores.shape == (1, 1)


def test_predict_all_users_all_items(fitted_model: PreferenceModel) -> None:
    all_users = np.arange(_N_USERS)
    all_items = np.arange(_N_ITEMS)
    scores = fitted_model.predict(all_users, all_items)
    assert scores.shape == (_N_USERS, _N_ITEMS)


# ---------------------------------------------------------------------------
# 2. predict_pairs length
# ---------------------------------------------------------------------------

def test_predict_pairs_length(fitted_model: PreferenceModel) -> None:
    users = np.array([0, 1, 2, 3, 4, 5])
    items = np.array([0, 1, 2, 3, 4, 5])
    preds = fitted_model.predict_pairs(users, items)
    assert preds.ndim == 1, "predict_pairs must return a 1-D array"
    assert len(preds) == len(users), (
        f"Expected length {len(users)}, got {len(preds)}"
    )


def test_predict_pairs_single_pair(fitted_model: PreferenceModel) -> None:
    preds = fitted_model.predict_pairs(np.array([0]), np.array([0]))
    assert preds.ndim == 1
    assert len(preds) == 1


# ---------------------------------------------------------------------------
# 3. Predicted scores roughly on 1-10 scale (allow [0, 12] slack)
# ---------------------------------------------------------------------------

def test_predict_score_range(fitted_model: PreferenceModel) -> None:
    all_users = np.arange(_N_USERS)
    all_items = np.arange(_N_ITEMS)
    scores = fitted_model.predict(all_users, all_items)
    assert scores.min() >= 0, f"Scores below 0: min={scores.min():.4f}"
    assert scores.max() <= 12, f"Scores above 12: max={scores.max():.4f}"


def test_predict_pairs_score_range(fitted_model: PreferenceModel) -> None:
    users = _RATINGS["user_id"].to_numpy()
    items = _RATINGS["game_id"].to_numpy()
    preds = fitted_model.predict_pairs(users, items)
    assert preds.min() >= 0, f"Pair scores below 0: min={preds.min():.4f}"
    assert preds.max() <= 12, f"Pair scores above 12: max={preds.max():.4f}"


# ---------------------------------------------------------------------------
# 4. predict and predict_pairs agree on matching inputs
# ---------------------------------------------------------------------------

def test_predict_vs_predict_pairs_consistency(fitted_model: PreferenceModel) -> None:
    """Diagonal of the full matrix must equal predict_pairs output."""
    users = np.array([0, 1, 2, 3])
    items = np.array([0, 1, 2, 3])
    matrix = fitted_model.predict(users, items)
    pairs = fitted_model.predict_pairs(users, items)
    np.testing.assert_allclose(
        np.diag(matrix), pairs, rtol=1e-5,
        err_msg="predict diagonal and predict_pairs differ",
    )


# ---------------------------------------------------------------------------
# 5. Round-trip: save then load reproduces identical predictions
# ---------------------------------------------------------------------------

def test_save_load_round_trip(tmp_path: Path) -> None:
    model = PreferenceModel(no_components=_NO_COMPONENTS)
    model.fit(_RATINGS)

    pkl_path = tmp_path / "m.pkl"
    model.save(pkl_path)

    loaded = PreferenceModel.load(pkl_path)

    users = np.arange(_N_USERS)
    items = np.arange(_N_ITEMS)

    original_matrix = model.predict(users, items)
    loaded_matrix = loaded.predict(users, items)
    np.testing.assert_allclose(
        original_matrix, loaded_matrix, rtol=1e-6,
        err_msg="predict output differs after save/load",
    )

    original_pairs = model.predict_pairs(users, items)
    loaded_pairs = loaded.predict_pairs(users, items)
    np.testing.assert_allclose(
        original_pairs, loaded_pairs, rtol=1e-6,
        err_msg="predict_pairs output differs after save/load",
    )


def test_save_load_scalar_attributes(tmp_path: Path) -> None:
    """Scalar meta-attributes are preserved through the round-trip."""
    model = PreferenceModel(no_components=_NO_COMPONENTS)
    model.fit(_RATINGS)

    pkl_path = tmp_path / "m2.pkl"
    model.save(pkl_path)
    loaded = PreferenceModel.load(pkl_path)

    assert loaded.n_users_ == model.n_users_
    assert loaded.n_items_ == model.n_items_
    np.testing.assert_allclose(loaded.mu_, model.mu_, rtol=1e-9)


# ---------------------------------------------------------------------------
# 6. Unfitted model raises RuntimeError
# ---------------------------------------------------------------------------

def test_predict_raises_if_not_fitted() -> None:
    model = PreferenceModel(no_components=_NO_COMPONENTS)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(np.array([0]), np.array([0]))


def test_predict_pairs_raises_if_not_fitted() -> None:
    model = PreferenceModel(no_components=_NO_COMPONENTS)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict_pairs(np.array([0]), np.array([0]))
