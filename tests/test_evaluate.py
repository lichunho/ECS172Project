"""Unit tests for the _split helper in scripts/evaluate.py.

Pure in-memory — no data/ files, no trained model, no network.
Uses importlib to load scripts/evaluate.py without requiring scripts/ to be a package.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Import _split from scripts/evaluate.py via file path
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EVALUATE_PATH = _REPO_ROOT / "scripts" / "evaluate.py"

# Add repo root to sys.path so that scripts/evaluate.py's own
# `from src... import` works when the module is loaded.
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_spec = importlib.util.spec_from_file_location("evaluate", _EVALUATE_PATH)
_evaluate = importlib.util.module_from_spec(_spec)
# We do NOT call exec_module here because evaluate.py calls matplotlib.use("Agg")
# at import time which is fine, but the `if __name__ == "__main__": main()` guard
# means the heavy side-effects (argparse, file I/O) only run when invoked directly.
_spec.loader.exec_module(_evaluate)

_split = _evaluate._split


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ratings(
    n_users: int = 10,
    ratings_per_user: int = 20,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a synthetic ratings DataFrame with columns user_id, game_id, rating."""
    rng = np.random.default_rng(seed)
    rows = []
    for uid in range(n_users):
        game_ids = rng.choice(1000, size=ratings_per_user, replace=False)
        for gid in game_ids:
            rows.append({
                "user_id": uid,
                "game_id": int(gid),
                "rating": float(rng.integers(1, 11)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSplit:
    def test_disjoint_per_user(self):
        """No (user_id, game_id) pair should appear in both train and test."""
        ratings = _make_ratings(n_users=10, ratings_per_user=20, seed=42)
        train, test = _split(ratings, test_frac=0.2, seed=0)

        train_keys = set(zip(train["user_id"], train["game_id"]))
        test_keys = set(zip(test["user_id"], test["game_id"]))
        assert train_keys.isdisjoint(test_keys), (
            "Some (user_id, game_id) rows appear in both train and test."
        )

    def test_union_covers_all_rows(self):
        """train + test should contain exactly every row from the original DataFrame."""
        ratings = _make_ratings(n_users=8, ratings_per_user=15, seed=7)
        train, test = _split(ratings, test_frac=0.2, seed=1)

        assert len(train) + len(test) == len(ratings)

    def test_test_users_appear_in_train(self):
        """Every user_id in test should also appear in train (given enough ratings)."""
        # With 20 ratings per user and test_frac=0.2, each user has ~4 test rows
        # and ~16 train rows, so every user appears in both splits.
        ratings = _make_ratings(n_users=10, ratings_per_user=20, seed=3)
        train, test = _split(ratings, test_frac=0.2, seed=2)

        train_users = set(train["user_id"].unique())
        for uid in test["user_id"].unique():
            assert uid in train_users, (
                f"User {uid} appears in test but has no train rows."
            )

    def test_per_user_test_fraction_approx(self):
        """Each user's test fraction should be close to test_frac."""
        test_frac = 0.2
        ratings = _make_ratings(n_users=15, ratings_per_user=50, seed=99)
        train, test = _split(ratings, test_frac=test_frac, seed=5)

        for uid in ratings["user_id"].unique():
            total = len(ratings[ratings["user_id"] == uid])
            n_test = len(test[test["user_id"] == uid])
            actual_frac = n_test / total
            # Allow ±1 item of slack due to integer rounding
            slack = 1.0 / total
            assert abs(actual_frac - test_frac) <= test_frac + slack, (
                f"User {uid}: test_frac={actual_frac:.3f}, expected ~{test_frac}"
            )

    def test_deterministic_with_same_seed(self):
        """Calling _split twice with the same seed gives identical results."""
        ratings = _make_ratings(n_users=5, ratings_per_user=10, seed=0)
        train1, test1 = _split(ratings, test_frac=0.2, seed=42)
        train2, test2 = _split(ratings, test_frac=0.2, seed=42)

        pd.testing.assert_frame_equal(train1.reset_index(drop=True),
                                      train2.reset_index(drop=True))
        pd.testing.assert_frame_equal(test1.reset_index(drop=True),
                                      test2.reset_index(drop=True))

    def test_different_seeds_give_different_splits(self):
        """Two different seeds should (almost certainly) produce different splits."""
        ratings = _make_ratings(n_users=10, ratings_per_user=20, seed=0)
        _, test_a = _split(ratings, test_frac=0.2, seed=1)
        _, test_b = _split(ratings, test_frac=0.2, seed=2)

        keys_a = set(zip(test_a["user_id"], test_a["game_id"]))
        keys_b = set(zip(test_b["user_id"], test_b["game_id"]))
        assert keys_a != keys_b

    def test_high_test_frac(self):
        """test_frac=0.5 — disjointness and coverage still hold."""
        ratings = _make_ratings(n_users=6, ratings_per_user=20, seed=11)
        train, test = _split(ratings, test_frac=0.5, seed=8)

        train_keys = set(zip(train["user_id"], train["game_id"]))
        test_keys = set(zip(test["user_id"], test["game_id"]))
        assert train_keys.isdisjoint(test_keys)
        assert len(train) + len(test) == len(ratings)

    def test_no_rows_dropped(self):
        """Every original row ends up in exactly one partition."""
        ratings = _make_ratings(n_users=4, ratings_per_user=10, seed=55)
        train, test = _split(ratings, test_frac=0.3, seed=0)

        combined = pd.concat([train, test]).sort_values(
            ["user_id", "game_id"]
        ).reset_index(drop=True)
        original = ratings.sort_values(
            ["user_id", "game_id"]
        ).reset_index(drop=True)

        pd.testing.assert_frame_equal(combined, original)
