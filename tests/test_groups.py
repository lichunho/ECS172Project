"""pytest tests for src/groups.py.

No data/ files, no network, no Ollama. All DataFrames are built in-memory.
RNGs are seeded for determinism.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.groups import Group, compute_taste_clusters, sample_groups  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

# 4 clusters, 8 users per cluster = 32 users total.
# Users are numbered 0..31; cluster assignment is user_id // 8.
_N_CLUSTERS = 4
_USERS_PER_CLUSTER = 8
_TOTAL_USERS = _N_CLUSTERS * _USERS_PER_CLUSTER


def _make_taste_clusters() -> dict[int, int]:
    """Synthetic taste_clusters: {user_id: cluster_label}, 4 clusters x 8 users."""
    return {uid: uid // _USERS_PER_CLUSTER for uid in range(_TOTAL_USERS)}


def _make_ratings_and_games(min_user_ratings: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build tiny in-memory ratings + games frames for compute_taste_clusters.

    Layout:
    - 3 games, each with 1 category.
    - 10 "active" users (IDs 0-9), each rated all 3 games → 30 ratings ≥ min_user_ratings.
    - 2 "sparse" users (IDs 10-11), each rated only 1 game → below any min > 1.
    """
    rng = np.random.default_rng(0)

    # Active users: 10 users × 3 games = 30 ratings each (well above any threshold)
    active_user_ids = list(range(10))
    game_ids = [101, 102, 103]
    rows = [
        {"user_id": uid, "game_id": gid, "rating": float(rng.integers(1, 11))}
        for uid in active_user_ids
        for gid in game_ids
    ]
    # Sparse users: each rates exactly 1 game
    rows += [
        {"user_id": 10, "game_id": 101, "rating": 5.0},
        {"user_id": 11, "game_id": 102, "rating": 7.0},
    ]
    ratings = pd.DataFrame(rows)

    # Games: one category each so the profile matrix is 3-dimensional
    games = pd.DataFrame([
        {"game_id": 101, "categories": ["Strategy"]},
        {"game_id": 102, "categories": ["Party"]},
        {"game_id": 103, "categories": ["Abstract"]},
    ])

    return ratings, games


# ---------------------------------------------------------------------------
# Tests: sample_groups
# ---------------------------------------------------------------------------

class TestSampleGroups:
    def setup_method(self):
        self.taste_clusters = _make_taste_clusters()
        self.user_ids = np.array(list(self.taste_clusters.keys()))
        self.sizes = [3, 4, 5]
        self.mix = {"similar": 0.34, "divergent": 0.33, "random": 0.33}

    def test_group_sizes_within_allowed(self):
        """Every returned group's size must be one of the values in sizes."""
        groups = sample_groups(
            self.user_ids,
            n_groups=50,
            sizes=self.sizes,
            seed=42,
            taste_clusters=self.taste_clusters,
            mix=self.mix,
        )
        for g in groups:
            assert len(g.user_ids) in self.sizes, (
                f"group size {len(g.user_ids)} not in {self.sizes}"
            )

    def test_type_distribution_roughly_matches_mix(self):
        """Over many groups the observed type fractions are close to mix (±0.15)."""
        n = 300
        groups = sample_groups(
            self.user_ids,
            n_groups=n,
            sizes=self.sizes,
            seed=7,
            taste_clusters=self.taste_clusters,
            mix=self.mix,
        )
        counts = {"similar": 0, "divergent": 0, "random": 0}
        for g in groups:
            counts[g.group_type] += 1
        for gtype, expected_frac in self.mix.items():
            observed_frac = counts[gtype] / n
            assert abs(observed_frac - expected_frac) < 0.15, (
                f"{gtype}: expected ~{expected_frac:.2f}, got {observed_frac:.2f}"
            )

    def test_divergent_groups_have_distinct_clusters(self):
        """Every divergent group's members must come from distinct clusters."""
        groups = sample_groups(
            self.user_ids,
            n_groups=200,
            sizes=self.sizes,
            seed=0,
            taste_clusters=self.taste_clusters,
            mix=self.mix,
        )
        divergent = [g for g in groups if g.group_type == "divergent"]
        assert divergent, "No divergent groups produced — increase n_groups or check mix"
        for g in divergent:
            member_clusters = [self.taste_clusters[uid] for uid in g.user_ids]
            assert len(member_clusters) == len(set(member_clusters)), (
                f"divergent group has repeated cluster: members={g.user_ids}, "
                f"clusters={member_clusters}"
            )

    def test_similar_groups_share_one_cluster(self):
        """Every similar group's members must all belong to the same cluster."""
        groups = sample_groups(
            self.user_ids,
            n_groups=200,
            sizes=self.sizes,
            seed=1,
            taste_clusters=self.taste_clusters,
            mix=self.mix,
        )
        similar = [g for g in groups if g.group_type == "similar"]
        assert similar, "No similar groups produced — increase n_groups or check mix"
        for g in similar:
            member_clusters = {self.taste_clusters[uid] for uid in g.user_ids}
            assert len(member_clusters) == 1, (
                f"similar group spans multiple clusters: members={g.user_ids}, "
                f"clusters={member_clusters}"
            )

    def test_returns_correct_number_of_groups(self):
        n = 25
        groups = sample_groups(
            self.user_ids,
            n_groups=n,
            sizes=self.sizes,
            seed=99,
            taste_clusters=self.taste_clusters,
        )
        assert len(groups) == n

    def test_groups_are_Group_instances(self):
        groups = sample_groups(
            self.user_ids,
            n_groups=10,
            sizes=self.sizes,
            seed=3,
            taste_clusters=self.taste_clusters,
        )
        for g in groups:
            assert isinstance(g, Group)


# ---------------------------------------------------------------------------
# Tests: compute_taste_clusters
# ---------------------------------------------------------------------------

class TestComputeTasteClusters:
    def test_only_active_users_returned(self):
        """Users with fewer than min_user_ratings ratings must be absent."""
        ratings, games = _make_ratings_and_games()
        # Users 10 and 11 each have only 1 rating; set threshold to 2 to exclude them.
        result = compute_taste_clusters(
            ratings, games, n_clusters=2, min_user_ratings=2, seed=0
        )
        assert 10 not in result, "sparse user 10 should be excluded"
        assert 11 not in result, "sparse user 11 should be excluded"

    def test_active_users_are_all_present(self):
        """Every user with >= min_user_ratings must appear in the result."""
        ratings, games = _make_ratings_and_games()
        min_r = 3
        result = compute_taste_clusters(
            ratings, games, n_clusters=2, min_user_ratings=min_r, seed=0
        )
        counts = ratings["user_id"].value_counts()
        expected_active = set(counts[counts >= min_r].index)
        assert expected_active == set(result.keys()), (
            f"active users mismatch: expected {expected_active}, got {set(result.keys())}"
        )

    def test_label_count_does_not_exceed_n_clusters(self):
        """Number of distinct cluster labels must be <= n_clusters."""
        ratings, games = _make_ratings_and_games()
        n_clusters = 3
        result = compute_taste_clusters(
            ratings, games, n_clusters=n_clusters, min_user_ratings=2, seed=0
        )
        distinct_labels = set(result.values())
        assert len(distinct_labels) <= n_clusters, (
            f"got {len(distinct_labels)} distinct labels, expected <= {n_clusters}"
        )

    def test_returns_dict(self):
        ratings, games = _make_ratings_and_games()
        result = compute_taste_clusters(
            ratings, games, n_clusters=2, min_user_ratings=2, seed=42
        )
        assert isinstance(result, dict)
        # Values should be ints (cluster labels)
        for v in result.values():
            assert isinstance(v, int)

    def test_threshold_equal_to_max_ratings_keeps_those_users(self):
        """Users whose rating count equals min_user_ratings exactly are included."""
        ratings, games = _make_ratings_and_games()
        # Each active user has exactly 3 ratings (one per game).
        result = compute_taste_clusters(
            ratings, games, n_clusters=2, min_user_ratings=3, seed=0
        )
        # All 10 active users (IDs 0-9) have 3 ratings and should be present.
        for uid in range(10):
            assert uid in result, f"user {uid} with exactly 3 ratings should be included"


# ---------------------------------------------------------------------------
# Tests: graceful degrade (taste_clusters=None)
# ---------------------------------------------------------------------------

class TestGracefulDegrade:
    def test_all_groups_random_when_no_clusters(self):
        """When taste_clusters is None every group must have group_type == 'random'."""
        user_ids = np.arange(50)
        groups = sample_groups(
            user_ids,
            n_groups=40,
            sizes=[3, 4],
            seed=5,
            taste_clusters=None,
        )
        for g in groups:
            assert g.group_type == "random", (
                f"expected 'random', got '{g.group_type}'"
            )

    def test_sizes_still_respected_when_no_clusters(self):
        """Group sizes must still be drawn from sizes even with taste_clusters=None."""
        user_ids = np.arange(50)
        sizes = [2, 3]
        groups = sample_groups(
            user_ids,
            n_groups=30,
            sizes=sizes,
            seed=6,
            taste_clusters=None,
        )
        for g in groups:
            assert len(g.user_ids) in sizes

    def test_correct_count_when_no_clusters(self):
        user_ids = np.arange(20)
        n = 15
        groups = sample_groups(user_ids, n_groups=n, sizes=[3], seed=8, taste_clusters=None)
        assert len(groups) == n
