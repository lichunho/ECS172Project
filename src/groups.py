"""Simulated group sampling for evaluation.

No real groups exist, so we manufacture them. The critical requirement (see
plan.md): the eval set must contain *divergent-taste* groups, otherwise every
aggregation method ranks items identically and the fairness work is invisible.

We build a coarse per-user taste profile (mean rating per game category),
cluster users with k-means, then compose three kinds of group:
  - 'similar'   : all members drawn from one taste cluster,
  - 'divergent' : each member drawn from a *different* taste cluster,
  - 'random'    : members drawn uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

_SETTINGS = ["casual", "party", "competitive"]
_FAMILIARITY = ["friends", "strangers"]
_PLAYTIME_BUDGETS = [30, 60, 90, 120, None]


@dataclass
class Group:
    user_ids: list
    context: dict
    constraints: dict
    group_type: str = field(default="random")


def compute_taste_clusters(
    ratings: pd.DataFrame,
    games: pd.DataFrame,
    n_clusters: int,
    min_user_ratings: int,
    seed: int | None = None,
) -> dict[int, int]:
    """Cluster active users by category-preference profile.

    Returns {user_id: cluster_label} for users with >= `min_user_ratings`
    ratings. Users below that threshold are omitted (excluded from sampling).
    """
    counts = ratings["user_id"].value_counts()
    active = counts[counts >= min_user_ratings].index
    sub = ratings[ratings["user_id"].isin(active)]

    cats = games[["game_id", "categories"]].explode("categories").dropna(subset=["categories"])
    merged = sub.merge(cats, on="game_id")
    profile = (
        merged.groupby(["user_id", "categories"])["rating"]
        .mean()
        .unstack(fill_value=0.0)
    )

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(profile.to_numpy())
    return dict(zip(profile.index.to_numpy().tolist(), labels.tolist()))


def _sample_constraints(rng: np.random.Generator, size: int) -> dict:
    budget = _PLAYTIME_BUDGETS[rng.integers(len(_PLAYTIME_BUDGETS))]
    return {"n_players": size, "max_playtime_min": budget}


def _sample_context(rng: np.random.Generator) -> dict:
    return {
        "setting": _SETTINGS[rng.integers(len(_SETTINGS))],
        "familiarity": _FAMILIARITY[rng.integers(len(_FAMILIARITY))],
    }


def sample_groups(
    user_ids: np.ndarray,
    n_groups: int,
    sizes: list[int],
    seed: int | None = None,
    taste_clusters: dict[int, int] | None = None,
    mix: dict[str, float] | None = None,
) -> list[Group]:
    """Sample evaluation groups of users.

    Args:
        user_ids: pool of candidate user ids.
        n_groups: number of groups to produce.
        sizes: allowed group sizes; one is sampled per group.
        seed: RNG seed.
        taste_clusters: {user_id: cluster_label}. Required for 'similar' and
            'divergent' groups; if None, all groups are 'random'.
        mix: fractions for {'similar','divergent','random'} (default even split
            when clusters are available).

    Returns:
        list[Group].
    """
    rng = np.random.default_rng(seed)

    if taste_clusters:
        clustered_users = np.array(list(taste_clusters.keys()))
        by_cluster: dict[int, list] = {}
        for uid, lbl in taste_clusters.items():
            by_cluster.setdefault(lbl, []).append(uid)
        cluster_labels = list(by_cluster.keys())
        mix = mix or {"similar": 1 / 3, "divergent": 1 / 3, "random": 1 / 3}
    else:
        clustered_users = np.asarray(user_ids)
        by_cluster = {}
        cluster_labels = []
        mix = {"similar": 0.0, "divergent": 0.0, "random": 1.0}

    pool = np.asarray(user_ids)
    groups: list[Group] = []
    for _ in range(n_groups):
        size = int(sizes[rng.integers(len(sizes))])
        roll = rng.random()
        if roll < mix["similar"] and cluster_labels:
            gtype = "similar"
        elif roll < mix["similar"] + mix["divergent"] and len(cluster_labels) >= 2:
            gtype = "divergent"
        else:
            gtype = "random"

        if gtype == "similar":
            lbl = cluster_labels[rng.integers(len(cluster_labels))]
            members = list(by_cluster[lbl])
            picks = rng.choice(members, size=min(size, len(members)), replace=False)
        elif gtype == "divergent":
            chosen_labels = rng.choice(cluster_labels, size=min(size, len(cluster_labels)), replace=False)
            picks = [int(rng.choice(by_cluster[lbl])) for lbl in chosen_labels]
        else:
            picks = rng.choice(pool, size=size, replace=False)

        groups.append(
            Group(
                user_ids=[int(x) for x in picks],
                context=_sample_context(rng),
                constraints=_sample_constraints(rng, size),
                group_type=gtype,
            )
        )
    return groups
