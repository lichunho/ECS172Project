"""Simulated group sampling for evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Group:
    user_ids: list
    context: dict
    constraints: dict


def sample_groups(
    user_ids: np.ndarray,
    n_groups: int,
    sizes: list[int],
    seed: int | None = None,
) -> list[Group]:
    """Sample groups of users from the rating matrix for evaluation."""
    rng = np.random.default_rng(seed)
    user_ids = np.asarray(user_ids)
    if user_ids.size == 0 or n_groups <= 0:
        return []
    if not sizes:
        raise ValueError("sizes must contain at least one group size")

    settings = np.array(["casual", "party", "competitive"])
    familiarity = np.array(["friends", "strangers"])
    groups: list[Group] = []

    for _ in range(n_groups):
        size = int(rng.choice(sizes))
        if size <= 0:
            raise ValueError(f"invalid group size: {size}")
        sampled = rng.choice(user_ids, size=size, replace=size > len(user_ids))
        setting = str(rng.choice(settings, p=[0.45, 0.30, 0.25]))
        fam = str(rng.choice(familiarity, p=[0.65, 0.35]))

        if setting == "party":
            max_time = int(rng.choice([30, 45, 60], p=[0.25, 0.45, 0.30]))
        elif setting == "competitive":
            max_time = int(rng.choice([60, 90, 120], p=[0.25, 0.45, 0.30]))
        else:
            max_time = int(rng.choice([45, 60, 90], p=[0.35, 0.40, 0.25]))

        groups.append(
            Group(
                user_ids=[int(user_id) for user_id in sampled],
                context={"setting": setting, "familiarity": fam},
                constraints={"n_players": size, "max_playtime_min": max_time},
            )
        )

    return groups
