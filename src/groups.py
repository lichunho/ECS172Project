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
    raise NotImplementedError
