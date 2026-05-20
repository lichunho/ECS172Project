"""Produce top-K group recommendations for a single group config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import aggregation, constraints, context  # noqa: E402
from src.preference import PreferenceModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # 1. Load games + trained model.
    # 2. games = constraints.filter_feasible(games, cfg["constraints"]).
    # 3. score_matrix = model.predict(cfg["user_ids"], games.game_id.values).
    # 4. group_scores = aggregation.aggregate(score_matrix, **cfg["aggregation"]).
    # 5. group_scores = context.adjust(group_scores, games, cfg["context"]).
    # 6. write top-K to results/<run_name>/recommendations.json.
    raise NotImplementedError("scaffold only; orchestration outlined above")


if __name__ == "__main__":
    main()
