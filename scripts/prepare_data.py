"""Load raw BGG data, clean, filter sparse rows, write processed parquet files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ratings = data.load_ratings(cfg["raw_dir"])
    # games = data.load_games(cfg["raw_dir"])
    # ratings, games = data.preprocess(
    #     ratings, games,
    #     cfg["min_ratings_per_user"], cfg["min_ratings_per_game"],
    # )
    # write to cfg["processed_dir"]
    raise NotImplementedError("scaffold only; see src/data.py")


if __name__ == "__main__":
    main()
