"""Load raw BGG data, clean, filter sparse rows, write processed parquet files."""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data  # noqa: E402


def _archive_if_exists(processed_dir: Path) -> None:
    """Move processed_dir to a dated archive sibling if it is non-empty."""
    if processed_dir.exists() and any(processed_dir.iterdir()):
        archive_name = f"{date.today()}_processed"
        archive_path = processed_dir.parent.parent / "archive" / archive_name
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(processed_dir), str(archive_path))
        print(f"Archived existing processed data -> {archive_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    processed_dir = Path(cfg["processed_dir"])

    _archive_if_exists(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ratings...")
    ratings = data.load_ratings(cfg["raw_dir"])
    print(f"  raw ratings: {len(ratings):,} rows")

    print("Loading games...")
    games = data.load_games(cfg["raw_dir"])
    print(f"  raw games: {len(games):,} rows")

    print("Preprocessing (k-core filter)...")
    ratings, games = data.preprocess(
        ratings, games,
        cfg["min_ratings_per_user"], cfg["min_ratings_per_game"],
    )

    print("Writing parquet files...")
    ratings.to_parquet(processed_dir / "ratings.parquet", index=False)
    games.to_parquet(processed_dir / "games.parquet", index=False)

    n_users = ratings["user_id"].nunique()
    n_games = games["game_id"].nunique()
    print(
        f"\nDone.\n"
        f"  ratings rows : {len(ratings):,}\n"
        f"  unique users : {n_users:,}\n"
        f"  unique games : {n_games:,}\n"
        f"  output dir   : {processed_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
