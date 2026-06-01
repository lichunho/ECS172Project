"""Sample evaluation groups from the processed ratings."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import groups  # noqa: E402


def _archive_if_exists(path: Path) -> None:
    if path.exists():
        archive = Path("archive") / f"{date.today()}_groups_{path.parent.name}"
        archive.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(archive / path.name))
        print(f"Archived existing groups -> {archive}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    processed_dir = Path(cfg["processed_dir"])
    ratings = pd.read_parquet(processed_dir / "ratings.parquet", columns=["user_id", "game_id", "rating"])
    games = pd.read_parquet(processed_dir / "games_annotated.parquet", columns=["game_id", "categories"])

    print("Computing taste clusters...")
    clusters = groups.compute_taste_clusters(
        ratings, games,
        n_clusters=cfg["n_taste_clusters"],
        min_user_ratings=cfg["min_user_ratings"],
        seed=cfg["seed"],
    )
    user_ids = list(clusters.keys())
    print(f"  {len(user_ids):,} active users across {cfg['n_taste_clusters']} clusters")

    sampled = groups.sample_groups(
        user_ids,
        cfg["n_groups"],
        cfg["group_sizes"],
        seed=cfg["seed"],
        taste_clusters=clusters,
        mix=cfg.get("group_mix"),
    )

    out_dir = Path("results") / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "groups.json"
    _archive_if_exists(out_path)
    with open(out_path, "w") as f:
        json.dump([asdict(g) for g in sampled], f, indent=2)

    counts = pd.Series([g.group_type for g in sampled]).value_counts().to_dict()
    print(f"Wrote {len(sampled)} groups -> {out_path}  (types: {counts})")


if __name__ == "__main__":
    main()
