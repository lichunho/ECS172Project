"""Add LLM-annotated context features (party-friendliness, interaction, competitiveness)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import annotate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # games = pd.read_parquet(Path(cfg["processed_dir"]) / "games.parquet")
    # games = annotate.annotate_games(games)
    # games.to_parquet(Path(cfg["processed_dir"]) / "games_annotated.parquet")
    raise NotImplementedError("scaffold only; see src/annotate.py")


if __name__ == "__main__":
    main()
