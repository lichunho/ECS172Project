"""Train the preference model and save artifacts under models/<run_name>/."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data  # noqa: E402
from src.preference import PreferenceModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    processed_dir = Path(cfg.get("processed_dir", "data/processed"))
    model_dir = Path("models") / cfg["run_name"]
    model_dir.mkdir(parents=True, exist_ok=True)

    subset_cfg = cfg.get("subset", {})
    ratings, games = data.load_processed_subset(processed_dir, subset_cfg)

    model = PreferenceModel(
        loss=cfg["loss"],
        no_components=int(cfg["no_components"]),
        learning_rate=float(cfg["learning_rate"]),
        epochs=int(cfg["epochs"]),
    ).fit(ratings, games)

    model_path = model_dir / "model.pkl"
    model.save(str(model_path))

    summary = {
        "model_path": str(model_path),
        "n_ratings": int(len(ratings)),
        "n_users": int(ratings["user_id"].nunique()),
        "n_games": int(games["game_id"].nunique()),
        "config": {
            "loss": cfg["loss"],
            "no_components": int(cfg["no_components"]),
            "learning_rate": float(cfg["learning_rate"]),
            "epochs": int(cfg["epochs"]),
        },
    }
    with (model_dir / "train_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved model -> {model_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
