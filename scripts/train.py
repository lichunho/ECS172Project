"""Train the preference model and save artifacts under models/<run_name>/."""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preference import PreferenceModel  # noqa: E402


def _archive_if_exists(model_dir: Path) -> None:
    if model_dir.exists() and any(model_dir.iterdir()):
        archive = Path("archive") / f"{date.today()}_model_{model_dir.name}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(model_dir), str(archive))
        print(f"Archived existing model -> {archive}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    processed_dir = Path(cfg["processed_dir"])
    model_dir = Path("models") / cfg["run_name"]
    _archive_if_exists(model_dir)

    print("Loading ratings...")
    ratings = pd.read_parquet(processed_dir / "ratings.parquet")
    print(f"  {len(ratings):,} ratings")

    print("Fitting preference model...")
    model = PreferenceModel(
        loss=cfg["loss"],
        no_components=cfg["no_components"],
        learning_rate=cfg["learning_rate"],
        epochs=cfg["epochs"],
        bias_reg=cfg.get("bias_reg", 5.0),
    ).fit(ratings)

    out = model_dir / "model.pkl"
    model.save(str(out))
    print(f"Saved model -> {out.resolve()}")


if __name__ == "__main__":
    main()
