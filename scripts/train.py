"""Train the preference model and save artifacts under models/<run_name>/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preference import PreferenceModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ratings = pd.read_parquet(...); item_features = pd.read_parquet(...)
    # model = PreferenceModel(
    #     loss=cfg["loss"], no_components=cfg["no_components"],
    #     learning_rate=cfg["learning_rate"], epochs=cfg["epochs"],
    # ).fit(ratings, item_features)
    # model.save(f"models/{cfg['run_name']}/model.pkl")
    raise NotImplementedError("scaffold only; see src/preference.py")


if __name__ == "__main__":
    main()
