"""Sample evaluation groups from the processed ratings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data  # noqa: E402
from src import groups  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    processed_dir = Path(cfg.get("processed_dir", "data/processed"))
    results_dir = Path("results") / cfg["run_name"]
    results_dir.mkdir(parents=True, exist_ok=True)

    ratings, _ = data.load_processed_subset(processed_dir, cfg.get("subset", {}))
    user_ids = ratings["user_id"].drop_duplicates().to_numpy()
    sampled = groups.sample_groups(
        user_ids,
        int(cfg["n_groups"]),
        [int(size) for size in cfg["group_sizes"]],
        seed=cfg.get("seed"),
    )

    payload = [
        {
            "group_id": index,
            "user_ids": group.user_ids,
            "context": group.context,
            "constraints": group.constraints,
        }
        for index, group in enumerate(sampled)
    ]

    with (results_dir / "groups.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    pd.DataFrame(payload).to_json(results_dir / "groups_flat.json", orient="records", indent=2)
    print(f"Wrote {len(payload)} groups -> {results_dir / 'groups.json'}")


if __name__ == "__main__":
    main()
