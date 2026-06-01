"""Produce top-K group recommendations for a single group config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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

    user_ids = np.asarray(cfg["user_ids"], dtype=int)
    if user_ids.size == 0:
        print("No user_ids in config; populate group.yaml `user_ids`. Nothing to do.")
        return

    processed_dir = Path(cfg["processed_dir"])
    games = pd.read_parquet(processed_dir / "games_annotated.parquet")
    model = PreferenceModel.load(str(Path("models") / cfg["model_run_name"] / "model.pkl"))

    # 1. Constraint filter (before ranking).
    feasible = constraints.filter_feasible(games, cfg["constraints"])
    item_ids = feasible["game_id"].to_numpy()

    # 2. Per-user scores, then 3. fairness aggregation, then 4. context adjust.
    score_matrix = model.predict(user_ids, item_ids)
    group_scores = aggregation.aggregate(score_matrix, **cfg["aggregation"])
    group_scores = context.adjust(group_scores, feasible, cfg["context"])

    top_k = cfg["top_k"]
    order = np.argsort(-group_scores)[:top_k]
    recs = [
        {
            "game_id": int(feasible.iloc[idx]["game_id"]),
            "name": str(feasible.iloc[idx]["name"]),
            "group_score": float(group_scores[idx]),
        }
        for idx in order
    ]

    out_dir = Path("results") / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "recommendations.json"
    with open(out_path, "w") as f:
        json.dump({"user_ids": user_ids.tolist(), "context": cfg["context"],
                   "constraints": cfg["constraints"], "recommendations": recs}, f, indent=2)
    print(f"Wrote top-{top_k} recommendations -> {out_path}")
    for r in recs:
        print(f"  {r['group_score']:.3f}  {r['name']}")


if __name__ == "__main__":
    main()
