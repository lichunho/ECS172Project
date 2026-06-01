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

from src import aggregation, constraints, context, data  # noqa: E402
from src.preference import PreferenceModel  # noqa: E402


def _resolve_model_path(cfg: dict) -> Path:
    explicit = cfg.get("model_path")
    if explicit:
        path = Path(explicit)
        if path.exists():
            return path

    candidates = []
    model_run_name = cfg.get("model_run_name")
    if model_run_name:
        candidates.append(Path("models") / model_run_name / "model.pkl")
    candidates.append(Path("models") / cfg["run_name"] / "model.pkl")
    candidates.append(Path("models") / "lightfm_baseline" / "model.pkl")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    model_files = list(Path("models").glob("*/model.pkl"))
    if len(model_files) == 1:
        return model_files[0]
    if len(model_files) > 1:
        return max(model_files, key=lambda path: path.stat().st_mtime)
    raise FileNotFoundError("no trained model found under models/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    processed_dir = Path(cfg.get("processed_dir", "data/processed"))
    results_dir = Path("results") / cfg["run_name"]
    results_dir.mkdir(parents=True, exist_ok=True)

    games_path = processed_dir / "games_annotated.parquet"
    if not games_path.exists():
        games_path = processed_dir / "games.parquet"
    games = pd.read_parquet(games_path)

    subset_cfg = cfg.get("subset", {})
    if subset_cfg.get("max_games") is not None and len(games) > 0:
        ratings = data.load_parquet_head(processed_dir / "ratings.parquet", subset_cfg.get("max_ratings"), columns=["user_id", "game_id", "rating"])
        top_games = ratings["game_id"].value_counts().head(int(subset_cfg["max_games"])).index
        games = games[games["game_id"].isin(top_games)].copy()

    user_ids = cfg.get("user_ids") or []
    if not user_ids:
        ratings = data.load_parquet_head(processed_dir / "ratings.parquet", subset_cfg.get("max_ratings"), columns=["user_id", "game_id", "rating"])
        available_users = ratings["user_id"].drop_duplicates().to_numpy()
        n_players = int(cfg.get("constraints", {}).get("n_players", 1))
        rng = np.random.default_rng(cfg.get("seed"))
        user_ids = [int(user_id) for user_id in rng.choice(available_users, size=n_players, replace=False)]
        print(f"No user_ids provided; sampled group of size {n_players} from processed ratings.")

    model_path = _resolve_model_path(cfg)
    model = PreferenceModel.load(str(model_path))
    allowed_user_ids = np.array(sorted(int(user_id) for user_id in model._user_id_map.keys()), dtype=int)

    feasible = constraints.filter_feasible(games, cfg.get("constraints", {}))
    if feasible.empty:
        raise ValueError("no feasible games remain after constraint filtering")

    known_item_ids = set(int(item_id) for item_id in model._item_id_map.keys())
    feasible = feasible[feasible["game_id"].isin(known_item_ids)].copy()
    if feasible.empty:
        raise ValueError("no feasible games remain after intersecting with the trained model item set")

    candidate_ids = feasible["game_id"].to_numpy()

    if user_ids:
        user_ids = [int(user_id) for user_id in user_ids if int(user_id) in model._user_id_map]
        if not user_ids:
            n_players = int(cfg.get("constraints", {}).get("n_players", 1))
            rng = np.random.default_rng(cfg.get("seed"))
            user_ids = [int(user_id) for user_id in rng.choice(allowed_user_ids, size=n_players, replace=False)]
            print("Configured user_ids were not in the trained model; sampled a valid group instead.")
    else:
        n_players = int(cfg.get("constraints", {}).get("n_players", 1))
        rng = np.random.default_rng(cfg.get("seed"))
        user_ids = [int(user_id) for user_id in rng.choice(allowed_user_ids, size=n_players, replace=False)]
        print(f"No user_ids provided; sampled group of size {n_players} from trained-model users.")

    score_matrix = model.predict(np.asarray(user_ids, dtype=int), candidate_ids)
    group_scores = aggregation.aggregate(score_matrix, **cfg.get("aggregation", {}))
    group_scores = context.adjust(group_scores, feasible, cfg.get("context", {}))

    top_k = int(cfg.get("top_k", 10))
    order = np.argsort(group_scores)[::-1][:top_k]
    recommendations = feasible.iloc[order].copy()
    recommendations["score"] = group_scores[order]

    payload = recommendations[["game_id", "name", "score"]].to_dict(orient="records")
    with (results_dir / "recommendations.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    recommendations[["game_id", "name", "score"]].to_csv(results_dir / "recommendations.csv", index=False)

    print(recommendations[["game_id", "name", "score"]].to_string(index=False))


if __name__ == "__main__":
    main()
