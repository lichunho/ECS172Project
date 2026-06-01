"""Run full evaluation: ours vs. baselines on simulated groups."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import aggregation, baselines, constraints, context, data, metrics  # noqa: E402
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


def _load_or_create_groups(cfg: dict, processed_dir: Path, results_dir: Path) -> list[dict]:
    groups_path = results_dir / "groups.json"
    if groups_path.exists():
        with groups_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    from src import groups as groups_module

    ratings = pd.read_parquet(processed_dir / "ratings.parquet")
    sampled = groups_module.sample_groups(
        ratings["user_id"].drop_duplicates().to_numpy(),
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
    results_dir.mkdir(parents=True, exist_ok=True)
    with groups_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def _mean_ratings_from_matrix(submatrix: np.ndarray) -> np.ndarray:
    mask = submatrix > 0
    counts = mask.sum(axis=0)
    totals = submatrix.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        means = np.divide(totals, counts, out=np.zeros_like(totals, dtype=float), where=counts > 0)
    return means


def _top_items_from_scores(item_ids: np.ndarray, scores: np.ndarray, top_k: int) -> np.ndarray:
    order = np.argsort(scores)[::-1][:top_k]
    return item_ids[order]


def _save_plots(summary: pd.DataFrame, results_dir: Path) -> None:
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_to_plot = [col for col in summary.columns if col != "method"]
    for metric_name in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(summary["method"], summary[metric_name], color="#2f6fed")
        ax.set_title(metric_name)
        ax.set_ylabel(metric_name)
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{metric_name}.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    processed_dir = Path(cfg.get("processed_dir", "data/processed"))
    results_dir = Path("results") / cfg["run_name"]
    results_dir.mkdir(parents=True, exist_ok=True)

    ratings, games = data.load_processed_subset(processed_dir, cfg.get("subset", {}))

    groups_payload = _load_or_create_groups(cfg, processed_dir, results_dir)
    model_path = _resolve_model_path(cfg)
    model = PreferenceModel.load(str(model_path))

    rating_matrix = sparse.csr_matrix(
        (
            ratings["rating"].to_numpy(dtype=np.float32),
            (
                ratings["user_id"].to_numpy(dtype=np.int32),
                ratings["game_id"].to_numpy(dtype=np.int32),
            ),
        ),
        shape=(int(ratings["user_id"].max()) + 1, int(ratings["game_id"].max()) + 1),
    )

    k_values = [int(k) for k in cfg.get("k_values", [10])]
    top_k = max(k_values)
    baseline_names = list(cfg.get("baselines", ["average", "least_misery", "random"]))
    agg_cfg = cfg.get("aggregation", {})

    rows: list[dict] = []
    known_user_ids = set(int(user_id) for user_id in model._user_id_map.keys())
    known_item_ids = set(int(item_id) for item_id in model._item_id_map.keys())
    for group_index, group in enumerate(groups_payload):
        group_users = np.asarray([user_id for user_id in group["user_ids"] if int(user_id) in known_user_ids], dtype=int)
        if group_users.size == 0:
            continue
        group_context = group.get("context", {})
        group_constraints = group.get("constraints", {})
        feasible = constraints.filter_feasible(games, group_constraints)
        feasible = feasible[feasible["game_id"].isin(known_item_ids)].copy()
        if feasible.empty:
            continue

        candidate_ids = feasible["game_id"].to_numpy(dtype=int)
        predicted = model.predict(group_users, candidate_ids)
        actual = rating_matrix[group_users][:, candidate_ids].toarray()
        actual_means = _mean_ratings_from_matrix(actual)
        relevance = {int(item_id): float(score) for item_id, score in zip(candidate_ids, actual_means)}
        relevant_set = set(candidate_ids[np.argsort(actual_means)[::-1][: min(top_k, len(candidate_ids))]])

        method_scores: dict[str, np.ndarray] = {
            "ours": context.adjust(aggregation.aggregate(predicted, **agg_cfg), feasible, group_context),
        }
        if "average" in baseline_names:
            method_scores["average"] = baselines.average_baseline(predicted)
        if "least_misery" in baseline_names:
            method_scores["least_misery"] = baselines.least_misery_baseline(predicted)
        if "random" in baseline_names:
            method_scores["random"] = baselines.random_baseline(len(candidate_ids), seed=(cfg.get("seed") or 0) + group_index)

        for method_name, scores in method_scores.items():
            for k in k_values:
                ranked_items = _top_items_from_scores(candidate_ids, scores, k)
                ranked_indices = np.argsort(scores)[::-1][:k]
                per_user_scores = predicted[:, ranked_indices].mean(axis=1)
                rows.append(
                    {
                        "group_id": group_index,
                        "method": method_name,
                        "k": k,
                        "ndcg": metrics.ndcg_at_k(ranked_items, relevance, k),
                        "precision": metrics.precision_at_k(ranked_items, relevant_set, k),
                        "rmse": metrics.rmse(actual_means, scores),
                        "satisfaction_variance": metrics.satisfaction_variance(per_user_scores),
                        "min_satisfaction": metrics.min_satisfaction(per_user_scores),
                    }
                )

    per_group = pd.DataFrame(rows)
    per_group.to_csv(results_dir / "per_group_metrics.csv", index=False)

    summary = (
        per_group.groupby(["method", "k"], as_index=False)
        .agg(
            ndcg=("ndcg", "mean"),
            precision=("precision", "mean"),
            rmse=("rmse", "mean"),
            satisfaction_variance=("satisfaction_variance", "mean"),
            min_satisfaction=("min_satisfaction", "mean"),
        )
    )
    summary.to_csv(results_dir / "metrics_summary.csv", index=False)

    for k in k_values:
        subset = summary[summary["k"] == k].copy()
        if subset.empty:
            continue
        _save_plots(subset.drop(columns=["k"]), results_dir / f"k_{k}")

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
