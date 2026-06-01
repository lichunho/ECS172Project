"""Run full evaluation: ours vs. baselines on simulated groups.

Leak-free protocol: a fresh PreferenceModel is trained here on a per-user
train/test split (independent of scripts/train.py's full-data model). Held-out
ratings are the ground truth. For each simulated group the candidate pool is the
union of members' held-out items; each method ranks that pool, and we score the
ranking against every member's held-out ratings to get accuracy (NDCG@k, P@k) and
fairness (variance / min of per-member NDCG).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import aggregation, baselines, constraints, context, metrics  # noqa: E402
from src.groups import Group  # noqa: E402
from src.preference import PreferenceModel  # noqa: E402

_METHODS = ["ours", "ours_no_context", "average", "least_misery", "random"]


def _split(ratings: pd.DataFrame, test_frac: float, seed: int):
    """Per-user train/test split; returns (train_df, test_df)."""
    shuffled = ratings.sample(frac=1.0, random_state=seed)
    rank = shuffled.groupby("user_id").cumcount()
    n = shuffled.groupby("user_id")["user_id"].transform("size")
    test_mask = (rank < (n * test_frac)).to_numpy()
    return shuffled[~test_mask].copy(), shuffled[test_mask].copy()


def _group_scores(method, score_matrix, candidates, feasible, ctx, agg_cfg, seed, gamma=None):
    if method == "average":
        return baselines.average_baseline(score_matrix)
    if method == "least_misery":
        return baselines.least_misery_baseline(score_matrix)
    if method == "random":
        return baselines.random_baseline(len(candidates), seed=seed)
    # ours_no_context: fairness aggregation only (context layer ablated).
    gs = aggregation.aggregate(score_matrix, **agg_cfg)
    if method == "ours_no_context":
        return gs
    # ours: fairness aggregation + context adjustment
    return context.adjust(gs, feasible, ctx, gamma=gamma)


def _evaluate(cfg: dict, context_gamma: float | None = None):
    """Run the full group evaluation and return (table, model_rmse, n_groups, skipped).

    Pure compute: reads inputs and returns a tidy metrics DataFrame; the caller
    owns persistence. `context_gamma` overrides the context re-weighting strength
    for the 'ours' method (None = module default), which the gamma sweep varies.
    """
    processed_dir = Path(cfg["processed_dir"])
    out_dir = Path("results") / cfg["run_name"]
    seed = cfg["seed"]
    k_values = cfg["k_values"]
    agg_cfg = cfg["aggregation"]
    threshold = cfg["relevant_threshold"]

    ratings = pd.read_parquet(processed_dir / "ratings.parquet", columns=["user_id", "game_id", "rating"])
    games = pd.read_parquet(processed_dir / "games_annotated.parquet")
    games_by_id = games.set_index("game_id")

    with open(out_dir / "groups.json") as f:
        sim_groups = [Group(**g) for g in json.load(f)]

    print("Splitting ratings and fitting eval model on the train split...")
    train, test = _split(ratings, cfg["test_frac"], seed)
    model = PreferenceModel(
        no_components=cfg["model"]["no_components"],
        bias_reg=cfg["model"].get("bias_reg", 5.0),
    ).fit(train)

    # Model-level RMSE on held-out ratings.
    pred = model.predict_pairs(test["user_id"].to_numpy(), test["game_id"].to_numpy())
    model_rmse = metrics.rmse(test["rating"].to_numpy(), pred)
    print(f"  held-out RMSE: {model_rmse:.4f}")

    # Per-user held-out items/ratings for fast lookup.
    test_by_user: dict[int, dict[int, float]] = {
        uid: dict(zip(grp["game_id"].to_numpy(), grp["rating"].to_numpy()))
        for uid, grp in test.groupby("user_id")
    }

    # records[(method, k, scope)] -> list of (ndcg, precision, sat_var, min_sat)
    records: dict[tuple, list] = defaultdict(list)
    skipped = 0

    for g in sim_groups:
        members = [u for u in g.user_ids if u in test_by_user]
        if len(members) < 2:
            skipped += 1
            continue

        pool = sorted({iid for u in members for iid in test_by_user[u]})
        if len(pool) < 2:
            skipped += 1
            continue

        feasible = constraints.filter_feasible(games_by_id.loc[pool].reset_index(), g.constraints)
        candidates = feasible["game_id"].to_numpy()
        if len(candidates) < 2:
            skipped += 1
            continue

        score_matrix = model.predict(np.asarray(members), candidates)

        for method in _METHODS:
            gs = _group_scores(method, score_matrix, candidates, feasible,
                               g.context, agg_cfg, seed, gamma=context_gamma)
            order = np.argsort(-gs)
            ranked = candidates[order]

            for k in k_values:
                member_ndcgs = []
                member_precs = []
                for m in members:
                    rel_map = test_by_user[m]
                    rel_in_order = np.array([rel_map.get(int(it), 0.0) for it in ranked])
                    member_ndcgs.append(metrics.ndcg_at_k(ranked, rel_in_order, k))
                    relevant = {it for it, r in rel_map.items() if r >= threshold}
                    member_precs.append(metrics.precision_at_k(ranked, relevant, k))

                ndcg = float(np.mean(member_ndcgs))
                prec = float(np.mean(member_precs))
                # Fairness in rating space: each member's mean predicted utility over
                # the top-k recommended items. Decoupled from ranking-accuracy
                # magnitude, so variance/min measure equity rather than echoing NDCG.
                member_sat = score_matrix[:, order[:k]].mean(axis=1)
                sat_var = metrics.satisfaction_variance(member_sat)
                min_sat = metrics.min_satisfaction(member_sat)
                row = (ndcg, prec, sat_var, min_sat)
                records[(method, k, "overall")].append(row)
                records[(method, k, g.group_type)].append(row)

    # Aggregate to a tidy table.
    rows = []
    for (method, k, scope), vals in sorted(records.items()):
        arr = np.array(vals)
        rows.append({
            "method": method, "k": k, "scope": scope, "n_groups": len(vals),
            "ndcg": arr[:, 0].mean(), "precision": arr[:, 1].mean(),
            "sat_variance": arr[:, 2].mean(), "min_satisfaction": arr[:, 3].mean(),
        })
    table = pd.DataFrame(rows)
    return table, model_rmse, len(sim_groups), skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path("results") / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    table, model_rmse, n_groups, skipped = _evaluate(cfg)

    table.to_csv(out_dir / "metrics.csv", index=False)
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"model_rmse": model_rmse, "n_groups": n_groups,
                   "skipped": skipped}, f, indent=2)

    _plots(table, cfg["k_values"], out_dir)
    print(f"\nSkipped {skipped} groups (too few feasible candidates).")
    print(table[table["scope"] == "overall"].to_string(index=False))
    print(f"\nWrote metrics -> {out_dir/'metrics.csv'} and plots -> {out_dir}")


def _plots(table: pd.DataFrame, k_values, out_dir: Path) -> None:
    k = k_values[-1]
    overall = table[(table["scope"] == "overall") & (table["k"] == k)]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(overall["method"], overall["ndcg"])
    ax.set_ylabel(f"mean NDCG@{k}")
    ax.set_title(f"Accuracy by method (NDCG@{k})")
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_by_method.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for _, r in overall.iterrows():
        ax.scatter(r["sat_variance"], r["ndcg"], s=80)
        ax.annotate(r["method"], (r["sat_variance"], r["ndcg"]),
                    textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel(f"satisfaction variance @{k}  (lower = fairer)")
    ax.set_ylabel(f"mean NDCG@{k}  (higher = more accurate)")
    ax.set_title("Accuracy vs. fairness trade-off")
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_vs_fairness.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
