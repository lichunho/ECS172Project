"""Sweep the context re-weighting strength (context.GAMMA) for the 'ours' method.

Re-runs the leak-free group evaluation at several gamma values and reports how
the 'ours' overall accuracy/fairness moves, so we can pick the point where the
context layer stops costing accuracy. Reference methods (average, least_misery,
ours_no_context) are gamma-invariant and printed once for comparison.

    python scripts/sweep_gamma.py --config configs/eval.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate import _evaluate  # noqa: E402

GAMMAS = [0.0, 0.1, 0.25, 0.5]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path("results") / cfg["run_name"]
    cols = ["gamma", "method", "k", "ndcg", "precision", "sat_variance", "min_satisfaction"]
    collected = []
    refs = None

    for gamma in GAMMAS:
        print(f"\n=== gamma = {gamma} ===")
        table, _, _, _ = _evaluate(cfg, context_gamma=gamma)
        overall = table[table["scope"] == "overall"].copy()
        ours = overall[overall["method"] == "ours"].copy()
        ours["gamma"] = gamma
        collected.append(ours[cols])
        if refs is None:
            refs = overall[overall["method"].isin(["average", "least_misery", "ours_no_context"])].copy()
            refs["gamma"] = "n/a"

    sweep = pd.concat([refs[cols]] + collected, ignore_index=True)
    sweep.to_csv(out_dir / "gamma_sweep.csv", index=False)

    print("\n==== gamma sweep (overall scope) ====")
    print(sweep.sort_values(["k", "gamma"], kind="stable").to_string(index=False))
    print(f"\nWrote sweep -> {out_dir/'gamma_sweep.csv'}")


if __name__ == "__main__":
    main()
