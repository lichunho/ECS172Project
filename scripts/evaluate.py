"""Run full evaluation: ours vs. baselines on simulated groups."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import aggregation, baselines, constraints, context, metrics  # noqa: E402
from src.preference import PreferenceModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # 1. Load groups produced by simulate_groups.py.
    # 2. For each group: apply constraint filtering, score with model + ours,
    #    score with each baseline, then compute NDCG@K, P@K, RMSE, satisfaction
    #    variance, min satisfaction.
    # 3. Aggregate per-method metrics into a table; write CSV + plots under
    #    results/<run_name>/.
    raise NotImplementedError("scaffold only; orchestration outlined above")


if __name__ == "__main__":
    main()
