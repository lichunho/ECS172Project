"""Sample evaluation groups from the processed ratings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import groups  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # users = pd.read_parquet(...)["user_id"].unique()
    # sampled = groups.sample_groups(users, cfg["n_groups"], cfg["group_sizes"], cfg["seed"])
    # persist to results/<run_name>/groups.json
    raise NotImplementedError("scaffold only; see src/groups.py")


if __name__ == "__main__":
    main()
