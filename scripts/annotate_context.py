"""Add LLM-annotated and rule-derived context features to the games dataset.

Reads:  <processed_dir>/games.parquet
Writes: <processed_dir>/games_annotated.parquet

Archives any existing games_annotated.parquet to
  archive/<YYYY-MM-DD>_annotations/games_annotated.parquet
before overwriting (CLAUDE.md archive-before-destructive-change rule).
"""

from __future__ import annotations

import argparse
import datetime
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import annotate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate games with context labels.")
    parser.add_argument("--config", required=True, help="Path to data YAML config.")
    args = parser.parse_args()

    load_dotenv()
    host = os.environ.get("OLLAMA_HOST", "").strip()
    port = os.environ.get("OLLAMA_PORT", "11434").strip()
    if not host:
        print("ERROR: OLLAMA_HOST is not set. Copy .env.example to .env and set it.")
        sys.exit(1)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    processed_dir = Path(cfg["processed_dir"])
    llm_cfg = cfg["llm"]
    model = llm_cfg["model"]
    timeout = float(llm_cfg.get("timeout_s", 120))
    options = llm_cfg.get("options")
    concurrency = int(llm_cfg.get("concurrency", 1))
    checkpoint_every = int(llm_cfg.get("checkpoint_every", 100))

    src_path = processed_dir / "games.parquet"
    dst_path = processed_dir / "games_annotated.parquet"
    # Persists across runs to enable Tier-3 resume; NOT archived with the output.
    checkpoint_path = processed_dir / "games_annotated.tier3_checkpoint.parquet"

    # Archive any existing annotated file before overwriting.
    if dst_path.exists():
        archive_dir = Path("archive") / f"{datetime.date.today().isoformat()}_annotations"
        archive_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(dst_path), str(archive_dir / "games_annotated.parquet"))
        print(f"Archived existing annotations -> {archive_dir}/")

    games = pd.read_parquet(src_path)
    print(f"Loaded {len(games)} games from {src_path}")

    # Restrict the expensive Tier-3 LLM pass to games with enough ratings.
    tier3_ids = None
    min_ratings = cfg.get("annotation", {}).get("tier3_min_ratings")
    if min_ratings:
        counts = pd.read_parquet(processed_dir / "ratings.parquet")["game_id"].value_counts()
        tier3_ids = set(counts[counts >= int(min_ratings)].index)
        print(f"Tier-3 scope: {len(tier3_ids)} games with >= {min_ratings} ratings")

    annotated = annotate.annotate_games(
        games,
        host=host,
        port=port,
        model=model,
        options=options,
        timeout=timeout,
        concurrency=concurrency,
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
        tier3_ids=tier3_ids,
    )

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.to_parquet(dst_path)
    print(f"Wrote {len(annotated)} annotated games -> {dst_path}")


if __name__ == "__main__":
    main()
