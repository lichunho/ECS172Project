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

    src_path = processed_dir / "games.parquet"
    dst_path = processed_dir / "games_annotated.parquet"

    # Archive any existing annotated file before overwriting.
    if dst_path.exists():
        archive_dir = Path("archive") / f"{datetime.date.today().isoformat()}_annotations"
        archive_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(dst_path), str(archive_dir / "games_annotated.parquet"))
        print(f"Archived existing annotations → {archive_dir}/")

    games = pd.read_parquet(src_path)
    print(f"Loaded {len(games)} games from {src_path}")

    annotated = annotate.annotate_games(
        games,
        host=host,
        port=port,
        model=model,
        options=options,
        timeout=timeout,
    )

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.to_parquet(dst_path)
    print(f"Wrote {len(annotated)} annotated games → {dst_path}")


if __name__ == "__main__":
    main()
