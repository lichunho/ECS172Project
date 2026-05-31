"""Smoke test for src/annotate.annotate_games against a live Ollama server.

Run with:
    python tests/test_annotate.py --config configs/data.yaml

Builds a small in-memory DataFrame of 3 well-known games, calls annotate_games,
then asserts label correctness and prints the resulting label table.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import annotate  # noqa: E402

# Allowed categorical values for coop_vs_competitive
_COOP_VALUES = {"coop", "semi-coop", "team", "competitive"}

# All expected label columns after annotation
_TIER1_LABELS = ["language_dependence", "min_age_fit", "complexity"]
_TIER2_LABELS = ["player_elimination", "coop_vs_competitive", "social_conflict"]
_TIER3_LABELS = [
    "party_friendliness",
    "interaction_level",
    "competitiveness",
    "downtime",
    "teach_time",
    "mixed_skill_robustness",
]
_ALL_LABELS = _TIER1_LABELS + _TIER2_LABELS + _TIER3_LABELS


def _build_test_games() -> pd.DataFrame:
    """Build a minimal in-memory DataFrame of 3 well-known games.

    Tier-1 optional fields are present for Codenames (to exercise parse path)
    and absent for Twilight Imperium (to exercise the pd.NA path).
    """
    return pd.DataFrame([
        {
            # Codenames: light party/word game, team-based cooperative play
            "game_id": 178900,
            "name": "Codenames",
            "description": (
                "Two teams compete to identify their agents by giving one-word "
                "clues that point to multiple words on the board, while avoiding "
                "the assassin. Fast, fun, great for parties."
            ),
            "categories": ["Word Game", "Party Game"],
            "mechanics": ["Communication Limits", "Team-Based Game", "Voting"],
            "weight": 1.3,
            # Tier-1 optional fields present
            "minage": 14,
            "suggested_playerage": 10,
            "language_dependence_poll": 4,  # heavy language dependence
        },
        {
            # Twilight Imperium 4e: heavy, competitive, long, negotiation
            "game_id": 233078,
            "name": "Twilight Imperium (Fourth Edition)",
            "description": (
                "An epic game of galactic conquest, politics, and trade. Players "
                "build space empires and vie for control of the galaxy via military, "
                "economic, and political power over many hours."
            ),
            "categories": ["Science Fiction", "Space Exploration", "Wargame"],
            "mechanics": ["Area Majority / Influence", "Negotiation", "Variable Player Powers"],
            "weight": 4.3,
            # Tier-1 optional fields absent — exercises pd.NA path
        },
        {
            # Pandemic: fully cooperative, accessible, low conflict
            "game_id": 30549,
            "name": "Pandemic",
            "description": (
                "Players work as a team of disease-control specialists to treat "
                "infections around the world while gathering resources for cures. "
                "Fully cooperative — players win or lose together."
            ),
            "categories": ["Medical", "Cooperative Game"],
            "mechanics": ["Cooperative Game", "Hand Management", "Point to Point Movement"],
            "weight": 2.4,
            "minage": 8,
            # No language_dependence_poll for this row (optional)
        },
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test for annotate_games.")
    parser.add_argument("--config", required=True, help="Path to data YAML config.")
    args = parser.parse_args()

    load_dotenv()
    host = os.environ.get("OLLAMA_HOST", "").strip()
    port = os.environ.get("OLLAMA_PORT", "11434").strip()
    if not host:
        print("OLLAMA_HOST is not set. Copy .env.example to .env and set it.")
        return 1

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    llm_cfg = cfg["llm"]
    model = llm_cfg["model"]
    timeout = float(llm_cfg.get("timeout_s", 120))
    options = llm_cfg.get("options")

    print(f"Server: http://{host}:{port}  model: {model}")
    print("Building test games DataFrame…")
    games = _build_test_games()

    print("Calling annotate_games (Tier 3 makes 3 LLM requests)…\n")
    try:
        result = annotate.annotate_games(
            games, host=host, port=port, model=model, options=options, timeout=timeout
        )
    except Exception as exc:
        print(f"ERROR: annotate_games raised: {exc}")
        return 1

    # ---- assertions --------------------------------------------------------

    errors: list[str] = []

    # 1. All label columns must exist
    for col in _ALL_LABELS:
        if col not in result.columns:
            errors.append(f"Missing label column: {col}")

    # 2. Tier-3 labels must be ints in [1,5] for all rows
    for col in _TIER3_LABELS:
        if col not in result.columns:
            continue
        for idx, val in result[col].items():
            if not isinstance(val, int) or not (1 <= val <= 5):
                errors.append(f"{col}[row {idx}]={val!r} is not an int in 1–5")

    # 3. coop_vs_competitive must be one of the allowed strings
    if "coop_vs_competitive" in result.columns:
        for idx, val in result["coop_vs_competitive"].items():
            if val not in _COOP_VALUES:
                errors.append(
                    f"coop_vs_competitive[row {idx}]={val!r} not in {_COOP_VALUES}"
                )

    # 4. Pandemic (row index 2) must come back "coop"
    if "coop_vs_competitive" in result.columns:
        pandemic_val = result.loc[2, "coop_vs_competitive"]
        if pandemic_val != "coop":
            errors.append(
                f"Pandemic coop_vs_competitive={pandemic_val!r}, expected 'coop'"
            )

    # 5. Twilight Imperium (row 1) — Tier-1 optional fields were omitted from
    #    input, so min_age_fit must be pd.NA.
    for col in ["min_age_fit"]:
        if col in result.columns:
            val = result.loc[1, col]
            if not pd.isna(val):
                errors.append(
                    f"TI4 {col}={val!r} expected pd.NA (optional input field absent)"
                )

    # ---- print results -----------------------------------------------------

    label_cols = [c for c in _ALL_LABELS if c in result.columns]
    print("Results (label columns only):")
    print(result[["name"] + label_cols].to_string(index=False))

    if errors:
        print("\nASSERTION FAILURES:")
        for e in errors:
            print(f"  ✗ {e}")
        return 1

    print("\nAll assertions passed.")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
