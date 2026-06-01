"""LLM-based and rule-based annotation of board game context features.

Three-tier labeling strategy (see plan_m2.md):
  Tier 1 — reshaped from BGG structured fields (no LLM)
  Tier 2 — derived from category/mechanic tag lookup (no LLM)
  Tier 3 — LLM judgment via Ollama (src/llm.py)

Single job: given a games DataFrame, produce one row per game with every
context label column appended.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src import llm


# ---------------------------------------------------------------------------
# Tier-2 lookup table: mechanic/category tokens → label values
# Match case-insensitively against the union of a game's categories+mechanics.
# ---------------------------------------------------------------------------

# Maps a token substring to (label_name, value).
# Value is 1 (absent/false) or 5 (present/true) for binaries; a string for
# coop_vs_competitive. Evaluated top-to-bottom; first match per label wins.
_TIER2_RULES: list[tuple[str, str, int | str]] = [
    # player_elimination: eliminated players sit idle → fairness hazard
    ("player elimination",  "player_elimination", 5),

    # coop_vs_competitive: structural trust/conflict level of the game
    # Most specific first (semi-coop, team-based) before generic "cooperative"
    ("semi-cooperative",    "coop_vs_competitive", "semi-coop"),
    ("semi cooperative",    "coop_vs_competitive", "semi-coop"),
    ("team-based",          "coop_vs_competitive", "team"),
    ("team based",          "coop_vs_competitive", "team"),
    ("cooperative",         "coop_vs_competitive", "coop"),
    # "competitive" is the default fallback; set after all rows processed

    # social_conflict: negotiation / take-that → friction, kingmaking risk
    ("negotiation",         "social_conflict", 5),
    ("take that",           "social_conflict", 5),
    ("take-that",           "social_conflict", 5),
    ("trading",             "social_conflict", 5),
]

# Default values for Tier-2 labels when no rule fires
_TIER2_DEFAULTS: dict[str, int | str] = {
    "player_elimination": 1,
    "coop_vs_competitive": "competitive",
    "social_conflict": 1,
}

# ---------------------------------------------------------------------------
# Tier-3 JSON schema for LLM-constrained decoding
# ---------------------------------------------------------------------------

_TIER3_LABELS = [
    "party_friendliness",
    "interaction_level",
    "competitiveness",
    "downtime",
    "teach_time",
    "mixed_skill_robustness",
]

_TIER3_SCHEMA: dict = {
    "type": "object",
    "properties": {lbl: {"type": "integer"} for lbl in _TIER3_LABELS},
    "required": _TIER3_LABELS,
}

_TIER3_PROMPT_TEMPLATE = """\
Rate the board game "{name}" on each dimension below as an integer from 1 to 5.
Use only the provided scale; output strict JSON with no extra keys.

Description: {description}
Categories: {categories}
Mechanics: {mechanics}
BGG weight (1=light, 5=heavy): {weight}

Scale definitions:
  party_friendliness   — 1=terrible for casual/party play, 5=ideal for it
  interaction_level    — 1=almost zero player interaction, 5=highly interactive
  competitiveness      — 1=gentle/cooperative feel, 5=cutthroat competitive
  downtime             — 1=minimal waiting between turns, 5=very long downtime
  teach_time           — 1=rules explained in under 2 min, 5=hours of learning
  mixed_skill_robustness — 1=experts crush novices, 5=game self-balances mixed skill
"""


# ---------------------------------------------------------------------------
# Helper: normalise categories/mechanics to a single lower-cased string
# ---------------------------------------------------------------------------

def _tags_string(value: object) -> str:
    """Return a lower-cased concatenation of category/mechanic tokens.

    Accepts a list/ndarray (from parquet) or a delimited string (pipe, comma,
    semicolon) — BGG data arrives in either form depending on the source.
    """
    if isinstance(value, (list, np.ndarray)):
        return " | ".join(str(v) for v in value).lower()
    if pd.isna(value):
        return ""
    return str(value).lower()


# ---------------------------------------------------------------------------
# Tier 1 helpers
# ---------------------------------------------------------------------------

def _complexity_from_weight(weight: object) -> int | float:
    """Round BGG averageweight (float 1.0–5.0) to an integer 1–5.

    Returns pandas NA if the value is missing or non-numeric.
    """
    try:
        w = float(weight)
    except (TypeError, ValueError):
        return pd.NA
    if math.isnan(w):
        return pd.NA
    return max(1, min(5, round(w)))


def _tier1_row(row: pd.Series) -> dict:
    """Extract Tier-1 label values for one game row.

    Returns a dict of {label: value | pd.NA}.
    """
    result: dict = {}

    # complexity — directly from BGG averageweight
    result["complexity"] = _complexity_from_weight(row.get("weight"))

    # language_dependence — BGG 5-level poll (1=no dependence, 5=unplayable
    # without language). If M1 hasn't parsed it yet, leave NA.
    ld = row.get("language_dependence_poll")
    result["language_dependence"] = pd.NA if pd.isna(ld) else int(max(1, min(5, round(float(ld)))))

    # min_age_fit — derived from minage / suggested_playerage.
    # M1 may expose either or both; we use whichever is present.
    # Remap raw age (e.g. 10) to a 1–5 accessibility scale:
    #   ≤6 → 5 (everyone), 7–9 → 4, 10–12 → 3, 13–15 → 2, 16+ → 1
    age_val = row.get("suggested_playerage")
    if pd.isna(age_val):
        age_val = row.get("minage")
    if pd.isna(age_val):
        result["min_age_fit"] = pd.NA
    else:
        try:
            age = float(age_val)
        except (TypeError, ValueError):
            result["min_age_fit"] = pd.NA
        else:
            if age <= 6:
                result["min_age_fit"] = 5
            elif age <= 9:
                result["min_age_fit"] = 4
            elif age <= 12:
                result["min_age_fit"] = 3
            elif age <= 15:
                result["min_age_fit"] = 2
            else:
                result["min_age_fit"] = 1

    return result


def _apply_tier1(games: pd.DataFrame) -> pd.DataFrame:
    """Add Tier-1 label columns to *games* (in-place copy). Returns new df."""
    tier1_rows = [_tier1_row(row) for _, row in games.iterrows()]
    tier1_df = pd.DataFrame(tier1_rows, index=games.index)
    return pd.concat([games, tier1_df], axis=1)


# ---------------------------------------------------------------------------
# Tier 2 helpers
# ---------------------------------------------------------------------------

def _tier2_row(row: pd.Series) -> dict:
    """Derive Tier-2 label values for one game via the rule table."""
    tags = _tags_string(row.get("categories", "")) + " " + _tags_string(row.get("mechanics", ""))

    result: dict = {lbl: default for lbl, default in _TIER2_DEFAULTS.items()}

    for token, label, value in _TIER2_RULES:
        # Stop updating a label once its first (most specific) rule fires.
        if token in tags and result[label] == _TIER2_DEFAULTS[label]:
            result[label] = value

    return result


def _apply_tier2(games: pd.DataFrame) -> pd.DataFrame:
    """Add Tier-2 label columns. Returns new df."""
    tier2_rows = [_tier2_row(row) for _, row in games.iterrows()]
    tier2_df = pd.DataFrame(tier2_rows, index=games.index)
    return pd.concat([games, tier2_df], axis=1)


# ---------------------------------------------------------------------------
# Tier 3 helpers
# ---------------------------------------------------------------------------

def _build_prompt(row: pd.Series) -> str:
    cats  = _tags_string(row.get("categories", ""))
    mechs = _tags_string(row.get("mechanics", ""))
    desc  = str(row.get("description", "")).strip()[:1000]  # avoid token overflow
    weight = row.get("weight", "unknown")
    return _TIER3_PROMPT_TEMPLATE.format(
        name=row.get("name", "Unknown"),
        description=desc or "(no description provided)",
        categories=cats or "(none)",
        mechanics=mechs or "(none)",
        weight=weight,
    )


def _parse_tier3_response(content: str, game_name: str) -> dict:
    """Parse and clamp the LLM JSON response for one game.

    Raises ValueError with the game name if JSON is invalid or a required key
    is missing / not an integer.
    """
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Tier-3 LLM returned invalid JSON for '{game_name}': {exc}\nRaw: {content!r}"
        ) from exc

    result: dict = {}
    for lbl in _TIER3_LABELS:
        if lbl not in data:
            raise ValueError(
                f"Tier-3 LLM response for '{game_name}' is missing key '{lbl}'. "
                f"Got: {list(data.keys())}"
            )
        val = data[lbl]
        if not isinstance(val, int):
            # tolerate float-encoded ints from some models
            try:
                val = int(val)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Tier-3 label '{lbl}' for '{game_name}' is not an integer: {val!r}"
                )
        result[lbl] = max(1, min(5, val))  # clamp to [1,5]

    return result


def _load_tier3_checkpoint(path: str | os.PathLike | None) -> dict:
    """Load {game_id: {label: value}} from a checkpoint parquet, or {} if absent."""
    if path is None or not Path(path).exists():
        return {}
    df = pd.read_parquet(path)
    return {
        row["game_id"]: {lbl: row[lbl] for lbl in _TIER3_LABELS}
        for _, row in df.iterrows()
    }


def _save_tier3_checkpoint(path: str | os.PathLike, done: dict) -> None:
    """Atomically write {game_id: {label: value}} to a checkpoint parquet."""
    rows = [{"game_id": gid, **labels} for gid, labels in done.items()]
    df = pd.DataFrame(rows, columns=["game_id", *_TIER3_LABELS])
    tmp = Path(str(path) + ".tmp")
    df.to_parquet(tmp)
    os.replace(tmp, path)


def _tier3_one(
    row: pd.Series,
    *,
    host: str,
    port: str | int,
    model: str,
    options: dict | None,
    timeout: float,
) -> tuple[object, dict]:
    """Annotate one game via the LLM. Returns (game_id, labels)."""
    game_id = row.get("game_id")
    game_name = str(row.get("name", f"id={game_id}"))
    messages = [{"role": "user", "content": _build_prompt(row)}]
    content = llm.chat(
        host, port, model, messages,
        options=options,
        fmt=_TIER3_SCHEMA,
        timeout=timeout,
    )
    return game_id, _parse_tier3_response(content, game_name)


def _apply_tier3(
    games: pd.DataFrame,
    *,
    host: str,
    port: str | int,
    model: str,
    options: dict | None,
    timeout: float,
    concurrency: int = 1,
    checkpoint_path: str | os.PathLike | None = None,
    checkpoint_every: int = 100,
    tier3_ids: set | None = None,
) -> pd.DataFrame:
    """Call the LLM for each game (concurrently) and add Tier-3 label columns.

    Resumable: games already present in the checkpoint at *checkpoint_path* are
    skipped, and results are flushed there every *checkpoint_every* completions
    so a crash resumes where it left off. A game whose call fails is logged and
    left out of the checkpoint, so the next run retries it.

    If *tier3_ids* is given, only those game_ids are candidates for the LLM pass;
    every other game gets NA Tier-3 labels (backfill later by widening the set).
    """
    done = _load_tier3_checkpoint(checkpoint_path)
    candidates = games if tier3_ids is None else games[games["game_id"].isin(tier3_ids)]
    todo = candidates[~candidates["game_id"].isin(done.keys())]
    print(
        f"Tier-3: {len(done)} cached, {len(todo)} to annotate "
        f"(concurrency={concurrency})"
    )

    if len(todo) > 0:
        with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {
                ex.submit(
                    _tier3_one, row,
                    host=host, port=port, model=model,
                    options=options, timeout=timeout,
                ): row.get("game_id")
                for _, row in todo.iterrows()
            }
            since_flush = 0
            for fut in cf.as_completed(futures):
                gid = futures[fut]
                try:
                    game_id, labels = fut.result()
                except Exception as exc:  # one bad game must not abort the run
                    print(f"  WARN Tier-3 failed for game_id={gid}: {exc} (retry next run)")
                    continue
                done[game_id] = labels
                since_flush += 1
                if checkpoint_path is not None and since_flush >= checkpoint_every:
                    _save_tier3_checkpoint(checkpoint_path, done)
                    since_flush = 0
                    print(f"  checkpoint: {len(done)}/{len(games)} done")
            if checkpoint_path is not None:
                _save_tier3_checkpoint(checkpoint_path, done)

    na_row = {lbl: pd.NA for lbl in _TIER3_LABELS}
    tier3_rows = [done.get(row.get("game_id"), na_row) for _, row in games.iterrows()]
    tier3_df = pd.DataFrame(tier3_rows, index=games.index)
    return pd.concat([games, tier3_df], axis=1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def annotate_games(
    games: pd.DataFrame,
    *,
    host: str,
    port: str | int,
    model: str,
    options: dict | None,
    timeout: float,
    concurrency: int = 1,
    checkpoint_path: str | os.PathLike | None = None,
    checkpoint_every: int = 100,
    tier3_ids: set | None = None,
) -> pd.DataFrame:
    """Annotate each game with context labels across three tiers.

    Returns a new DataFrame with all original columns plus the label columns
    listed below. Row order and index are preserved.

    INPUT COLUMN CONTRACT
    ---------------------
    Required (must be present for any annotation to work):
      game_id      — unique game identifier
      name         — human-readable game title
      description  — BGG game description text (may be empty string)
      categories   — list[str] OR pipe/comma/semicolon-delimited string of BGG categories
      mechanics    — list[str] OR pipe/comma/semicolon-delimited string of BGG mechanics
      weight       — BGG averageweight float in [1.0, 5.0] (used for complexity + Tier-3 prompt)

    Optional Tier-1 fields (per-row; missing rows get pd.NA for that label):
      language_dependence_poll — numeric language dependence level (1–5); produced by M1
      minage                   — BGG minimum age (raw integer)
      suggested_playerage      — BGG suggested player age from poll (raw integer)

    OUTPUT LABEL COLUMNS
    --------------------
    Tier 1 (no LLM — reshaped BGG fields):
      language_dependence    int 1–5, or pd.NA if field absent
      min_age_fit            int 1–5, or pd.NA if field absent
      complexity             int 1–5, or pd.NA if weight absent

    Tier 2 (no LLM — rule lookup from categories + mechanics):
      player_elimination     int 1 (absent) or 5 (present)
      coop_vs_competitive    str in {"coop","semi-coop","team","competitive"}
      social_conflict        int 1 (absent) or 5 (present)

    Tier 3 (LLM judgment):
      party_friendliness     int 1–5
      interaction_level      int 1–5
      competitiveness        int 1–5
      downtime               int 1–5
      teach_time             int 1–5
      mixed_skill_robustness int 1–5
    """
    out = _apply_tier1(games)
    out = _apply_tier2(out)
    out = _apply_tier3(
        out,
        host=host, port=port, model=model, options=options, timeout=timeout,
        concurrency=concurrency,
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
        tier3_ids=tier3_ids,
    )
    return out
