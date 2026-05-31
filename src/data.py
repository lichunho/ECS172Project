"""BoardGameGeek data loading and preprocessing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# Cat:* column names in games.csv → label values in the categories list
_CAT_COLS = [
    "Cat:Thematic",
    "Cat:Strategy",
    "Cat:War",
    "Cat:Family",
    "Cat:CGS",
    "Cat:Abstract",
    "Cat:Party",
    "Cat:Childrens",
]


def _parse_good_player_counts(val) -> list[int]:
    """Parse a GoodPlayers string like \"['2','3','4+']\" into a list of ints.

    Each token's leading integer is used, so '4+' and '4 ' both become 4.
    Empty / NaN values return an empty list.
    """
    if not isinstance(val, str) or not val.strip():
        return []
    # Strip outer brackets/whitespace and split on commas
    val = val.strip().strip("[]")
    if not val:
        return []
    result = []
    for token in val.split(","):
        token = token.strip().strip("'\"")
        # Extract leading integer (handles '4+', '4 players', etc.)
        digits = ""
        for ch in token:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits:
            result.append(int(digits))
    return result


def load_ratings(raw_dir: str | Path) -> pd.DataFrame:
    """Load raw user-game ratings from user_ratings.csv.

    Returns DataFrame with columns: user_id (int32), game_id (int32 holding raw
    BGGId — will be remapped to a shared factorized id in preprocess), rating (float32).

    NOTE on game_id: we keep the raw BGGId here so that preprocess can build a
    single shared BGGId→game_id mapping from games.parquet after alignment.
    user_id is factorized here because users exist only in this table.
    """
    raw_dir = Path(raw_dir)
    df = pd.read_csv(
        raw_dir / "user_ratings.csv",
        usecols=["Username", "Rating", "BGGId"],
        dtype={"BGGId": "int32", "Rating": "float32"},
    )

    # Drop unrated rows. BGG ratings live on a 1–10 scale, so anything below 1
    # (including the 0 = "unrated" sentinel) is not a real score.
    df = df[df["Rating"] >= 1].copy()

    # Drop rows with a missing username — pd.factorize would otherwise map NaN
    # to the -1 sentinel and collapse them into one phantom user.
    df = df[df["Username"].notna()]

    # Dedup (Username, BGGId) keeping last occurrence
    df = df.drop_duplicates(subset=["Username", "BGGId"], keep="last")

    # Factorize usernames → dense int32 user_id
    codes, _ = pd.factorize(df["Username"])
    df["user_id"] = codes.astype("int32")

    # Rename BGGId → game_id (raw BGGId; remapped in preprocess)
    df = df.rename(columns={"BGGId": "game_id", "Rating": "rating"})

    return df[["user_id", "game_id", "rating"]].reset_index(drop=True)


def load_games(raw_dir: str | Path) -> pd.DataFrame:
    """Load game metadata from games.csv joined with mechanics.csv.

    Returns DataFrame with the canonical games schema. game_id holds the raw
    BGGId (int32); the shared factorized mapping is applied in preprocess after
    alignment with ratings so the mapping lives in exactly one place.
    """
    raw_dir = Path(raw_dir)

    # --- read games.csv ---
    games_df = pd.read_csv(raw_dir / "games.csv")

    # --- read mechanics.csv ---
    mech_df = pd.read_csv(raw_dir / "mechanics.csv")

    # --- join: left join so every game is kept even if absent from mechanics ---
    # copy() defragments the wide merged frame before we add list-type columns
    df = games_df.merge(mech_df, on="BGGId", how="left").copy()

    # --- reshape Cat:* one-hot columns → list of category labels ---
    # Strip 'Cat:' prefix so values are e.g. 'Thematic', 'Strategy'
    def _onehot_to_list(row, cols, prefix=""):
        return [col[len(prefix):] for col in cols if row.get(col, 0) == 1]

    df["categories"] = df.apply(
        lambda row: _onehot_to_list(row, _CAT_COLS, prefix="Cat:"), axis=1
    )

    # --- reshape one-hot mechanic columns (all columns in mech_df except BGGId) ---
    mechanic_cols = [c for c in mech_df.columns if c != "BGGId"]

    def _mechanics_list(row):
        return [col for col in mechanic_cols if row.get(col, 0) == 1]

    df["mechanics"] = df.apply(_mechanics_list, axis=1)

    # --- numeric: replace 0 → NaN for fields where 0 means "not provided" ---
    zero_to_nan_cols = [
        "GameWeight", "MinPlayers", "MaxPlayers",
        "ComMinPlaytime", "ComMaxPlaytime", "MfgAgeRec", "ComAgeRec",
        "BestPlayers",
    ]
    for col in zero_to_nan_cols:
        if col in df.columns:
            df[col] = df[col].replace(0, pd.NA).astype("Float64")

    # --- parse GoodPlayers string → list[int] ---
    df["good_player_counts"] = df["GoodPlayers"].apply(_parse_good_player_counts)

    # --- select and rename to canonical schema ---
    out = df[[
        "BGGId", "Name", "Description",
        "categories", "mechanics",
        "GameWeight", "MinPlayers", "MaxPlayers",
        "ComMinPlaytime", "ComMaxPlaytime",
        "MfgAgeRec", "ComAgeRec",
        "BestPlayers", "good_player_counts",
    ]].rename(columns={
        "BGGId": "game_id",
        "Name": "name",
        "Description": "description",
        "GameWeight": "weight",
        "MinPlayers": "min_players",
        "MaxPlayers": "max_players",
        "ComMinPlaytime": "min_playtime",
        "ComMaxPlaytime": "max_playtime",
        "MfgAgeRec": "minage",
        "ComAgeRec": "suggested_playerage",
        "BestPlayers": "best_player_count",
    })

    # Cast game_id to int32 (raw BGGId; remapped in preprocess)
    out = out.copy()
    out["game_id"] = out["game_id"].astype("int32")

    return out.reset_index(drop=True)


def preprocess(
    ratings: pd.DataFrame,
    games: pd.DataFrame,
    min_ratings_per_user: int,
    min_ratings_per_game: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean, align, and sparsity-filter ratings and games via iterative k-core.

    Shared game_id mapping: both loaders return raw BGGId as game_id. Here we
    build a single BGGId→dense_int mapping from the games table after alignment,
    apply it to both tables, and drop any ratings whose BGGId is absent from
    games. This keeps the shared mapping logic in exactly one place.

    Steps:
    1. Align: drop ratings for games not in the games table.
    2. Iterative k-core: alternate user-prune / item-prune until stable.
    3. Filter games to surviving game_ids.
    4. Remap raw BGGId → dense int32 game_id consistently across both tables.

    No normalization or binarization (deferred to M4).
    """
    # Step 1 — align: keep only ratings whose game_id (raw BGGId) is in games
    known_games = set(games["game_id"])
    ratings = ratings[ratings["game_id"].isin(known_games)].copy()

    # Step 2 — iterative k-core (vectorized; loops only a handful of times)
    while True:
        prev_len = len(ratings)

        # Prune users below threshold
        user_counts = ratings["user_id"].value_counts()
        keep_users = user_counts[user_counts >= min_ratings_per_user].index
        ratings = ratings[ratings["user_id"].isin(keep_users)]

        # Prune games below threshold
        game_counts = ratings["game_id"].value_counts()
        keep_games = game_counts[game_counts >= min_ratings_per_game].index
        ratings = ratings[ratings["game_id"].isin(keep_games)]

        if len(ratings) == prev_len:
            # Stable — no rows removed this full pass
            break

    ratings = ratings.copy()

    # Step 3 — filter games to surviving game_ids
    games = games[games["game_id"].isin(set(ratings["game_id"]))].copy()

    # Step 4 — build shared BGGId→dense_int mapping from surviving games,
    # then apply to both tables so game_id is a contiguous dense int32.
    sorted_bgids = sorted(games["game_id"].unique())
    bgid_to_dense = {bgid: idx for idx, bgid in enumerate(sorted_bgids)}

    games["game_id"] = games["game_id"].map(bgid_to_dense).astype("int32")
    ratings["game_id"] = ratings["game_id"].map(bgid_to_dense).astype("int32")

    return ratings.reset_index(drop=True), games.reset_index(drop=True)
