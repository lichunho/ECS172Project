# Data Schema

All processed artifacts live under `data/processed/` (gitignored). Column names are the
fixed contract between stages — downstream modules read these names directly.

## `ratings.parquet`

Produced by `prepare_data`; consumed by `preference`, `groups`, `train`.

| Column | Type | Notes |
|---|---|---|
| `user_id` | int32 | factorized from `Username` |
| `game_id` | int32 | dense id; **shared mapping with `games.parquet`** |
| `rating` | float32 | raw BGG 1–10; never binarized in M1 (binarization is an M4 modeling choice) |

## `games.parquet`

Produced by `prepare_data`; consumed by `annotate`, `constraints`, `context`.

| Column | Type | Source / notes |
|---|---|---|
| `game_id` | int32 | `BGGId` remapped to shared dense id |
| `name` | str | `Name` |
| `description` | str | `Description` (punctuation-stripped + lemmatized in this dump — degraded for LLM prompts) |
| `categories` | list[str] | reshaped from the 8 one-hot `Cat:*` columns |
| `mechanics` | list[str] | reshaped from one-hot columns in `mechanics.csv`, joined on `BGGId` |
| `weight` | Float64 | `GameWeight` (~1–5); `0 → NaN` |
| `min_players` | Float64 | `MinPlayers`; `0 → NaN` |
| `max_players` | Float64 | `MaxPlayers`; `0 → NaN` |
| `min_playtime` | Float64 | `ComMinPlaytime`; `0 → NaN` |
| `max_playtime` | Float64 | `ComMaxPlaytime`; `0 → NaN` |
| `minage` | Float64 | `MfgAgeRec`; `0 → NaN` |
| `suggested_playerage` | Float64 | `ComAgeRec` (community age poll) |
| `best_player_count` | Float64 | `BestPlayers`; `0 → NaN`. Used at recommend-time (M5) |
| `good_player_counts` | list[int] | parsed from `GoodPlayers` (`'4+'` → leading int `4`) |

> The constraint filter expects `min_players`, `max_players`, `min_playtime`,
> `max_playtime`. `best_player_count` / `good_player_counts` feed a recommend-time
> player-count fit (M5), **not** a static annotation label.

## `games_annotated.parquet`

Produced by `annotate_context` = all `games.parquet` columns **plus** the label columns
below. Row order/index preserved. Labels are integer **1–5** unless noted.

### Tier 1 — reshaped BGG fields (no LLM)

| Label | Meaning | Source / rule |
|---|---|---|
| `complexity` | teach burden / accessibility | `weight` rounded to int 1–5; `NA` if weight absent |
| `min_age_fit` | maturity gate for mixed-age groups | remap age → 1–5 (`≤6→5, 7–9→4, 10–12→3, 13–15→2, 16+→1`); uses `suggested_playerage` then `minage`; `NA` if both absent |
| `language_dependence` | text-reliance barrier | **all-NA by design** — no reliable source column in the Kaggle dump; deferred to "M1b" (BGG-API poll fetch) |

### Tier 2 — category/mechanic rule lookup (no LLM)

Rules match case-insensitively against the union of a game's `categories` + `mechanics`;
the first (most specific) matching rule per label wins, else a default.

| Label | Values | Rule (token → value), default |
|---|---|---|
| `player_elimination` | 1 (absent) / 5 (present) | `"player elimination"` → 5; default 1 |
| `coop_vs_competitive` | `coop` / `semi-coop` / `team` / `competitive` | `semi-cooperative`→`semi-coop`, `team-based`→`team`, `cooperative`→`coop`; default `competitive` |
| `social_conflict` | 1 (absent) / 5 (present) | `negotiation` / `take that` / `take-that` / `trading` → 5; default 1 |

### Tier 3 — LLM judgment (one Ollama call per in-scope game, int 1–5)

| Label | Meaning (1 → 5) |
|---|---|
| `party_friendliness` | terrible → ideal for casual/party play |
| `interaction_level` | almost zero interaction → highly interactive |
| `competitiveness` | gentle/cooperative → cutthroat |
| `downtime` | minimal waiting → very long downtime between turns |
| `teach_time` | rules in < 2 min → hours of learning |
| `mixed_skill_robustness` | experts crush novices → game self-balances mixed skill |

Tier-3 labels are present only for games within
`annotation.tier3_min_ratings` scope; out-of-scope games carry `pd.NA`. The LLM prompt
feeds title, (truncated) description, categories, mechanics, and weight; decoding is
constrained to a JSON object of the six integer labels.

The full input/output column contract is documented in the `annotate_games` docstring in
[`../src/annotate.py`](../src/annotate.py).
