# Module Reference

Public surface of each `src/` module. "Implemented" = real logic; "scaffold" = signature
exists, body raises `NotImplementedError`.

## `src/data.py` — implemented

BGG load + preprocess.

- `load_ratings(raw_dir) -> DataFrame[user_id, game_id, rating]` — read
  `user_ratings.csv`; drop `rating < 1`; dedup `(Username, BGGId)` keeping last;
  factorize `Username → user_id`; keep raw `BGGId` as `game_id` (remapped later).
- `load_games(raw_dir) -> DataFrame` — read `games.csv`, left-join `mechanics.csv` on
  `BGGId`; reshape one-hot `Cat:*` and mechanic columns into `categories` / `mechanics`
  lists; `0 → NaN` for numeric metadata; parse `GoodPlayers → good_player_counts`.
  Returns the canonical games schema (see [data-schema.md](data-schema.md)).
- `preprocess(ratings, games, min_ratings_per_user, min_ratings_per_game) -> (ratings, games)`
  — align (drop ratings for unknown games), **iterative k-core** prune until stable,
  build one shared `BGGId → dense int32` mapping applied to both tables. No
  normalization/binarization (deferred to M4).

## `src/annotate.py` — implemented

Three-tier context labeling.

- `annotate_games(games, *, host, port, model, options, timeout, concurrency=1,
  checkpoint_path=None, checkpoint_every=100, tier3_ids=None) -> DataFrame` — appends
  all label columns. Tier 1 + Tier 2 are pure functions of the row; Tier 3 calls the LLM
  (threaded, resumable via checkpoint). If `tier3_ids` is given, only those games get the
  LLM pass; others get NA Tier-3 labels. The docstring carries the full input/output
  column contract.

Internal helpers (`_apply_tier1/2/3`, `_tier2_row`, `_build_prompt`,
`_parse_tier3_response`, checkpoint load/save) are private — call `annotate_games`.

## `src/llm.py` — implemented

Thin Ollama HTTP client (no retries, no streaming, no caching by design).

- `health_check(host, port, timeout=10.0) -> list[str]` — `GET /api/tags`; returns
  available model tags. Raises `OllamaError` on failure.
- `chat(host, port, model, messages, options=None, fmt=None, timeout=120.0) -> str` —
  `POST /api/chat` (non-streaming); returns the assistant message content. `fmt` enables
  structured output (`"json"` or a JSON-schema dict). Raises `OllamaError` on failure.
- `OllamaError(RuntimeError)` — unreachable server or non-200 response.

## `src/preference.py` — scaffold

- `class PreferenceModel(loss="warp", no_components=64, learning_rate=0.05, epochs=30)`
  — LightFM wrapper. `fit(ratings, item_features=None)`, `predict(user_ids, item_ids) ->
  (len(users), len(items)) array`, `save(path)`, `load(path)`. All bodies raise.

## `src/constraints.py` — scaffold

- `filter_feasible(games, constraints) -> DataFrame` — drop games violating
  `n_players` / `max_playtime_min`. Reads `min_players, max_players, min_playtime,
  max_playtime`.

## `src/aggregation.py` — scaffold

- `aggregate(score_matrix, method="fairness_penalty", fairness_weight=0.5) -> 1-D array`
  — collapse `(n_users_in_group, n_items)` to one score per item.
  `method ∈ {average, least_misery, fairness_penalty}` (the last = mean + weight·min −
  disagreement penalty; the penalty formula is an open design question, see
  [roadmap.md](roadmap.md)).

## `src/context.py` — scaffold

- `adjust(scores, games, context) -> 1-D array` — re-weight item scores by
  `context = {setting, familiarity}` using annotated features
  (`party_friendliness`, `interaction_level`, `competitiveness`).

## `src/groups.py` — scaffold

- `@dataclass Group(user_ids, context, constraints)`
- `sample_groups(user_ids, n_groups, sizes, seed=None) -> list[Group]` — manufacture
  evaluation groups (no real groups exist in the data).

## `src/baselines.py` — scaffold

- `average_baseline(score_matrix)`, `least_misery_baseline(score_matrix)`,
  `random_baseline(n_items, seed=None)` — comparators returning a per-item score vector.

## `src/metrics.py` — scaffold

- Accuracy: `ndcg_at_k(ranked_items, relevance, k)`,
  `precision_at_k(ranked_items, relevant_set, k)`, `rmse(y_true, y_pred)`.
- Fairness: `satisfaction_variance(per_user_scores)`, `min_satisfaction(per_user_scores)`.
