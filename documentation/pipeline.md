# Pipeline

Six stages, each a thin script that takes `--config <yaml>`, loads it with
`yaml.safe_load`, and writes artifacts namespaced by `run_name`. Run in order.

```bash
python scripts/prepare_data.py     --config configs/data.yaml
python scripts/annotate_context.py --config configs/data.yaml
python scripts/simulate_groups.py  --config configs/eval.yaml
python scripts/train.py            --config configs/model.yaml
python scripts/recommend.py        --config configs/group.yaml
python scripts/evaluate.py         --config configs/eval.yaml
```

## Stage reference

### 1. `prepare_data.py` → `src/data.py` — implemented

Loads raw BGG CSVs, cleans, sparsity-filters, writes processed parquet.

- **Reads:** `data/raw/{user_ratings.csv, games.csv, mechanics.csv}` (`raw_dir`).
- **Writes:** `data/processed/{ratings.parquet, games.parquet}` (`processed_dir`).
- **Key logic:** drop ratings `< 1` (BGG 0 = unrated); dedup `(user, game)`; join
  `mechanics.csv` on `BGGId`; reshape one-hot `Cat:*` / mechanic columns to list
  columns; convert BGG `0 → NaN` for players/playtime/age/weight; **iterative k-core**
  prune (alternate user/item until stable) at `min_ratings_per_user` /
  `min_ratings_per_game`; build one shared `BGGId → dense int32 game_id` mapping applied
  to both tables.
- **Archive:** moves an existing non-empty `processed_dir` to
  `archive/<date>_processed/` before writing.

### 2. `annotate_context.py` → `src/annotate.py` — implemented and run

Adds context-label columns to the games table via three tiers.

- **Reads:** `<processed_dir>/games.parquet` (+ `ratings.parquet` for the Tier-3 scope
  filter); LLM host/port from `.env`; model/options from `configs/data.yaml`.
- **Writes:** `<processed_dir>/games_annotated.parquet`.
- **Tiers:** Tier 1 = reshape BGG fields (no LLM); Tier 2 = category/mechanic rule
  lookup (no LLM); Tier 3 = one Ollama chat call per game, JSON-schema-constrained.
- **Tier-3 scope:** only games with `≥ annotation.tier3_min_ratings` ratings get the
  LLM pass; the rest get NA Tier-3 labels (backfill by lowering the threshold and
  re-running). Resumable via `games_annotated.tier3_checkpoint.parquet`.
- **Archive:** moves an existing `games_annotated.parquet` to
  `archive/<date>_annotations/` before writing. The checkpoint file is **not** archived
  (it enables resume).

See [data-schema.md](data-schema.md) for the label scheme.

### 3. `simulate_groups.py` → `src/groups.py` — scaffold

Sample evaluation groups (`groups.sample_groups`) from processed ratings; intended to
persist to `results/<run_name>/groups.json`. Body raises `NotImplementedError`.

### 4. `train.py` → `src/preference.py` — scaffold

Fit `PreferenceModel` (LightFM) on ratings + item features; save to
`models/<run_name>/model.pkl`. `fit/predict/save/load` raise `NotImplementedError`.

### 5. `recommend.py` — scaffold

Top-K group recommendation for one group config: load games + model → constraint filter
→ predict → aggregate → context adjust → write `results/<run_name>/recommendations.json`.
Orchestration is outlined in comments; body raises `NotImplementedError`.

### 6. `evaluate.py` — scaffold

Run full evaluation (ours vs. baselines) over simulated groups; compute NDCG@K, P@K,
RMSE, satisfaction variance, min satisfaction; write a metrics table + plots under
`results/<run_name>/`. Body raises `NotImplementedError`.

## Artifact namespacing

Outputs are keyed by the `run_name` in each config. Before regenerating over an existing
`run_name`, archive the old generation (see
[architecture.md](architecture.md#design-conventions-from-claudemd)) or bump `run_name`.
`data/`, `models/`, and `results/` are gitignored.
