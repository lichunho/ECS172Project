# Configuration

`configs/` holds one YAML per stage (versioned in git). Configs are the source of truth
for paths and `run_name`. Secrets (the LLM host) live in `.env`, not in YAML.

## `configs/data.yaml` — used by `prepare_data` + `annotate_context`

| Key | Example | Meaning |
|---|---|---|
| `raw_dir` | `data/raw` | input CSV directory |
| `processed_dir` | `data/processed` | output parquet directory |
| `bgg_source` | `kaggle` | provenance marker (placeholder: `kaggle` \| `bgg_api` \| `local`) |
| `min_ratings_per_user` | `5` | k-core: min ratings to keep a user |
| `min_ratings_per_game` | `10` | k-core: min ratings to keep a game |
| `llm.model` | `gemma4:latest` | Ollama model tag for Tier-3 |
| `llm.timeout_s` | `120` | per-request timeout |
| `llm.options` | `{temperature: 0.0}` | Ollama sampling params |
| `llm.concurrency` | `1` | Tier-3 worker threads (see note) |
| `llm.checkpoint_every` | `100` | flush Tier-3 checkpoint every N games |
| `annotation.tier3_min_ratings` | `500` | only games with ≥ this many ratings get the LLM pass; `null`/omit = all games |
| `annotation.labels` | list | declared label schema (name, tier, scale) — documentation of the scheme |

> **Concurrency note (from the config comment):** the Ollama host serializes on one GPU,
> so `concurrency > 1` only causes queue timeouts without speeding anything up. Raise it
> only if `OLLAMA_NUM_PARALLEL` is increased server-side.

## `configs/model.yaml` — used by `train`

| Key | Example | Meaning |
|---|---|---|
| `loss` | `warp` | LightFM loss: `warp` \| `bpr` \| `logistic` \| `warp-kos` |
| `no_components` | `64` | latent dimensionality |
| `learning_rate` | `0.05` | SGD learning rate |
| `epochs` | `30` | training epochs |
| `item_features` | `[genre, mechanics, complexity]` | content features fed to the hybrid model |
| `run_name` | `lightfm_baseline` | namespaces `models/<run_name>/` |

## `configs/group.yaml` — used by `recommend`

| Key | Example | Meaning |
|---|---|---|
| `user_ids` | `[]` | BGG user ids forming the group |
| `context.setting` | `party` | social setting: `casual` \| `party` \| `competitive` |
| `context.familiarity` | `friends` | `friends` \| `strangers` |
| `constraints.n_players` | `4` | group size (feasibility filter) |
| `constraints.max_playtime_min` | `60` | playtime ceiling |
| `aggregation.method` | `fairness_penalty` | `average` \| `least_misery` \| `fairness_penalty` |
| `aggregation.fairness_weight` | `0.5` | blends min satisfaction with the mean |
| `top_k` | `10` | size of the recommended list |
| `run_name` | `group_example` | namespaces `results/<run_name>/` |

## `configs/eval.yaml` — used by `simulate_groups` + `evaluate`

| Key | Example | Meaning |
|---|---|---|
| `k_values` | `[5, 10]` | K for NDCG@K / Precision@K |
| `n_groups` | `200` | number of simulated groups |
| `group_sizes` | `[3, 4, 5]` | sizes sampled when simulating groups |
| `baselines` | `[average, least_misery, random]` | comparators to evaluate against |
| `seed` | `42` | RNG seed for reproducible sampling |
| `run_name` | `eval_baseline` | namespaces `results/<run_name>/` |

## `.env` {#env}

Secrets only; gitignored. Copy from `.env.example`.

| Key | Example | Meaning |
|---|---|---|
| `OLLAMA_HOST` | `board-llm.tailXXXX.ts.net` | Ollama host: Tailscale MagicDNS name or `100.x` IP |
| `OLLAMA_PORT` | `11434` | Ollama port |

`annotate_context.py`, `test_llm.py`, and `test_annotate.py` read these via
`python-dotenv`; the host is **required** (they exit if `OLLAMA_HOST` is unset).
