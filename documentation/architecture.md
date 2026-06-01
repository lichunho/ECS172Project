# Architecture

The system is a group recommender that jointly optimizes preference accuracy, fairness
across group members, real-world feasibility (player count, playtime), and social
context (casual / party / competitive). See [`../project.md`](../project.md) for the
research framing.

## Four conceptual components

| Component | Module | Responsibility |
|---|---|---|
| Individual preference modeling | `src/preference.py` | LightFM hybrid CF + content model → per-user × per-item scores |
| Constraint-aware filtering | `src/constraints.py` | Hard-remove infeasible games (player count, playtime) **before** ranking |
| Fairness-aware aggregation | `src/aggregation.py` | Collapse a per-user × per-item score matrix into one group score per item |
| Context-aware adjustment | `src/context.py` | Re-score by social setting and group familiarity |

## Supporting modules

| Module | Responsibility |
|---|---|
| `src/data.py` | Load + preprocess raw BGG data (k-core sparsity filter, shared `game_id` mapping) |
| `src/annotate.py` | Three-tier context labeling of games (rules + LLM) |
| `src/llm.py` | Thin Ollama HTTP client (health check + non-streaming chat) |
| `src/groups.py` | Sample simulated evaluation groups from the rating matrix |
| `src/baselines.py` | Comparators: average / least-misery / random aggregation |
| `src/metrics.py` | Accuracy (NDCG@K, Precision@K, RMSE) + fairness (satisfaction variance, min satisfaction) |

Per-module API detail: [modules.md](modules.md).

## Data flow

```
raw BGG CSVs (data/raw/)
  → prepare_data     → data/processed/{ratings,games}.parquet
  → annotate_context → data/processed/games_annotated.parquet
  → simulate_groups  → eval groups (test harness)
  → train            → models/<run_name>/
  → recommend        → ranked list:  predict → constrain → aggregate → context-adjust
  → evaluate         → results/<run_name>/ (accuracy + fairness vs. baselines)
```

The recommend stage is a fixed pipeline over the four components, in order:

1. `preference.PreferenceModel.predict` → per-user score matrix.
2. `constraints.filter_feasible` → drop infeasible games.
3. `aggregation.aggregate` → one group score per item.
4. `context.adjust` → re-weight by social setting / familiarity, then take top-K.

## Design conventions (from CLAUDE.md)

- **Thin entry points.** `scripts/*.py` do arg parsing + YAML load + orchestration
  only; real logic lives in `src/`. Each script prepends the repo root to `sys.path`
  so `from src... import` works when run from the project root.
- **Paths flow from config.** There is no central config *module*; each stage reads its
  paths (`raw_dir`, `processed_dir`, `run_name`, …) from the loaded YAML and passes them
  down. No literal paths buried in `src/`.
- **One job per module**, call the public interface (don't reach into a sibling's
  internals), and add behavior via new flags/modules rather than branching inside a core
  loop.
- **Archive before destructive change.** Before regenerating artifacts under an existing
  `run_name` (or processed data / annotations), the old generation is moved to a dated
  `archive/YYYY-MM-DD_<desc>/` sibling that no config references. `prepare_data.py` and
  `annotate_context.py` already implement this.
