**Project Summary**

This repository contains an experiment pipeline for group-aware recommendation using a LightFM-based preference model, group aggregation strategies, and context/constraint handling. I completed missing library modules, fixed runtime issues, and added scripts and configs so you can run end-to-end experiments on a data subset for fast iteration.

**What I Implemented**

- **`src/preference.py`**: Implemented a `PreferenceModel` wrapper around LightFM (fit, predict, save, load). Handles interaction matrix creation, item features, and model persistence.
- **`src/data.py`**: Added subset loaders and named profiles (`current`, `x5`) to run experiments quickly on smaller data slices.
- **`src/aggregation.py`**: Group aggregation strategies (average, least misery, fairness penalty).
- **`src/context.py`**: Context-aware score adjustments (party/competitive/familiarity rewrites).
- **`src/groups.py`**: Group sampling utilities for simulation and evaluation.
- **`src/constraints.py`**: Feasibility filtering (player count, playtime, etc.).
- **`src/metrics.py`**: Metrics implementations: `ndcg_at_k`, `precision_at_k`, `rmse`, `satisfaction_variance`, `min_satisfaction`.
- **`src/baselines.py`**: Simple baselines (average, least misery, random).
- **Scripts**: Completed orchestration scripts in `scripts/`: `train.py`, `recommend.py`, `simulate_groups.py`, `evaluate.py`, `prepare_data.py`, `annotate_context.py`.

**Key Fixes & Notes**

- Resolved LightFM issues (writable sparse matrices and predict input shapes).
- Ensured recommendation/evaluation only score items and users that exist in the trained model mapping to avoid KeyErrors.
- Fixed tokenization and handling of array-valued item features in parquet files.
- Added robust subset profiles to speed local development and presentations.

**Metrics — `satisfaction_variance`**

- **Definition**: `satisfaction_variance` is the variance of per-user satisfaction scores for a recommended set (i.e., how spread out users' satisfaction is within the same recommendation). Lower values indicate more consistent satisfaction across group members; higher values indicate uneven satisfaction.
- **Computation**: For each group and recommended list, compute each member's satisfaction (e.g., predicted or actual rating for items aggregated across the list) and then take the statistical variance across users. Implemented in `src/metrics.py` as `np.var(user_scores)`.

**How to Run (from repo root)**
Use the virtual environment in `.venv` (activate if not already):

```bash
source .venv/bin/activate
```

Train a model on the configured subset (see `configs/model.yaml`):

```bash
python scripts/train.py --config configs/model.yaml
```

Simulate groups and generate recommendations:

```bash
python scripts/simulate_groups.py --config configs/group.yaml
python scripts/recommend.py --config configs/group.yaml
```

Evaluate recommendations (produces CSVs and PNG plots under `results/<run_name>`):

```bash
python scripts/evaluate.py --config configs/eval.yaml
```

**Switching data subset size**

- Edit `subset.profile` in `configs/*.yaml` and set to `current` or `x5` to pick a smaller or larger subset for faster or more realistic runs.

**Where outputs go**

- Models: `models/<run_name>/model.pkl` and `train_summary.json`.
- Results: `results/<run_name>/recommendations.csv`, `groups.json`, `per_group_metrics.csv`, `metrics_summary.csv`, and plots (PNG).

**Files I Modified / Implemented**

- `src/preference.py`, `src/data.py`, `src/aggregation.py`, `src/context.py`, `src/groups.py`, `src/constraints.py`, `src/metrics.py`, `src/baselines.py`.
- `scripts/train.py`, `scripts/recommend.py`, `scripts/simulate_groups.py`, `scripts/evaluate.py`, plus small updates to configs in `configs/`.

**Next Steps / Suggestions**

- Add a `full` subset profile for a single-command full-data run (optional).
- Improve `PreferenceModel.predict` error messages for unknown IDs.
- Add unit tests for new metric functions and end-to-end smoke tests.

See this file for a concise overview and the `src/` and `scripts/` directories for implementation details.
