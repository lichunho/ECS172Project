# ECS172Project

Context-aware and fairness-aware group recommender system for board games.
See [`project.md`](project.md) for the full proposal.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note: `lightfm` may require build tools (Xcode CLI on macOS, build-essential on Linux). If
installation fails, install a system C/C++ toolchain first and retry.

## Pipeline

All scripts read a YAML config and write outputs under `results/<run_name>/`.

```bash
python scripts/prepare_data.py     --config configs/data.yaml
python scripts/annotate_context.py --config configs/data.yaml
python scripts/simulate_groups.py  --config configs/eval.yaml
python scripts/train.py            --config configs/model.yaml
python scripts/recommend.py        --config configs/group.yaml
python scripts/evaluate.py         --config configs/eval.yaml
```

## Layout

- `src/` — reusable modules (data, preference model, constraints, aggregation, context, baselines, metrics).
- `scripts/` — thin entrypoints; arg parsing and orchestration only.
- `configs/` — YAML configs versioned in git.
- `data/`, `models/`, `results/` — gitignored artifacts.
