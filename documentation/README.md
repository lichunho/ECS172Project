# Documentation — ECS172Project

Reference documentation for the **context-aware and fairness-aware group recommender
for board games** (ECS 172 course project).

This folder is the developer-facing reference. For the research proposal see
[`../project.md`](../project.md); for agent/contributor working rules see
[`../CLAUDE.md`](../CLAUDE.md); for the milestone planning docs see [`../plans/`](../plans/).

## Contents

| Doc | What it covers |
|---|---|
| [setup.md](setup.md) | Environment, dependencies, dataset download, LLM server, smoke tests |
| [architecture.md](architecture.md) | The four conceptual components, modules, and data flow |
| [pipeline.md](pipeline.md) | The six pipeline stages, run commands, and implementation status |
| [data-schema.md](data-schema.md) | Processed table schemas and the context-label scheme |
| [configuration.md](configuration.md) | Every config key in `configs/*.yaml` and the `.env` contract |
| [modules.md](modules.md) | Public API of each `src/` module |
| [roadmap.md](roadmap.md) | Milestone map (M1–M6): what is built vs. scaffold, open design questions |

## Status at a glance

- **M1 — Data foundation** (`prepare_data`): implemented.
- **M2 — Context labels** (`annotate_context`): implemented and run.
- **M3–M6** (simulate groups, train, recommend, evaluate): **scaffold only** — module
  bodies raise `NotImplementedError`.

Implementation status changes as milestones land. Treat the status notes here as a
snapshot; confirm against `git log` and the module bodies before relying on a stage.
See [roadmap.md](roadmap.md) for details.
