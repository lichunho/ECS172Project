# Roadmap & Milestone Status

The project is built milestone-by-milestone. Each milestone is one pipeline stage; a
per-milestone planning doc in [`../plans/`](../plans/) resolves the design before code is
written. This page summarizes status and the open design questions. The authoritative map
is [`../plans/plan.md`](../plans/plan.md).

> Status is a snapshot. Confirm against `git log` and the module bodies before relying on
> a stage.

## The spine

```
[M1] prepare_data     → data/processed/ (ratings, games)
[M2] annotate_context → games_annotated (context labels)
[M3] simulate_groups  → eval groups (test harness)
[M4] train            → models/<run_name>/ (preference model)
[M5] recommend        → ranked list (predict → constrain → context → aggregate)
[M6] evaluate         → results/<run_name>/ (accuracy + fairness vs. baselines)
```

## Status

| # | Milestone | Status | Plan doc |
|---|---|---|---|
| M1 | Data foundation | **Implemented** | [m1_plan.md](../plans/m1_plan.md) |
| M2 | Context labels | **Implemented and run** | [plan_m2.md](../plans/plan_m2.md) |
| M3 | Group simulation harness | Scaffold | — |
| M4 | Preference engine (LightFM) | Scaffold | — |
| M5a | Constraint filter | Scaffold | — |
| M5b | Context adjustment | Scaffold | — |
| M5c | Fairness aggregation | Scaffold | — |
| M6 | Evaluation + baselines | Scaffold | — |

### M1 — Data foundation (done)

Kaggle BGG dump → `ratings.parquet` + `games.parquet`. Keeps raw 1–10 ratings (no
binarization), iterative k-core at 5/10, shared `game_id` mapping. Two dump-specific
decisions: `language_dependence` has no reliable source column (deferred to a future
"M1b" BGG-API fetch); player-count fit moved out of static annotation into a
recommend-time computation (M5).

### M2 — Context labels (done)

Three-tier labeling: Tier 1 reshapes BGG fields, Tier 2 applies category/mechanic rules,
Tier 3 calls Ollama (JSON-schema-constrained, int 1–5). Tier-3 runs only on games above a
ratings threshold; resumable via checkpoint. `language_dependence` stays all-NA by design.

## Open design questions (decide before implementing the scaffold stages)

These are flagged in `plan.md` as underspecified in the proposal:

- **M3 — group divergence.** Sizes? How many groups? The harness **must** include
  divergent-taste groups, otherwise the aggregation methods look identical and the
  fairness work is invisible.
- **M4 — modeling choice.** WARP/implicit vs. logistic/explicit decides whether RMSE is
  even coherent (RMSE only makes sense if predicting on the 1–10 scale). LightFM identity-
  feature matrix shape is a known footgun.
- **M5b — context re-scoring.** No concrete rule yet: multiplicative boost? per-setting
  calibration?
- **M5c — disagreement penalty.** The `fairness_penalty` formula has no defined
  disagreement metric (variance? pairwise spread?) and no defined `fairness_weight`
  trade-off.
- **M6 — fairness stance.** Optimize satisfaction variance and min-satisfaction jointly,
  or report them as alternatives? Metrics must match the M4 modeling choice.

## Cross-cutting risk

Accuracy ↔ fairness is a fundamental trade-off, not a bug. It only becomes **visible**
with divergent-taste groups (M3) plus paired mean/dispersion metrics (M6). If M3 only
makes random/similar groups, the fairness contribution looks pointless.
