# Plan — Architecture Layout & Milestones

Context-aware and fairness-aware **group recommender for board games** (ECS 172).
This document maps the architecture into components and the problems/milestones to
solve. It describes *what* must be built and the open design questions to resolve —
not line-by-line implementation. Current state: scaffold only (every `src/` function
and script body raises `NotImplementedError`; configs and signatures exist).

## The spine (data flow)

```
raw BGG data
  → [M1] prepare_data     → data/processed/ (ratings, games)
  → [M2] annotate_context → games_annotated (context labels)
  → [M3] simulate_groups  → eval groups (the test harness)
  → [M4] train            → models/<run_name>/ (per-user preference model)
  → [M5] recommend        → ranked list (predict → constrain → context → aggregate)
  → [M6] evaluate         → results/<run_name>/ (accuracy + fairness vs baselines)
```

Six pipeline stages, each a thin script over `src/` modules. The contracts
(signatures, configs) exist; the bodies do not.

## Components (responsibility of each)

**Foundation layer**
- **Data ingest** (`data.py`) — load BGG ratings + game metadata, sparsity-filter,
  emit clean tables. Trap: BGG uses `0` for missing player-count/playtime, not real zeros.
- **Context annotation** (`annotate.py`) — tag each game with `party_friendliness`,
  `interaction_level`, `competitiveness`. LLM- or rule-based.
- **Group simulation** (`groups.py`) — synthesize evaluation groups (no real groups
  exist). This is the test harness; without divergent-taste groups, fairness work is invisible.

**Modeling layer**
- **Individual preference** (`preference.py`) — LightFM hybrid CF+content; produces
  per-user × per-item scores. The engine the rest sits on.

**Decision layer (the recommend pipeline, in order)**
- **Constraint filter** (`constraints.py`) — hard-remove infeasible games (player
  count, playtime) *before* ranking.
- **Context adjust** (`context.py`) — re-score by social setting + familiarity
  (post-filtering re-rank).
- **Fairness aggregation** (`aggregation.py`) — collapse the score matrix to one
  group score: `average | least_misery | fairness_penalty`.

**Evaluation layer**
- **Baselines** (`baselines.py`) — average / least-misery / random comparators.
- **Metrics** (`metrics.py`) — NDCG@K, Precision@K, RMSE + fairness (satisfaction
  variance, min satisfaction).

## Milestones / problems to solve (dependency order)

| # | Milestone | Problem to solve | Open question to resolve first | Part B model |
|---|-----------|------------------|--------------------------------|--------------|
| **M1** | Data foundation | Clean ratings matrix + game metadata from raw BGG. | Which BGG source? Keep explicit 1–10 ratings or binarize? Sparsity filter threshold (k-core N)? | Sonnet |
| **M2** | Context labels | Attach context dimensions to every game. | RESOLVED — see "M2 resolved design" below. Three labeling tiers; **1–5 integer** scale for all labels; LLM client (`src/llm.py`, Ollama) already in place. | Sonnet |
| **M3** | Group simulation harness | Manufacture evaluation groups since none exist. | Group sizes? How many groups? Must include *divergent* groups (else aggregation methods look identical)? | Sonnet |
| **M4** | Preference engine | Train LightFM to produce per-user scores. | WARP (ranking/implicit) vs logistic (explicit)? Choice decides which metrics are coherent (RMSE only if predicting on 1–10). Identity-feature matrix shape is a known footgun. | Sonnet |
| **M5a** | Constraint filter | Drop infeasible games pre-ranking. | None major — most mechanical milestone. | Sonnet |
| **M5b** | Context adjustment | Turn context labels into a score re-weighting. | Underspecified in docs: multiplicative boost? per-setting calibration? Need a concrete rule. | Sonnet |
| **M5c** | Fairness aggregation | Combine per-user scores fairly. | Underspecified in docs: the "disagreement penalty" has no formula. Disagreement metric (variance? pairwise spread?) and how `fairness_weight` trades off? | Sonnet |
| **M6** | Evaluation + baselines | Measure accuracy *and* fairness vs baselines; show the trade-off. | Optimize variance and min-satisfaction jointly, or report as alternatives? Match metrics to M4 modeling choice. | Opus (analysis) |

## M2 resolved design (context labels)

Resolved in detail in its own file: **[`plan_m2.md`](plan_m2.md)**. Summary: three
labeling tiers (Tier 1 parse BGG polls/stats, Tier 2 rule-map category/mechanic tags,
Tier 3 LLM judgment via the committed `src/llm.py` Ollama client), all on an **integer
1–5** scale, plus new group-specific LLM labels (`downtime`/`kingmaking`, `teach_time`,
`mixed_skill_robustness`).

## Cross-cutting risks to decide up front

- **Accuracy ↔ fairness is a fundamental trade-off, not a bug.** It only becomes
  *visible* with divergent-taste groups (M3) + paired mean/dispersion metrics (M6).
  If M3 only makes random/similar groups, the fairness contribution looks pointless.
- **Three under-specified spots in the proposal** that need a decision before coding:
  the disagreement penalty formula (M5c), the context re-scoring mechanism (M5b), and
  the fairness-metric optimization stance (M6).
- **Modeling choice in M4 propagates** — implicit/WARP vs explicit/logistic decides
  whether RMSE is meaningful at all.
- **Versioning rule (CLAUDE.md):** every milestone that writes under a `run_name`
  must archive before overwriting.

## Status

Nothing implemented. This is the layout and milestone map, pending approval.
Launch implementation with **"execute"**; decide whether to resolve the three
under-specified design questions before or during implementation.
