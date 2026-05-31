# Plan — M2: Context Labels (annotation)

Detailed design for Milestone **M2** of the board-game group recommender. The
top-level milestone map lives in [`plan.md`](plan.md); this file is the resolved
design for the annotation stage only.

**Goal:** tag every game with context labels that support **group** recommendation
(fairness, taste-divergence, accessibility for the weakest member) — not just
single-user quality.

**Stage:** `scripts/annotate_context.py` → `src/annotate.py` →
`data/processed/games_annotated.parquet`.

## Guiding insight

The labels that move *group* outcomes are largely things BGG already structures, or
that derive mechanically from category/mechanic tags — so they need no LLM. Spend LLM
budget only where no structured signal exists. Hence three tiers by labeling method.

## Scale (resolved)

Every label is an **integer 1–5**. This is set by `tests/test_llm.py`'s schema and
matches BGG weight (1–5) and the `language_dependence` poll (5-level). Categoricals
stay strings; binaries map to 1/5. Normalize to [0,1] downstream in M5b if needed —
that's a re-scoring decision, not a labeling one. Declare label names + scale in a
`configs/data.yaml` block so the scheme isn't arbitrary (the open question this closes).

## Tier 1 — parse directly from BGG (no LLM)

Highest leverage, lowest cost. Structured fields/polls from the BGG XML API2.

| Label | Group meaning | Source |
|---|---|---|
| `best_player_count_fit` | does the game stay good at *this group's* size — core fairness lever | `suggested_numplayers` poll (Best/Rec/NotRec tallies) |
| `language_dependence` | excludes non-native / young members | `language_dependence` poll (5-level) |
| `min_age_fit` | maturity gate for mixed-age groups | `minage` + `suggested_playerage` poll |
| `complexity` | teach burden / accessibility | `statistics → averageweight` (already 1–5) |

**Division of labor (decided during M2 implementation):** the raw poll→scalar
distillation lives in **M1/`data.py`**, not here. `data.py` must emit pre-distilled
1–5 scalars `suggested_numplayers` and `language_dependence_poll` (plus raw `minage`/
`suggested_playerage` and `weight`); `annotate.py` only clamps/remaps them into the
named label columns, leaving `pd.NA` when a field is absent. So `best_player_count_fit`
(the core fairness lever) is **blocked on M1** and is currently untested — add a unit
test once M1 supplies a real scalar.

## Tier 2 — rule-derived from category/mechanic tags (no LLM)

A documented lookup table, written once, deterministic thereafter.

| Label | Group meaning | Rule |
|---|---|---|
| `player_elimination` (binary→1/5) | eliminated members sit idle → tanks min-satisfaction | "Player Elimination" mechanic present |
| `coop_vs_competitive` (categorical) | coop collapses taste divergence; competitive amplifies it | Cooperative / Semi-Cooperative / Team-Based links |
| `social_conflict` | negotiation / take-that → friction, kingmaking | "Negotiation" / "Take That" / "Trading" links |

## Tier 3 — LLM judgment (`src/llm.py` Ollama client, already up & tested)

Genuine judgment calls with no structured equivalent. One `llm.chat(..., fmt=schema)`
per game, schema-constrained to a JSON object of integer 1–5 labels.

| Label | Group meaning |
|---|---|
| `party_friendliness` *(existing)* | suitability for loud/casual social play |
| `interaction_level` *(existing)* | solo-ish multiplayer vs highly interactive |
| `competitiveness` *(existing)* | cutthroat vs gentle |
| `downtime` / `kingmaking` *(new)* | idle/disengaged members between turns → fairness hazard |
| `teach_time` *(new)* | onboarding cost borne by the weakest member |
| `mixed_skill_robustness` *(new)* | does an expert crush novices, or does the game self-balance? |

Prompt feeds title, description, categories, mechanics, weight; the JSON schema in
`tests/test_llm.py` extends to cover all six labels.

## Wiring decisions (confirmed against the committed LLM code)

- **`annotate_games` signature must grow.** `annotate_games(games)` cannot reach the
  server. Change to `annotate_games(games, *, host, port, model, options, timeout)`;
  `scripts/annotate_context.py` loads `.env` + `cfg["llm"]` (as `test_llm.py` does) and
  passes them in — keeps the entry point thin.
- **Caching belongs in `annotate.py`**, not `llm.py` (which is deliberately cache-free):
  skip games already present in the annotated parquet on re-run.
- **Archive before overwrite.** Move the old annotation set to
  `archive/YYYY-MM-DD_annotations/` before re-annotating (CLAUDE.md rule).

## Test plan

- Assert every label column exists and is an integer in 1–5.
- Spot-check known games: Codenames → high `party_friendliness`, low `complexity`;
  Twilight Imperium → high `teach_time` + `competitiveness`; Pandemic → coop, low divergence.
- Tier 1: verify a couple of poll parses against the live BGG page.

## Part B model assignments

- **Implement — Sonnet** (Tier 1 parse + Tier 2 rules + Tier 3 prompt/JSON wiring)
- **Verify — Sonnet** (label ranges, scale consistency, mapping correctness)
- **Run — Haiku** (`python scripts/annotate_context.py --config configs/data.yaml`)
- **Evaluate — Opus** (label distributions; do divergent games get divergent labels? —
  this is what makes the fairness work visible in M3/M6)

## Status

Design resolved, pending approval. LLM transport (`src/llm.py`) committed and tested.
Launch implementation with **"execute"**.
