**Experiments**

This document describes the experimental setup used in the project: the datasets, baselines, and evaluation metrics. It is intended to make the evaluation reproducible and to clarify what each reported number means.

**Datasets**

- **Processed interactions:** [data/processed/ratings.parquet](data/processed/ratings.parquet) contains user–game interaction records used to train and evaluate models (user id, game id, rating, timestamp, etc.). For fast iteration we provide configurable subset profiles in [configs/data.yaml](configs/data.yaml) (e.g. `current`, `x5`) which limit the number of ratings, users and games loaded.
- **Item catalog:** [data/processed/games.parquet](data/processed/games.parquet) contains game metadata and features used as item side information by the hybrid model.
- **Annotated context:** [data/processed/games_annotated.parquet](data/processed/games_annotated.parquet) augments the catalog with human / model-provided context labels (party, competitive, familiarity, etc.) used by the context-aware re-weighting component.

Notes:

- Use the `current` subset for quick smoke runs and `x5` (or the full dataset) for larger experiments. The exact sizes are controlled in `configs/data.yaml`.

**Baselines and Models**

- **Average (Group Mean):** for each candidate item, compute the arithmetic mean of members' predicted (or known) scores and rank by this mean. Implemented as `average_baseline` / `aggregate(..., method='average')`.
- **Least-misery:** rank items by the minimum member score (the group's most dissatisfied member). Implemented as `least_misery` / `aggregate(..., method='least_misery')`.
- **Random:** randomly order feasible items to provide a lower-bound baseline (`random_baseline`).
- **LightFM hybrid model (main model):** a matrix-factorization style model that can incorporate item features (catalog metadata). Trained on interactions and used to produce per-user per-item scores; group-level ranking is produced by applying an aggregation strategy to per-member scores.
- **Fairness-penalized aggregation (optional):** an aggregation that penalizes candidate items with high inter-member score variance to promote more equitable group satisfaction (`aggregate(..., method='fairness_penalty')`).

All baselines are implemented in [src/baselines.py](src/baselines.py) and group aggregation strategies are in [src/aggregation.py](src/aggregation.py).

**Evaluation Protocol**

- For each sampled or provided group:
  1. Filter candidate games by feasibility (player count, playtime) and any group constraints (see [src/constraints.py](src/constraints.py)).
  2. Score candidate games per member using the model (or use available ground-truth ratings for baseline simulations).
  3. Apply a group aggregation function to obtain a single group-level score per game.
  4. Produce the top-K recommendations for the group and compute evaluation metrics comparing the top-K to members' true preferences.
- Metrics are computed per-group and then aggregated (mean ± standard error) across the set of sampled groups.

**Evaluation Metrics**
Below are the metrics reported and their interpretation.

- **NDCG@K (Normalized Discounted Cumulative Gain):** captures ranking quality with position-sensitive gains. Relevance values $r_i$ can be the raw ratings or a graded relevance derived from ratings. We use the standard DCG formulation:

$$
\mathrm{DCG}@K = \sum_{i=1}^{K} \frac{2^{r_i}-1}{\log_2(i+1)}
$$

and

$$
\mathrm{NDCG}@K = \frac{\mathrm{DCG}@K}{\mathrm{IDCG}@K}
$$

where IDCG@K is the ideal DCG obtained by sorting by true relevance.

- **Precision@K:** fraction of the top-K recommendations that are relevant (binary hit). For a rating threshold $\tau$ (e.g. 4),

$$
\mathrm{Precision}@K = \frac{1}{K} \sum_{i=1}^{K} \mathbf{1}[r_i \ge \tau]
$$

- **RMSE (Root Mean Squared Error):** measures the accuracy of predicted scores against observed member ratings for recommended items. When predictions are at group-level we compare the predicted group score to the group's average true rating for the item; when predictions are per-user we compute per-user errors aggregated over recommended items.

$$
\mathrm{RMSE} = \sqrt{\frac{1}{N} \sum_{j=1}^{N} (\hat{y}_j - y_j)^2 }
$$

- **Satisfaction (per-member):** for a group and a recommended top-K, each member's satisfaction is the average of their ratings for items in the top-K. From per-member satisfactions we derive two group-level fairness-aware metrics:
  - **Satisfaction variance:** the sample variance of members' satisfactions within a group (lower is more equitable). This is reported as the mean variance across groups.
  - **Minimum satisfaction:** the minimum per-member satisfaction in the group (useful for worst-off analysis).

Reporting:

- We report each metric averaged across all evaluation groups, and include standard error (or confidence intervals) to convey variability across groups.
- When presenting ranking metrics (NDCG@K, Precision@K) we report them for multiple K values (e.g., K = 1, 5, 10) so readers can see short- and medium-length recommendation behavior.

References for metric definitions are standard IR literature (e.g., Järvelin & Kekäläinen for DCG/NDCG) and are applied here to group recommendations by treating group-level relevance as described above.

If you'd like, I can expand the metric definitions with the exact thresholds and rating normalization used in the codebase (for example the binary threshold used for Precision@K), or add a small worked example showing how per-member satisfaction and satisfaction variance are computed for a single group and top-K.
