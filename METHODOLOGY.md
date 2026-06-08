**Methodology**

This document summarizes the experimental methodology used in the project. It is written to be included in a research paper and captures dataset preparation, model design, group simulation, aggregation strategies, evaluation metrics, and reproducibility details.

**Dataset and Preprocessing:**

- **Source data:** Processed interaction and item files are stored under the repository's `data/processed` directory (ratings.parquet, games.parquet, games_annotated.parquet).
- **User–item interactions:** Explicit ratings are treated as the primary interaction signal. We collapse multiple interactions per user–item pair by taking the most recent explicit rating when necessary.
- **Item features & annotations:** Item metadata (categories, playtime, min/max players) and human / LLM-derived context annotations (party friendliness, competitiveness, interaction level, familiarity) are combined into tokenized item features for the hybrid model.
- **Subset profiles:** To support fast iteration, we use named subset profiles (e.g. `current`, `x5`) which deterministically select top users and items by interaction frequency and cap the total number of ratings; experiments report which profile was used.

**Model architecture:**

- We use LightFM (a hybrid matrix-factorization model) trained on implicit interactions derived from ratings. Item-side side-information is encoded as categorical tokens and supplied as an item feature matrix.
- The model learns low-dimensional latent vectors for users and items; predictions are computed via inner products between user and item embeddings and optionally adjusted using side-information during inference.

**Training procedure:**

- **Data split:** Unless otherwise stated, training uses the selected subset's full interaction matrix. For experiments that require validation, we perform an 80/20 chronological or random split depending on the ablation.
- **Optimization:** We train with the Bayesian Personalized Ranking (BPR) or WARP loss as appropriate for ranking objectives. Default hyperparameters used in the experiments are recorded in run artifacts (learning rate, epochs, embedding dimensionality, regularization).
- **Implementation notes:** Sparse interaction matrices are constructed and passed to LightFM ensuring writable buffers; item features are built with deterministic tokenization for reproducibility.

**Group generation & simulation:**

- **Group sampling:** Groups are sampled from the user population to simulate shared decision-making. Sampling strategies include uniform random sampling of users and stratified sampling by user activity level. Group sizes are either fixed or drawn from an empirical distribution.
- **Synthetic group preferences:** Per-group item scores are computed by aggregating members' predicted utilities (see aggregation methods). We also simulate plausible group-level ratings by aggregating individual historical ratings when available.

**Aggregation strategies:**

- **Average:** Group score for an item is the arithmetic mean of members' predicted scores.
- **Least-Misery:** Group score is the minimum predicted score across members, modeling conservative group decision making.
- **Fairness-penalized aggregation:** A tunable penalty term penalizes high variance across member utilities to promote equitable satisfaction; the aggregated score = mean - lambda \* std, where lambda is a fairness weight tuned in experiments.

**Baselines:**

- **Individual average baseline:** Aggregate individual historical means across group members.
- **Random baseline:** Random ranking of candidate items.
- **Least-misery baseline:** As defined above.

**Context-aware adjustment:**

- Context annotations (e.g., party friendliness, competitiveness) are used to reweight item scores at inference time to better match the situational intent. The adjustment is implemented as a multiplicative or additive re-scaling of item utilities based on annotation-to-context alignment scores.

**Constraints & Feasibility Filtering:**

- Candidate items are filtered by deterministic constraints such as allowed player count, maximum playtime, and other metadata-based feasibility checks prior to ranking.

**Evaluation metrics & protocol:**

- **Ranking metrics:** NDCG@K and Precision@K evaluate how well top-K recommendations match group-preferred items.
- **Utility metrics:** RMSE measures the error between predicted and (simulated or observed) group ratings when numeric group ratings are available.
- **Group fairness metrics:** We report `satisfaction_variance` (variance of member utilities within a group for the recommended set) and `min_satisfaction` (the minimum member utility across the group) to capture equitable outcome properties.
- **Statistical reporting:** For each experiment we report mean and standard error across independent group samples or repeated random seeds. When comparing methods we report paired differences and, where appropriate, statistical significance (t-test / bootstrap confidence intervals).

**Experimental protocol:**

- Each experimental run saves artifacts under `models/<run_name>` and `results/<run_name>` including the trained model, train summary, group samples, per-group per-method scores, and evaluation CSVs and plots.
- We sweep key hyperparameters (embedding size, learning rate, fairness weight lambda, aggregation variant) and report metric trends; final reported numbers are taken from held-out seeds and averaged across repeats.

**Ablation studies and analyses:**

- Ablations isolate the impact of (1) item features (metadata vs. annotated context), (2) aggregation strategy, and (3) feasibility constraints. We measure trade-offs between accuracy (NDCG/Precision) and fairness (satisfaction variance / min satisfaction).

**Reproducibility:**

- All code is in the repository. Primary scripts are `scripts/train.py`, `scripts/recommend.py`, `scripts/simulate_groups.py`, and `scripts/evaluate.py` which orchestrate data loading, training, simulation, and evaluation.
- Runtime configurations are stored in `configs/*.yaml`; results and hyperparameters for each run are saved alongside artifacts. Random seeds are logged for each run to enable deterministic replication.

**Limitations & ethical considerations:**

- Context annotations created with LLM assistance must be carefully validated; biases in annotations or training data can propagate to group outcomes.
- The fairness penalty is a pragmatic proxy for equitable outcomes but may not capture normative considerations; interpret results with care.

**Recommended reporting checklist (for paper):**

- Dataset statistics (users, items, interactions) for each subset profile used.
- Model hyperparameters and training details (loss, epochs, optimizer settings).
- Aggregation formulas and fairness weights.
- Evaluation metrics with mean ± SE across seeds and groups.
- Ablation study results isolating context, item features, and constraints.

This methodology provides a concise, reproducible foundation for the experiments reported in the project and is suitable for inclusion in a methods section of a research paper.
