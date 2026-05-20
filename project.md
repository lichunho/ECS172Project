# Context-Aware and Fairness-Aware Group Recommendation for Board Games

## Motivation and Goals

Group decision-making is difficult because it requires balancing multiple user preferences while also satisfying real-world constraints. Most recommender systems are designed for individual users, and group recommendations are often handled using simple strategies such as averaging preferences or applying least-misery rules. These approaches fail to capture important aspects of real-world decision-making.

Existing systems typically ignore context, feasibility constraints, and fairness across users. For example, a recommendation may fail to account for whether the setting is casual or competitive, whether the group members know each other well, how many people are playing, or how much time the group has available. This can lead to recommendations that are impractical or unsatisfying for part of the group.

Board games are an ideal domain for studying this problem because they require multiple participants, depend heavily on social context, and include strict constraints such as player count and playtime.

The goal of this project is to develop a group recommender system that jointly models user preferences, context, constraints, and fairness to produce recommendations that are accurate, fair, and feasible.

## Problem Statement

We study context-aware and constraint-aware group recommendation.

The input is a group of users with preferences, such as ratings, preferred genres, and preferred game mechanics.

The system also considers context, including the social setting, such as casual, party, or competitive, and group familiarity, such as whether the players are friends or strangers.

The system must also satisfy constraints, including player count and available time.

The output is a ranked list of board games that maximizes group satisfaction, satisfies constraints, and maintains fairness across users.

The main challenge is balancing trade-offs between average satisfaction, individual dissatisfaction, and real-world feasibility.

## Justification of Novelty

Recent work improves group recommendation through better aggregation and modeling, but does not fully address real-world constraints and context.

Tommasel and Diaz-Pace (2024) propose a Monte Carlo Tree Search approach for group recommendation, modeling recommendation as a sequential decision process. While effective for exploring recommendation sequences, their approach does not incorporate feasibility constraints such as player count or playtime.

Liu et al. (2024) introduce Identify-Then-Recommend, an unsupervised framework that dynamically identifies user groups and performs recommendation without labeled data. However, their work focuses on group formation rather than incorporating contextual factors or real-world constraints.

Wu et al. (2023) propose ConsRec, a graph-based framework using hypergraphs and multi-view learning to model group-user-item relationships. While it captures complex interactions, it focuses on preference aggregation and does not explicitly consider context or feasibility.

Existing methods focus on preference modeling but lack constraint-aware filtering, context-aware recommendation, and fairness-aware decision-making.

Our contribution is a unified framework that integrates preferences, context, constraints, and fairness in a real-world domain.

## Proposed Approach

Our system consists of four components.

### Individual Preference Modeling

We use a hybrid recommender model, such as LightFM, that combines collaborative filtering with content-based features. The model learns from user ratings and game attributes such as genre, mechanics, and complexity.

### Constraint-Aware Filtering

We enforce hard constraints such as player count and playtime by removing infeasible games before ranking.

### Fairness-Aware Group Aggregation

We compute a group score using average satisfaction, minimum satisfaction, and a penalty for disagreement. This balances overall enjoyment with fairness across users.

### Context-Aware Adjustment

We adjust scores based on context, such as social setting and group familiarity. For example, party settings favor simple, interactive games, while competitive settings favor strategic games.

## Dataset and Data Acquisition

We use the BoardGameGeek dataset as the primary data source. It includes user ratings, game metadata, player count ranges, and playtime.

We augment the dataset with additional features such as party-friendliness, interaction level, and competitiveness, obtained through LLM-based annotation or manual labeling.

Groups are simulated by sampling users from the dataset. Optionally, a small survey may be conducted for validation.

Preprocessing includes cleaning missing values, normalizing ratings, and encoding categorical features.

## Evaluation Plan

### Accuracy Metrics

We evaluate recommendation accuracy using NDCG@K, Precision@K, and RMSE.

### Fairness Metrics

We evaluate fairness using variance of user satisfaction and minimum satisfaction.

### Baselines

We compare our system against average aggregation, least misery, and random recommendation.

### Qualitative Evaluation

We use an LLM-as-judge to evaluate alignment with preferences, context, and constraints. If feasible, we also conduct a small human evaluation.

## References

Tommasel, A., & Diaz-Pace, J. A. (2024). Leveraging Monte Carlo Tree Search for Group Recommendation. RecSys 2024.

Liu, Y., Zhu, S., Yang, T., Ma, J., & Zhong, W. (2024). Identify Then Recommend: Towards Unsupervised Group Recommendation. NeurIPS 2024.

Wu, X., et al. (2023). ConsRec: Learning Consensus Behind Interactions for Group Recommendation. WWW 2023.