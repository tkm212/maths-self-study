"""Observations for ESL Ch. 11 dashboard pages."""

from __future__ import annotations

NEURAL_NETWORKS = [
    (
        "Training vs test dynamics",
        r"Training loss decreases monotonically under SGD; test error often improves early then flattens or rises (overfitting) (ESL §11.4).",
    ),
    (
        "Weight decay trade-off",
        r"$\lambda = 0$ fits all training noise; small $\lambda$ adds mild shrinkage; large $\lambda$ shrinks weights toward zero and the network approaches a near-linear model (ESL §11.5.2).",
    ),
    (
        "Architecture capacity",
        r"A single hidden layer suffices for most problems; more units increase capacity but require stronger regularisation (ESL §11.5.4).",
    ),
]

PROJECTION_PURSUIT = [
    (
        "PPR vs OLS",
        r"OLS fits a hyperplane; PPR with small $M$ is a single-index model; large $M$ can approximate continuous functions (ESL §11.2).",
    ),
    (
        "Ridge function shape",
        r"Steep S-curves indicate strong discrimination along that projection; flat functions suggest the term contributes little (ESL §11.2).",
    ),
]
