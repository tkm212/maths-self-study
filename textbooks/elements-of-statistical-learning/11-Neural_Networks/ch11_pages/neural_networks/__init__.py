"""Neural networks dashboard page."""

from __future__ import annotations

from ch11_pages.neural_networks.callbacks import register_callbacks
from ch11_pages.neural_networks.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch11.algorithms import (
    NEURAL_NETWORKS as NEURAL_NETWORKS_ALGORITHM,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch11.definitions import (
    NEURAL_NETWORKS as NEURAL_NETWORKS_DEFINITIONS,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch11.observations import (
    NEURAL_NETWORKS as NEURAL_NETWORKS_OBSERVATIONS,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch11.theorems import (
    NEURAL_NETWORKS as NEURAL_NETWORKS_THEOREMS,
)

NeuralNetworksPage = define_page(
    label="Neural networks",
    value="neural_networks",
    title="Neural networks",
    caption="§11.3-11.5 — Architecture, backpropagation, weight decay, and early stopping.",
    methodology=[
        r"A single hidden layer with $M$ sigmoid units is a two-stage regression: nonlinear features $Z_m$ then a GLM output (§11.3).",
        r"Backpropagation computes $\nabla_\theta R$ efficiently via the chain rule; SGD updates $\theta \leftarrow \theta - \eta \nabla_\theta R$ (§11.4).",
        r"Early stopping and $L^2$ weight decay both regularise — fewer epochs or larger $\lambda$ shrink effective capacity (§11.5.2).",
        r"Hidden unit count $M$ is a structural hyperparameter; CV accuracy selects architecture without a held-out test set (§11.5.4).",
    ],
    algorithm=NEURAL_NETWORKS_ALGORITHM,
    definitions=NEURAL_NETWORKS_DEFINITIONS,
    theorems=NEURAL_NETWORKS_THEOREMS,
    observations=NEURAL_NETWORKS_OBSERVATIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
