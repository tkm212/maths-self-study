"""Long-horizon gradients, reservoirs, leaky units, and clipping (§10.7-§10.11)."""

from __future__ import annotations

from dl_ch10_pages.dependencies.callbacks import register_callbacks
from dl_ch10_pages.dependencies.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch10.algorithms import ECHO_STATE
from maths_self_study.viz.textbooks.deep_learning.ch10.definitions import LONG_TERM as LONG_TERM_DEFINITIONS
from maths_self_study.viz.textbooks.deep_learning.ch10.observations import LONG_TERM as LONG_TERM_OBSERVATIONS

DependenciesPage = define_page(
    label="Long-term dependencies",
    value="dependencies",
    title="Products of Jacobians, reservoirs, and clips",
    caption="§10.7-§10.11 - Vanishing gradients and three partial remedies.",
    summary=(
        "A plain recurrence multiplies the same Jacobian once per step. Eigenvalues off "
        "the unit circle make long-range signals explode or disappear. Echo-state nets, "
        "leaky units, and gradient clipping each change a different piece of that story "
        "(§10.7-§10.11)."
    ),
    overview_parts=[
        (
            "Eigenvalue powers",
            r"For a linear map, $h^{(t)} = Q \Lambda^{t} Q^{\top} h^{(0)}$. "
            r"Components with $|\lambda| < 1$ die, and components with $|\lambda| > 1$ blow up (§10.7).",
        ),
        (
            "Reservoir",
            "Echo-state networks freeze a random recurrent pool near the edge of stability "
            "and learn only the readout, so the hard weights are never trained (§10.8).",
        ),
        (
            "Several time scales",
            r"A leaky unit with self-weight $\alpha \approx 1$ averages its input slowly. Skip links "
            r"across time are the other trick in §10.9.",
        ),
        (
            "Clipping",
            "Large gradients are rescaled to a maximum norm. Vanished gradients are left alone (§10.11.1).",
        ),
    ],
    methodology=[
        r"Plot $\log_{10}|\lambda|^{t}$ and a few iterates of $\tanh(\lambda s)$.",
        "Run one random reservoir and fit delayed-noise readouts.",
        "Compare a leaky pulse response with a clipped gradient vector.",
    ],
    definitions=LONG_TERM_DEFINITIONS,
    observations=LONG_TERM_OBSERVATIONS,
    algorithm=ECHO_STATE,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
