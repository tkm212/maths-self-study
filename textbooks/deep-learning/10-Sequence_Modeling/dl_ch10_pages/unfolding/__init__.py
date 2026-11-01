"""Unfolded dynamical systems (§10.1)."""

from __future__ import annotations

from dl_ch10_pages.unfolding.callbacks import register_callbacks
from dl_ch10_pages.unfolding.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch10.algorithms import UNFOLD
from maths_self_study.viz.textbooks.deep_learning.ch10.definitions import UNFOLDING as UNFOLD_DEFINITIONS

UnfoldingPage = define_page(
    label="Unfolding",
    value="unfolding",
    title="From a cycle to a shared chain",
    caption="§10.1 - Dynamical systems and unfolded graphs.",
    summary=(
        "A recurrent net is a dynamical system drawn as a loop. Unfolding the loop "
        "copies the same map once per time step, which is what lets one set of weights "
        "handle sequences of different lengths (§10.1)."
    ),
    overview_parts=[
        (
            "State update",
            "The next state is a function of the previous state and a shared parameter vector. That is equation 10.1.",
        ),
        (
            "External drive",
            r"Equation 10.4 feeds $x^{(t)}$ into the same map, so the state becomes a summary of the inputs seen so far.",
        ),
        (
            "Tied weights",
            r"Every copy of $f$ uses the same $\theta$. A feedforward net with a fresh matrix "
            r"at each position could not transfer a pattern to a new time index.",
        ),
    ],
    methodology=[
        "Iterate a scalar tanh map from a fixed initial state.",
        "Add a unit pulse at a chosen time and compare the driven trajectory.",
        "Plot the product of local derivatives back to the initial state.",
    ],
    definitions=UNFOLD_DEFINITIONS,
    algorithm=UNFOLD,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
