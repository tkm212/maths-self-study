"""RNN forward pass, teacher forcing, and BPTT (§10.2)."""

from __future__ import annotations

from dl_ch10_pages.rnn.callbacks import register_callbacks
from dl_ch10_pages.rnn.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch10.algorithms import BPTT
from maths_self_study.viz.textbooks.deep_learning.ch10.definitions import RNN as RNN_DEFINITIONS
from maths_self_study.viz.textbooks.deep_learning.ch10.observations import RNN as RNN_OBSERVATIONS

RnnPage = define_page(
    label="Recurrent nets",
    value="rnn",
    title="Hidden recurrence, teacher forcing, and BPTT",
    caption="§10.2 - The standard RNN and its gradient.",
    summary=(
        "The network in figure 10.3 maps an input sequence to an output sequence with "
        "one hidden-to-hidden matrix reused at every step. Teacher forcing trains a "
        "variant whose recurrence comes from the outputs, and BPTT sends the loss "
        "backward along the same chain (§10.2)."
    ),
    overview_parts=[
        (
            "Elman step",
            r"$h^{(t)} = \tanh(b + W h^{(t-1)} + U x^{(t)})$, then a softmax readout. "
            r"$U$, $W$, and $V$ do not depend on $t$.",
        ),
        (
            "Teacher forcing",
            r"If the loop is from output to hidden, training can insert the true $y^{(t-1)}$. "
            r"Free-running feeds the model's own prediction and drifts when $W$ is wrong.",
        ),
        (
            "BPTT",
            "The gradient of a late state with respect to an early state is a product of "
            "Jacobians. Short-range errors dominate that product (§10.2.2).",
        ),
        (
            "As a graphical model",
            r"The outputs define $p(y^{(t)} \mid y^{(<t)})$. Unlike an HMM, that "
            r"conditional can depend on the entire prefix (§10.2.3).",
        ),
    ],
    methodology=[
        r"Generate a sequence from $y_t = 0.9\, y_{t-1}$ plus a small drive.",
        "Predict it with a model whose coefficient is the slider, with and without the true prefix.",
        "Run a separate tanh hidden unit on a single cue and plot the BPTT product.",
    ],
    definitions=RNN_DEFINITIONS,
    observations=RNN_OBSERVATIONS,
    algorithm=BPTT,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
