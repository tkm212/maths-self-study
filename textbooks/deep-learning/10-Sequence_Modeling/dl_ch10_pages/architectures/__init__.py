"""Bidirectional, encoder-decoder, deep, and recursive nets (§10.3-§10.6)."""

from __future__ import annotations

from dl_ch10_pages.architectures.callbacks import register_callbacks
from dl_ch10_pages.architectures.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch10.algorithms import SEQ2SEQ
from maths_self_study.viz.textbooks.deep_learning.ch10.definitions import ARCHITECTURES as ARCH_DEFINITIONS
from maths_self_study.viz.textbooks.deep_learning.ch10.observations import ARCHITECTURES as ARCH_OBSERVATIONS

ArchitecturesPage = define_page(
    label="Architectures",
    value="architectures",
    title="Future context, fixed vectors, and trees",
    caption="§10.3-§10.6 - Bidirectional, seq2seq, deep, and recursive nets.",
    summary=(
        "A left-to-right chain is only one way to share a cell. Bidirectional nets read "
        "the future, encoder-decoder models squeeze a whole input into one vector, depth "
        "can sit inside each step, and a recursive net folds a tree instead of a chain "
        "(§10.3-§10.6)."
    ),
    overview_parts=[
        (
            "Bidirectional",
            r"The output at time $t$ is a function of a forward state and a backward state. "
            r"That helps when the label depends on tokens that have not been seen yet.",
        ),
        (
            "Sequence to sequence",
            r"The encoder's last state is a context vector $C$. The decoder is a second RNN "
            r"started from $C$, so input and output lengths can differ.",
        ),
        (
            "Depth and trees",
            "Stacking a transition multiplies the path length. A balanced tree over the "
            "same leaves is only logarithmic, which is the recursive-network idea.",
        ),
    ],
    methodology=[
        "Put a spike late in a sequence and score a label that turns on before the spike.",
        r"Store the last $k$ symbols of a random $\pm 1$ sequence and fit a linear decoder.",
        "Compare chain depth, tree depth, and chain depth times the number of stacked layers.",
    ],
    definitions=ARCH_DEFINITIONS,
    observations=ARCH_OBSERVATIONS,
    algorithm=SEQ2SEQ,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
