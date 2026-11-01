"""LSTM, GRU, and explicit memory (§10.10, §10.12)."""

from __future__ import annotations

from dl_ch10_pages.gated.callbacks import register_callbacks
from dl_ch10_pages.gated.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch10.algorithms import LSTM_STEP
from maths_self_study.viz.textbooks.deep_learning.ch10.definitions import GATED as GATED_DEFINITIONS
from maths_self_study.viz.textbooks.deep_learning.ch10.observations import GATED as GATED_OBSERVATIONS

GatedPage = define_page(
    label="Gated RNNs",
    value="gated",
    title="Cells that choose what to keep",
    caption="§10.10 and §10.12 - LSTM, GRU, and a soft memory read.",
    summary=(
        "Gated cells replace a single squashing map with a path whose multiplier can stay "
        "near 1. The LSTM forget gate is that path. A GRU does the same job with fewer "
        "gates, and an explicit memory moves the long-term store out of the hidden state "
        "(§10.10, §10.12)."
    ),
    overview_parts=[
        (
            "Cell state",
            r"The LSTM adds the gated candidate onto a copy of the previous cell. Backprop "
            r"through that copy multiplies by the forget gate, not by a $\tanh$ derivative.",
        ),
        (
            "GRU",
            "The update gate blends the old hidden state with a new candidate. A reset "
            "gate can drop the old state out of that candidate (§10.10.2).",
        ),
        (
            "Addressable store",
            "Section 10.12 separates memory from computation: a query matches keys, and "
            "the read vector is a weighted sum of values.",
        ),
    ],
    methodology=[
        "Write a cue of 1 into a vanilla RNN, an LSTM cell, and a GRU.",
        r"Hold the forget gate at $\sigma(b)$ and open the write gate only on the cue.",
        "Address one of four memory slots with a dot-product softmax.",
    ],
    definitions=GATED_DEFINITIONS,
    observations=GATED_OBSERVATIONS,
    algorithm=LSTM_STEP,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
