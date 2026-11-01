"""Definitions for Deep Learning Ch. 10 dashboard pages."""

from __future__ import annotations

UNFOLDING = [
    (
        "Unfolding",
        r"Apply a recurrence $\tau-1$ times so the cyclic graph becomes a chain "
        r"that reuses the same $\theta$ at every step (§10.1).",
    ),
    (
        "Dynamical system",
        r"The state update $s^{(t)} = f(s^{(t-1)}; \theta)$ carries information forward "
        r"with no external input (§10.1, eq. 10.1).",
    ),
    (
        "Driven state",
        r"An input $x^{(t)}$ enters the same shared map, so $s^{(t)}$ summarizes "
        r"the past sequence (§10.1, eq. 10.4).",
    ),
]

RNN = [
    (
        "Parameter sharing across time",
        r"The same $U$, $W$, and $V$ are used at every $t$, so the model applies "
        r"to sequences of different lengths (§10.2).",
    ),
    (
        "Teacher forcing",
        r"During training, feed the ground-truth previous output instead of the "
        r"model's own prediction (§10.2.1).",
    ),
    (
        "BPTT",
        r"Backpropagation through the unfolded graph. The gradient at an early "
        r"step is a product of later Jacobians (§10.2.2).",
    ),
    (
        "Sequence density",
        r"An RNN can factor the joint over tokens without an HMM-style conditional "
        r"independence assumption (§10.2.3).",
    ),
]

ARCHITECTURES = [
    (
        "Bidirectional RNN",
        r"A forward state reads the past and a backward state reads the future. "
        r"The output at $t$ uses both, so the full sequence must be observed (§10.3).",
    ),
    (
        "Encoder-decoder",
        r"The encoder compresses an input sequence into a fixed context $C$, often "
        r"its last hidden state. The decoder generates a new sequence from $C$ (§10.4).",
    ),
    (
        "Deep recurrence",
        r"Depth can sit in the input-to-hidden map, the hidden-to-hidden transition, "
        r"or the output map (§10.5).",
    ),
    (
        "Recursive network",
        r"The shared cell combines children on a tree rather than a chain, so the "
        r"longest path is logarithmic in the number of leaves (§10.6).",
    ),
]

LONG_TERM = [
    (
        "Vanishing and exploding products",
        r"In $h^{(t)} = W^{\top} h^{(t-1)}$, eigenvalues of $W$ are raised to $t$. "
        r"Magnitudes below 1 vanish and magnitudes above 1 explode (§10.7).",
    ),
    (
        "Echo state network",
        r"Draw recurrent weights so the reservoir echoes recent history, then train "
        r"only the linear readout (§10.8).",
    ),
    (
        "Leaky unit",
        r"A self-weight $\alpha$ near 1 makes the hidden state a slow average of its "
        r"inputs, a longer time scale than a plain tanh step (§10.9.2).",
    ),
    (
        "Gradient clipping",
        r"If $\|g\| > \tau$, replace $g$ by $\tau g / \|g\|$. This limits explosions "
        r"and does not restore a vanished gradient (§10.11.1).",
    ),
]

GATED = [
    (
        "LSTM cell",
        r"A forget gate multiplies the previous cell state, and an input gate writes "
        r"a new candidate. The self-loop weight can stay near 1 (§10.10.1).",
    ),
    (
        "GRU",
        r"An update gate interpolates between the previous hidden state and a "
        r"candidate, with a reset gate on the candidate's recurrent input (§10.10.2).",
    ),
    (
        "Explicit memory",
        r"A separate store is read by content-based weights, typically a softmax "
        r"over query-key scores (§10.12).",
    ),
]
