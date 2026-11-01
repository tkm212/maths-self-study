"""Algorithms for Deep Learning Ch. 10 dashboard pages."""

from __future__ import annotations

UNFOLD = (
    "Unfold a driven recurrence",
    [
        r"Write one step $s^{(t)} = f(s^{(t-1)}, x^{(t)}; \theta)$.",
        r"Substitute the same rule at $t-1, t-2, \ldots$ until the graph is a chain.",
        r"Tie $\theta$ across every copy of $f$ so length can change at test time (§10.1).",
    ],
)

BPTT = (
    "Backpropagation through time",
    [
        r"Run the shared map from $t = 1$ to $\tau$ and store each hidden state.",
        r"Sum the per-step losses $L = \sum_t L^{(t)}$.",
        r"Backpropagate from each $L^{(t)}$ to earlier states, multiplying Jacobians of $f$.",
        r"Accumulate $\partial L / \partial \theta$ across the tied copies of the weights (§10.2.2).",
    ],
)

SEQ2SEQ = (
    "Encoder-decoder forward pass",
    [
        r"Encode $x^{(1)}, \ldots, x^{(\tau_x)}$ with a shared RNN.",
        r"Set the context $C$ to the final encoder state (or another summary of those states).",
        r"Initialize the decoder from $C$ and emit $y^{(1)}, \ldots, y^{(\tau_y)}$ until a stop token (§10.4).",
    ],
)

ECHO_STATE = (
    "Echo-state readout",
    [
        r"Sample recurrent weights and rescale them to a chosen spectral radius.",
        r"Run the reservoir on the input. Do not train these weights.",
        r"Fit a linear readout from $h^{(t)}$ to the target, often with ridge regression (§10.8).",
    ],
)

LSTM_STEP = (
    "LSTM cell step",
    [
        r"Forget gate $f^{(t)} = \sigma(b^{f} + U^{f} x^{(t)} + W^{f} h^{(t-1)})$.",
        r"Input gate and candidate decide what to write into the cell.",
        r"Cell update $s^{(t)} = f^{(t)} \odot s^{(t-1)} + i^{(t)} \odot \tilde{s}^{(t)}$.",
        r"Output gate masks $\tanh(s^{(t)})$ to form $h^{(t)}$ (§10.10.1).",
    ],
)
