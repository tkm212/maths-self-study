"""Observations for Deep Learning Ch. 10 dashboard pages."""

from __future__ import annotations

RNN = [
    (
        "Exposure bias",
        r"Teacher forcing trains on true prefixes. At test time the model feeds its "
        r"own outputs, so early mistakes compound (§10.2.1).",
    ),
    (
        "Context $C$",
        r"An extra fixed vector, such as an image embedding, can enter every time "
        r"step when the sequence is conditioned on side information (§10.2.4).",
    ),
]

ARCHITECTURES = [
    (
        "When the future is legal",
        r"Bidirectional nets need the whole input before the first output. They fit "
        r"labeling, not causal generation (§10.3).",
    ),
    (
        "Bottleneck",
        r"A single vector $C$ has to carry everything the decoder will need. Long "
        r"inputs lose early tokens when the state is smaller than the sequence (§10.4).",
    ),
]

LONG_TERM = [
    (
        "Stable memory still vanishes",
        r"Holding a bit robustly pushes the map into a flat region, so the gradient "
        r"of a long-range interaction is exponentially smaller than a short-range one "
        r"(Bengio et al., 1994; §10.7).",
    ),
    (
        "Clipping is one-sided",
        r"Rescaling a huge gradient stops a blow-up. It cannot amplify a product of "
        r"many Jacobians that has already gone to zero (§10.11.1).",
    ),
    (
        "Skip connections in time",
        r"A link from $t$ to $t-d$, or removing some recurrent edges, shortens the "
        r"path that gradients have to cross (§10.9.1, §10.9.3).",
    ),
]

GATED = [
    (
        "Information-flow penalty",
        r"One alternative to gates is a regularizer that keeps "
        r"$\partial h^{(t)} / \partial h^{(t-1)}$ from shrinking (§10.11.2).",
    ),
    (
        "Memory networks",
        r"Section 10.12 sketches addressable stores (memory networks, NTMs). The "
        r"demo is only the soft read: a query retrieves a weighted sum of values.",
    ),
]
