"""Algorithms for Deep Learning Ch. 9 dashboard pages."""

from __future__ import annotations

CONV_FORWARD = (
    "2D convolution (cross-correlation) forward pass",
    [
        r"Pad input $I$ with $P$ zeros on each side if needed.",
        r"For each output index $(i,j)$, extract patch $I[iS:(iS+K), jS:(jS+K)]$.",
        r"Set activation $(I * K)[i,j] = \sum_{u,v} patch_{u,v}\, K_{u,v}$.",
        r"Stack outputs from multiple kernels to form channels (§9.1).",
    ],
)

MAX_POOL_FORWARD = (
    "Max pooling forward pass",
    [
        r"Partition the feature map into non-overlapping (or strided) windows.",
        r"Output the maximum value in each window (§9.2).",
        r"Backpropagation routes gradient only to the argmax location.",
    ],
)
