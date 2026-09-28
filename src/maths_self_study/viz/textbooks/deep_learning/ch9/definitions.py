"""Definitions for Deep Learning Ch. 9 dashboard pages."""

from __future__ import annotations

CONVOLUTION = [
    (
        "Cross-correlation",
        r"Deep learning 'convolution' slides a kernel over the input and sums "
        r"elementwise products at each location (§9.1).",
    ),
    (
        "Stride and padding",
        r"Stride subsamples the output grid; zero-padding controls border behavior "
        r"and output spatial size.",
    ),
]

POOLING = [
    (
        "Pooling",
        r"Downsample feature maps by aggregating local neighborhoods — max pooling "
        r"keeps salient activations, average pooling smooths (§9.2).",
    ),
    (
        "Translation tolerance",
        r"Pooling builds approximate invariance to small shifts when stacked with "
        r"convolution (§9.3).",
    ),
]

EDGE_FILTERS = [
    (
        "Local feature detectors",
        r"Hand-crafted kernels (Sobel, Laplacian) highlight edges before learned "
        r"filters dominate in deep stacks (§9.1).",
    ),
    (
        "Channel depth",
        r"Multiple kernels at the same location produce a stack of feature maps — "
        r"the channel dimension encodes different patterns.",
    ),
]

RECEPTIVE_FIELD = [
    (
        "Receptive field",
        r"Each output unit 'sees' a region of the input; depth, kernel size, stride, "
        r"and pooling multiply the effective field size (§9.3).",
    ),
    (
        "Parameter efficiency",
        r"Shared kernels use far fewer weights than a fully connected layer over "
        r"the full input grid.",
    ),
]

TRANSLATION = [
    (
        "Equivariance",
        r"A linear convolution commutes with translation: shifting the input shifts "
        r"the output by the same amount (same padding) (§9.3).",
    ),
    (
        "Invariance",
        r"Pooling discards exact position, trading equivariance for approximate "
        r"translation invariance.",
    ),
]

TOWER = [
    (
        "Spatial hierarchy",
        r"Repeated conv+pool blocks shrink spatial resolution while growing semantic "
        r"receptive field (§9.1-§9.3).",
    ),
    (
        "Output size",
        r"For one spatial axis: $H_{\mathrm{out}} = \lfloor (H_{\mathrm{in}} + 2P - K)/S \rfloor + 1$.",
    ),
]
