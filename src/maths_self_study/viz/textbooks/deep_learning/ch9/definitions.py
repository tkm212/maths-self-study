"""Definitions for Deep Learning Ch. 9 dashboard pages."""

from __future__ import annotations

CONVOLUTION = [
    (
        "Cross-correlation",
        r"Deep learning layers use cross-correlation (sum of elementwise products over "
        r"each patch) rather than flipping the kernel (§9.1).",
    ),
    (
        "Stride and padding",
        r"Stride and zero-padding control output grid size and border behavior (§9.5).",
    ),
]

MOTIVATION = [
    (
        "Sparse connectivity",
        r"Each output connects to a small local patch of the input, not the full grid "
        r"(§9.2).",
    ),
    (
        "Parameter sharing",
        r"The same kernel weights are reused at every spatial location, cutting "
        r"parameter count versus a fully connected layer (§9.2).",
    ),
    (
        "Translation equivariance",
        r"If the input shifts, the feature map shifts the same way: "
        r"$f(g(x)) = g(f(x))$ for translation $g$ (§9.2).",
    ),
    (
        "Receptive field",
        r"Deeper units integrate information from a larger input region; depth, "
        r"kernel size, stride, and pooling expand the field (§9.2, Fig. 9.4).",
    ),
    (
        "Edge detectors",
        r"Small kernels can detect edges efficiently; neuroscience motivates similar "
        r"filters (§9.2, §9.10).",
    ),
]

POOLING = [
    (
        "Pooling",
        r"Aggregate each neighborhood (max or average) to downsample feature maps "
        r"(§9.3).",
    ),
    (
        "Translation invariance",
        r"Pooling builds approximate invariance to small shifts of the input "
        r"(§9.3).",
    ),
    (
        "Infinitely strong prior",
        r"Conv encodes local, equivariant interactions; pooling encodes local "
        r"invariance, both as priors over weights (§9.4).",
    ),
    (
        "Classification towers",
        r"Repeated conv and pool blocks shrink spatial maps before a classifier "
        r"(§9.3, Fig. 9.11).",
    ),
]

VARIANTS = [
    (
        "Dilated convolution",
        r"Insert zeros between kernel elements to expand the receptive field without "
        r"adding parameters (§9.5).",
    ),
    (
        "1x1 convolution",
        r"Mix channels at each spatial location without changing H and W (§9.5).",
    ),
    (
        "Padding and stride",
        r"Control output tensor geometry; same formulas as §9.1 apply with effective "
        r"kernel size (§9.5).",
    ),
]

APPLICATIONS = [
    (
        "Structured outputs",
        r"Emit a full spatial map of predictions (e.g. pixel labels) rather than "
        r"a single vector (§9.6).",
    ),
    (
        "Grid dimensionality",
        r"Conv applies to 1D sequences, 2D images, and 3D volumes with the same "
        r"local connectivity idea (§9.7).",
    ),
    (
        "Efficient algorithms",
        r"FFT and Winograd reduce multiply-add cost but implement the same linear "
        r"operation (§9.8).",
    ),
    (
        "Random features",
        r"Untrained filters can already show edge-like selectivity (§9.9).",
    ),
    (
        "Neuroscience",
        r"Simple and complex cells motivate local filters such as Gabors (§9.10).",
    ),
    (
        "History",
        r"CNNs descend from Neocognitron and modern scalable conv hierarchies "
        r"(§9.11).",
    ),
]
