"""Structured outputs, data types, Gabor, random features (§9.6-§9.11)."""

from __future__ import annotations

from dl_ch09_pages.applications.callbacks import register_callbacks
from dl_ch09_pages.applications.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch9.definitions import APPLICATIONS as APPLICATION_DEFINITIONS
from maths_self_study.viz.textbooks.deep_learning.ch9.observations import APPLICATIONS as APPLICATION_OBSERVATIONS

ApplicationsPage = define_page(
    label="Applications",
    value="applications",
    title="Data types, outputs, and classical filters",
    caption="§9.6-§9.10 - Grids, labels, Gabor, and random features.",
    summary=(
        "Conv is not only for image classification: the same local filtering idea applies "
        "to sequences, volumes, and pixel-wise prediction tasks, with links to neuroscience "
        "and efficient implementation (§9.6-§9.11)."
    ),
    overview_parts=[
        (
            "Structured outputs",
            "Segmentation and related tasks need a label (or vector) at every grid cell, "
            "not a single summary vector (§9.6).",
        ),
        (
            "1D and 3D conv",
            "Time series and volumetric data use the same patch-and-share pattern along "
            "one or three spatial axes (§9.7).",
        ),
        (
            "Random and Gabor features",
            "Even fixed filters can look edge-selective; Gabors model simple and complex "
            "cells in visual cortex (§9.9-§9.10).",
        ),
        (
            "Fast algorithms and history",
            "FFT and Winograd reduce cost but implement the same math (§9.8); modern CNNs "
            "extend Neocognitron/LeNet-style hierarchies (§9.11).",
        ),
    ],
    methodology=[
        "Label each pixel by argmax over a small filter bank.",
        "Convolve a 1D signal and inspect a 3D volume slice.",
        "Rotate a Gabor kernel and compare random vs Sobel responses.",
    ],
    definitions=APPLICATION_DEFINITIONS,
    observations=APPLICATION_OBSERVATIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
