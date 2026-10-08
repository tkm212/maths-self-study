"""Input noise robustness page."""

from __future__ import annotations

from dl_ch07_pages.input_noise.callbacks import register_callbacks
from dl_ch07_pages.input_noise.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch7.definitions import INPUT_NOISE as INPUT_NOISE_DEFINITIONS

InputNoisePage = define_page(
    label="Input noise",
    value="input_noise",
    title="Noise robustness",
    caption="§7.5 — Inject noise into inputs during training.",
    summary=(
        "Adding noise to inputs during training encourages smooth, robust "
        "representations that tolerate small perturbations. Unlike label noise, "
        "input noise regularizes the mapping from x to features without directly "
        "corrupting the targets."
    ),
    methodology=[
        "Replace x with x + epsilon, epsilon ~ N(0, sigma^2), each training step.",
        "The network learns to average over perturbations.",
        "Too much noise underfits; a moderate sigma can improve validation error.",
    ],
    definitions=INPUT_NOISE_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
