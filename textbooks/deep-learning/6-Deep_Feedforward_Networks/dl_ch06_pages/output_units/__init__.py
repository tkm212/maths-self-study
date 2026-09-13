"""Output unit types page."""

from __future__ import annotations

from dl_ch06_pages.output_units.callbacks import register_callbacks
from dl_ch06_pages.output_units.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch6.definitions import OUTPUT_UNITS as OUTPUT_DEFINITIONS

OutputUnitsPage = define_page(
    label="Output units",
    value="output_units",
    title="Sigmoid and softmax outputs",
    caption="§6.2.2 — Bernoulli and Multinoulli output distributions.",
    summary=(
        "The output layer must match the target distribution: a sigmoid unit "
        "for binary Bernoulli labels, softmax for mutually exclusive classes. "
        "Logits are transformed into valid probabilities; cross-entropy is the "
        "natural loss when maximum likelihood is the training objective."
    ),
    methodology=[
        "Linear unit + Gaussian noise → regression (least squares / MLE).",
        "Sigmoid output → Bernoulli; one probability P(y=1|x).",
        "Softmax output → Multinoulli; vector sums to 1 over K classes.",
    ],
    definitions=OUTPUT_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
