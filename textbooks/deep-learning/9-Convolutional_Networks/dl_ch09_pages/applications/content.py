"""Body content for later Ch. 9 topics (§9.6-§9.11)."""

from __future__ import annotations

import dl_ch09_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.viz.latex import formula_group
from maths_self_study.viz.textbooks.deep_learning.ch9.formulas import CONV1D, GABOR_FORM


def render_body(theta, seed) -> html.Div:
    th = float(coerce_float(theta, default=helpers.GABOR_THETA_DEFAULT))
    sd = int(coerce_float(seed, default=helpers.RANDOM_SEED_DEFAULT))
    fig_labels, label_stats = helpers.plot_structured_labels()
    fig_1d, fig_3d, data_stats = helpers.plot_data_types()
    fig_gabor, fig_rand, feat_stats = helpers.plot_gabor_and_random(th, sd)
    rows = [
        ["Label classes used", f"{int(label_stats['n_classes'])}"],
        ["1D series length", f"{int(data_stats['series_len'])}"],
        ["Gabor theta (rad)", f"{feat_stats['gabor_theta']:.2f}"],
        ["Random vs Sobel |resp|", f"{feat_stats['random_mean_abs']:.3f} / {feat_stats['sobel_mean_abs']:.3f}"],
    ]
    return html.Div([
        html.H3("Structured spatial outputs"),
        html.P(
            "Pixel labeling keeps a full HxW output: each location gets its own prediction "
            "(§9.6). Here we argmax over three filter responses as a toy label map."
        ),
        graph(fig_labels),
        html.H3("Data types: 1D and 3D grids"),
        formula_group(
            ("1D cross-correlation", CONV1D),
            title="Key formulas (§9.7)",
        ),
        graph(fig_1d),
        graph(fig_3d),
        html.H3("Gabor filters and random features"),
        formula_group(
            ("Gabor envelope", GABOR_FORM),
            title="Key formulas (§9.10)",
        ),
        graph(fig_gabor),
        graph(fig_rand),
        html.P(
            "Random filters can mimic edge selectivity before training (§9.9). Fast conv "
            "implementations (FFT, Winograd) reduce cost but change only compute, not the "
            "math (§9.8). Modern CNNs build on Neocognitron and LeNet-style hierarchies "
            "(§9.11)."
        ),
        table(["Measure", "Value"], rows, caption="Applications summary"),
    ])
