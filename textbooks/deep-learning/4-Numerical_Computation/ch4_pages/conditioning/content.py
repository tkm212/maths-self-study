"""Body content for the conditioning page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.deep_learning import ch4_helpers as helpers
from maths_self_study.viz.formulas.ch4 import CONDITION_BOUND, CONDITION_NUMBER
from maths_self_study.viz.latex import formula_group


def render_body(kappa, delta) -> html.Div:
    ratio = max(coerce_float(kappa, default=10_000.0), 2.0)
    perturb = coerce_float(delta, default=helpers.CONDITIONING_DELTA)
    demo = helpers.conditioning_scenario(ratio, perturb)
    fig = helpers.plot_conditioning_demo(ratio, delta=perturb)
    dx = demo["delta_x"]
    rows = [
        ["epsilon (near-singularity)", f"{demo['epsilon']:.2e}"],
        ["kappa(A)", f"{demo['kappa']:.2e}"],
        ["Perturbation delta on b0", f"{perturb:.2e}"],
        ["x (exact)", f"[{demo['x'][0]:.4f}, {demo['x'][1]:.4f}]"],
        ["x' (perturbed)", f"[{demo['x_pert'][0]:.4f}, {demo['x_pert'][1]:.4f}]"],
        ["delta x", f"[{dx[0]:.4f}, {dx[1]:.4f}]"],
        ["Error amplification ||delta x|| / ||delta b||", f"{demo['amplification']:.2e}x"],
    ]
    return html.Div([
        html.H3("Nearly parallel rows amplify rounding error"),
        formula_group(
            ("Condition number", CONDITION_NUMBER),
            ("Error amplification bound", CONDITION_BOUND),
            title="Key formulas (§4.2)",
        ),
        html.P(
            "Rows of A are almost identical, so b and b + delta look the same but x and x' can differ wildly.",
            style={"color": "#475569", "fontSize": "0.95rem"},
        ),
        graph(fig),
        table(["Measure", "Value"], rows, caption="Conditioning summary"),
    ])
