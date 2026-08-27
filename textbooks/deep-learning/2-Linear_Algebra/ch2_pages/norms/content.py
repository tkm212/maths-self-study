"""Body content for the norms page."""

from __future__ import annotations

import numpy as np
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.deep_learning import ch2_helpers as helpers
from maths_self_study.linalg import cosine_similarity, lp_norm
from maths_self_study.viz.formulas.ch2 import COSINE_SIMILARITY, LP_NORM
from maths_self_study.viz.latex import formula_group


def _norm_label(p: float) -> str:
    if p == np.inf:
        return "L∞"
    if p == int(p):
        return f"L{int(p)}"
    return f"L{p:g}"


def render_body(x1, x2, inf_opts) -> html.Div:
    p_values: tuple[float, ...] = (1.0, 2.0, np.inf) if inf_opts and "inf" in inf_opts else (1.0, 2.0)
    fig = helpers.plot_lp_unit_balls(p_values=p_values)
    x = np.array([coerce_float(x1, default=3.0), coerce_float(x2, default=-4.0)])
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 1.0])
    cos_ab = cosine_similarity(a, b)

    norm_rows = [[_norm_label(p), f"{lp_norm(x, p):.4f}"] for p in p_values]
    norm_rows.append(["cos(e₁, (1,1))", f"{cos_ab:.4f}"])

    return html.Div([
        formula_group(
            ("Lᵖ norm", LP_NORM),
            ("Cosine similarity", COSINE_SIMILARITY),
            title="Key formulas (§2.3)",
        ),
        html.P(
            "The Lᵖ unit ball is {x : ‖x‖ₚ = 1}: all points exactly one unit from the origin in that norm. "
            "Each panel shows its boundary in ℝ².",
            style={"color": "#64748b", "fontSize": "0.9rem", "marginBottom": "8px"},
        ),
        graph(fig),
        table(
            ["Norm / metric", "Value"],
            norm_rows,
            caption=f"Norms of x = ({x[0]:g}, {x[1]:g})",
        ),
        html.P("cos(e₁, (1,1)) = 1/√2 → 45°", style={"color": "#64748b", "marginTop": "8px", "fontSize": "0.9rem"}),
    ])
