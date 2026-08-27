"""Dash callbacks for the random variables page."""

from __future__ import annotations

from ch3_pages.random_variables.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks, simplex_callback_inputs
from maths_self_study.dashboards.components import prob_simplex_ids

_JOINT_IDS = prob_simplex_ids("rv-j", ["00", "01", "10", "11"])
_PMF_IDS = prob_simplex_ids("rv-p", [0, 1, 2, 3])

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=[*simplex_callback_inputs(_JOINT_IDS), *simplex_callback_inputs(_PMF_IDS)],
    page="random_variables",
    simplex_groups=(_JOINT_IDS, _PMF_IDS),
)
