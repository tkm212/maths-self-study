"""Dash callbacks for the information theory page."""

from __future__ import annotations

from ch3_pages.information.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks, simplex_callback_inputs
from maths_self_study.dashboards.components import prob_simplex_ids

_P_IDS = prob_simplex_ids("info-p", [0, 1, 2, 3])
_Q_IDS = prob_simplex_ids("info-q", [0, 1, 2, 3])

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=[*simplex_callback_inputs(_P_IDS), *simplex_callback_inputs(_Q_IDS)],
    page="information",
    simplex_groups=(_P_IDS, _Q_IDS),
)
