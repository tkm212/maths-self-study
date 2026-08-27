"""Dash callbacks for the stability page."""

from __future__ import annotations

from ch4_pages.stability.content import render_body
from maths_self_study.dashboards.callbacks import define_page_callbacks, simplex_callback_inputs
from maths_self_study.dashboards.components import prob_simplex_ids

_LOGIT_IDS = prob_simplex_ids("stab-z", [0, 1, 2])

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=simplex_callback_inputs(_LOGIT_IDS),
    page="stability",
)
