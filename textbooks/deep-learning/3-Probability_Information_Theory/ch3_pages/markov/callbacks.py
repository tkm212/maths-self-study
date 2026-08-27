"""Dash callbacks for the Markov / structured models page."""

from __future__ import annotations

from ch3_pages.markov.content import render_body
from maths_self_study.dashboards.callbacks import (
    complement_pairs_callback_inputs,
    define_page_callbacks,
)

_COMPLEMENT_PREFIXES = ("mk-p", "mk-t0", "mk-t1", "mk-u0", "mk-u1")

register_callbacks = define_page_callbacks(
    render_body=render_body,
    inputs=complement_pairs_callback_inputs(_COMPLEMENT_PREFIXES),
    page="markov",
    complement_prefixes=_COMPLEMENT_PREFIXES,
)
