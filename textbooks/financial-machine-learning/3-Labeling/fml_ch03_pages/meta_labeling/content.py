"""Body content for the meta-labeling page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import text_box


def render_body() -> html.Div:
    return html.Div([
        text_box(
            steps=[
                "Primary model: decides when to bet and direction (long/short) from a signal or heuristic.",
                "Run triple-barrier labeling on primary events to get realized outcomes.",
                "Meta-model: binary label — take bet (1) if outcome profitable for primary side, else pass (0).",
                "Meta probability scales position size; abstaining improves precision at lower trade count.",
                "Workflow separates direction (primary) from sizing and filtering (meta).",
            ],
            title="Meta-labeling workflow (p. 50)",
        ),
        text_box(
            steps=[
                "Train primary on traditional features; generate events with CUSUM or similar.",
                "Label events with triple barrier; derive meta-label from whether primary side won.",
                "Train meta classifier on features at event time; use probability for bet size.",
                "Full meta-labeling demo requires a primary signal implementation — see book and mlfinlab.",
            ],
            title="Implementation notes",
        ),
    ])
