"""Chapter-level layout shells for Dash dashboards."""

from __future__ import annotations

from typing import Any

from dash import dcc, html

_TAB_WRAP_STYLE = """
#page-tabs .tab-container {
    flex-wrap: wrap !important;
    height: auto !important;
    row-gap: 4px;
}
#page-tabs .tab {
    white-space: normal !important;
    height: auto !important;
    min-height: 36px;
    line-height: 1.25;
    padding-top: 8px;
    padding-bottom: 8px;
}
"""


def page_shell(
    title: str,
    caption: str,
    filters: html.Div,
    body_id: str,
    *,
    methodology: list[str] | None = None,
) -> html.Div:
    from maths_self_study.dashboards.components import text_box

    children: list[Any] = [
        html.H2(
            title,
            style={
                "marginBottom": "4px",
                "overflowWrap": "anywhere",
                "wordBreak": "break-word",
                "lineHeight": 1.25,
            },
        ),
        html.P(caption, style={"color": "#64748b", "marginTop": 0}),
    ]
    if methodology:
        children.append(text_box(steps=methodology, title="How it works"))
    children.extend([filters, html.Div(id=body_id)])
    return html.Div(children)


def chapter_layout(
    *,
    title: str,
    subtitle: str,
    tabs: list[dict[str, str]],
    default_tab: str,
    book_href: str,
    book_link_text: str,
) -> html.Div:
    return html.Div(
        [
            html.H1(
                title,
                style={
                    "overflowWrap": "anywhere",
                    "wordBreak": "break-word",
                    "lineHeight": 1.2,
                },
            ),
            html.P(subtitle, style={"color": "#64748b"}),
            html.Style(_TAB_WRAP_STYLE),
            dcc.Tabs(
                id="page-tabs",
                value=default_tab,
                children=[dcc.Tab(label=tab["label"], value=tab["value"]) for tab in tabs],
                parent_style={
                    "flexWrap": "wrap",
                    "height": "auto",
                    "alignItems": "flex-start",
                },
            ),
            html.Div(id="page-content", style={"marginTop": "18px"}),
            html.Div(
                html.A(book_link_text, href=book_href, target="_blank"),
                style={"marginTop": "28px", "fontSize": "0.9rem"},
            ),
        ],
        style={
            "maxWidth": "1200px",
            "margin": "0 auto",
            "padding": "24px 20px",
            "fontFamily": "system-ui, sans-serif",
        },
    )
