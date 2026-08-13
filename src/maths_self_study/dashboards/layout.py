"""Chapter-level layout shells for Dash dashboards."""

from __future__ import annotations

from dash import dcc, html


def page_shell(title: str, caption: str, filters: html.Div, body_id: str) -> html.Div:
    return html.Div([
        html.H2(title, style={"marginBottom": "4px"}),
        html.P(caption, style={"color": "#64748b", "marginTop": 0}),
        filters,
        html.Div(id=body_id),
    ])


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
            html.H1(title),
            html.P(subtitle, style={"color": "#64748b"}),
            dcc.Tabs(
                id="page-tabs",
                value=default_tab,
                children=[dcc.Tab(label=tab["label"], value=tab["value"]) for tab in tabs],
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
