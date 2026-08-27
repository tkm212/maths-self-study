"""LaTeX formula rendering for Dash dashboards via KaTeX."""

from __future__ import annotations

from dash import html

KATEX_VERSION = "0.16.11"
KATEX_CDN = f"https://cdn.jsdelivr.net/npm/katex@{KATEX_VERSION}/dist"

_FORMULA_BLOCK_STYLE = {
    "padding": "14px 18px",
    "background": "#f8fafc",
    "border": "1px solid #e2e8f0",
    "borderRadius": "8px",
    "marginBottom": "16px",
    "overflowX": "auto",
}
_FORMULA_CAPTION_STYLE = {
    "fontSize": "0.85rem",
    "color": "#64748b",
    "marginBottom": "8px",
    "fontWeight": 600,
}
_FORMULA_SOURCE_STYLE = {
    "textAlign": "center",
    "fontSize": "1.05rem",
    "lineHeight": 1.6,
}


def katex_head_html() -> str:
    """KaTeX stylesheet — inject before ``</head>`` in the Dash index template."""
    return (
        f'<link rel="stylesheet" href="{KATEX_CDN}/katex.min.css" '
        f'crossorigin="anonymous" />'
    )


def katex_boot_script() -> str:
    """KaTeX bootstrap — inject before ``</body>`` in the Dash index template."""
    return f"""
<script src="{KATEX_CDN}/katex.min.js" crossorigin="anonymous"></script>
<script>
(function () {{
  var timer = null;

  function renderFormulas() {{
    if (typeof katex === "undefined") {{
      return;
    }}
    document.querySelectorAll(".math-latex-source:not([data-katex-rendered])").forEach(function (el) {{
      var latex = el.textContent.trim();
      if (!latex) {{
        return;
      }}
      var display = el.classList.contains("math-latex-display");
      try {{
        katex.render(latex, el, {{displayMode: display, throwOnError: false}});
        el.setAttribute("data-katex-rendered", "1");
      }} catch (err) {{
        el.textContent = latex;
      }}
    }});
  }}

  function scheduleRender() {{
    if (timer !== null) {{
      clearTimeout(timer);
    }}
    timer = setTimeout(renderFormulas, 50);
  }}

  function boot() {{
    scheduleRender();
    var observer = new MutationObserver(scheduleRender);
    observer.observe(document.body, {{childList: true, subtree: true}});
  }}

  if (document.readyState === "loading") {{
    document.addEventListener("DOMContentLoaded", boot);
  }} else {{
    boot();
  }}
  window.addEventListener("load", scheduleRender);
}})();
</script>
"""


def formula(
    latex: str,
    *,
    caption: str | None = None,
    display: bool = True,
    block_style: dict[str, str | int] | None = None,
) -> html.Div:
    """Styled KaTeX formula block for dashboard page bodies.

    Stores raw LaTeX in ``data-latex``; requires :func:`katex_head_html` and
    :func:`katex_boot_script` in the app index.
    """
    stripped = latex.strip()
    children: list[html.Div] = []
    if caption:
        children.append(html.Div(caption, style=_FORMULA_CAPTION_STYLE))
    display_class = "math-latex-display" if display else "math-latex-inline"
    children.append(
        html.Div(
            stripped,
            className=f"math-latex-source {display_class}",
            style=_FORMULA_SOURCE_STYLE if display else {"display": "inline"},
        )
    )
    return html.Div(
        children,
        className="math-formula-block",
        style=block_style or _FORMULA_BLOCK_STYLE,
    )


def formula_group(*items: tuple[str, str], title: str | None = None) -> html.Div:
    """Stack several captioned display formulas in one panel."""
    children: list[html.Div] = []
    if title:
        children.append(html.Div(title, style={**_FORMULA_CAPTION_STYLE, "marginBottom": "12px"}))
    for caption, latex in items:
        children.append(
            formula(
                latex,
                caption=caption,
                block_style={**_FORMULA_BLOCK_STYLE, "marginBottom": "10px"},
            )
        )
    return html.Div(children, style={"marginBottom": "16px"})
