"""Deep Learning Ch. 10 — Sequence Modeling dashboard (Dash).

Run from repo root:
    uv run python textbooks/deep-learning/10-Sequence_Modeling/dashboard.py
"""

from __future__ import annotations

from maths_self_study.dashboards.runner import load_chapter_pages, main_dashboard
from maths_self_study.demos.deep_learning.dashboard import create_deep_learning_dashboard

PAGES = load_chapter_pages(
    __file__,
    "dl_ch10_pages",
    "UnfoldingPage",
    "RnnPage",
    "ArchitecturesPage",
    "DependenciesPage",
    "GatedPage",
)


CH10_TAGLINE = (
    "Recurrent nets share one update across time, so a state can carry a sequence of any "
    "length. The rest of the chapter is about keeping that state informative: bidirectional "
    "context, encoder-decoder bottlenecks, and gates that do not wash the gradient out."
)


def create_app():
    return create_deep_learning_dashboard(
        __name__,
        chapter_number=10,
        chapter_title="Sequence Modeling: Recurrent and Recursive Nets",
        book_slug="rnn.html",
        book_link_text="Deep Learning Book — Sequence Modeling",
        pages=PAGES,
        default_page="unfolding",
        tagline=CH10_TAGLINE,
    )


app = create_app()
server = app.server

if __name__ == "__main__":
    main_dashboard(app, label="Deep Learning Ch. 10")
