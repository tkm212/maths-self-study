"""Deep Learning Ch. 7 — Regularization for Deep Learning dashboard (Dash).

Run from repo root:
    uv run python textbooks/deep-learning/7-Regularization_for_Deep_Learning/dashboard.py
"""

from __future__ import annotations

from maths_self_study.dashboards.runner import load_chapter_pages, main_dashboard
from maths_self_study.demos.deep_learning.dashboard import create_deep_learning_dashboard

PAGES = load_chapter_pages(
    __file__,
    "dl_ch07_pages",
    "WeightDecayPage",
    "EarlyStoppingPage",
    "DropoutPage",
    "InputNoisePage",
    "SemiSupervisedMultitaskPage",
    "ParameterSharingPage",
    "BaggingPage",
    "AdversarialPage",
    "TangentDistancePage",
)


def create_app():
    return create_deep_learning_dashboard(
        __name__,
        chapter_number=7,
        chapter_title="Regularization for Deep Learning",
        book_slug="regularization.html",
        book_link_text="Deep Learning Book — Regularization for Deep Learning",
        pages=PAGES,
        default_page="weight_decay",
    )


app = create_app()
server = app.server

if __name__ == "__main__":
    main_dashboard(app, label="Deep Learning Ch. 7")
