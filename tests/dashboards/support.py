"""Shared helpers for dashboard tests (not collected by pytest)."""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Iterator
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ESL_DASHBOARD_ROOT = REPO_ROOT / "textbooks/elements-of-statistical-learning"

CH2_DASHBOARD = REPO_ROOT / "textbooks/deep-learning/2-Linear_Algebra/dashboard.py"
CH3_DASHBOARD = REPO_ROOT / "textbooks/deep-learning/3-Probability_Information_Theory/dashboard.py"
CH4_DASHBOARD = REPO_ROOT / "textbooks/deep-learning/4-Numerical_Computation/dashboard.py"
CH5_DASHBOARD = REPO_ROOT / "textbooks/deep-learning/5-Machine_Learning_Basics/dashboard.py"
ESL_CH2_DASHBOARD = ESL_DASHBOARD_ROOT / "2-Supervised_Learning/dashboard.py"
ESL_CH3_DASHBOARD = ESL_DASHBOARD_ROOT / "3-Linear_Methods/dashboard.py"
ESL_CH4_DASHBOARD = ESL_DASHBOARD_ROOT / "4-Linear_Methods_Classification/dashboard.py"
ESL_CH7_DASHBOARD = ESL_DASHBOARD_ROOT / "7-Model_Assessment/dashboard.py"
ESL_CH11_DASHBOARD = ESL_DASHBOARD_ROOT / "11-Neural_Networks/dashboard.py"
ESL_CH12_DASHBOARD = ESL_DASHBOARD_ROOT / "12-SVM_Flexible_Discriminants/dashboard.py"
ESL_CH13_DASHBOARD = ESL_DASHBOARD_ROOT / "13-Prototype_Methods/dashboard.py"
ESL_CH14_DASHBOARD = ESL_DASHBOARD_ROOT / "14-Unsupervised_Learning/dashboard.py"
ESL_CH15_DASHBOARD = ESL_DASHBOARD_ROOT / "15-Random_Forests/dashboard.py"
ESL_CH16_DASHBOARD = ESL_DASHBOARD_ROOT / "16-Ensemble_Learning/dashboard.py"
ESL_CH17_DASHBOARD = ESL_DASHBOARD_ROOT / "17-Undirected_Graphical_Models/dashboard.py"
ESL_CH18_DASHBOARD = ESL_DASHBOARD_ROOT / "18-High_Dimensional_Problems/dashboard.py"
FML_DASHBOARD_ROOT = REPO_ROOT / "textbooks/financial-machine-learning"
FML_CH2_DASHBOARD = FML_DASHBOARD_ROOT / "2-Financial_Data_Structures/dashboard.py"
FML_CH3_DASHBOARD = FML_DASHBOARD_ROOT / "3-Labeling/dashboard.py"
FML_CH4_DASHBOARD = FML_DASHBOARD_ROOT / "4-Sample_Weights/dashboard.py"

CHAPTER_MODULE_ROOTS = tuple(f"ch{n}_{suffix}" for n in range(2, 19) for suffix in ("pages", "helpers", "data"))

# Static manifest — avoids importing every dashboard at pytest collection time.
ESL_CHAPTER_PAGES: dict[int, list[str]] = {
    2: ["k_nearest_neighbors", "least_squares_regression"],
    3: ["subset_selection", "ridge_regression", "lasso", "pcr_pls"],
    4: ["logistic_regression", "lda", "separating_hyperplanes"],
    5: ["splines", "smoothing_splines"],
    6: ["kernel_smoothers", "kernel_density"],
    7: ["bias_variance", "cross_validation"],
    8: ["em_algorithm", "bagging"],
    9: ["additive_models", "decision_trees"],
    10: ["boosting", "gradient_boosting"],
    11: ["neural_networks", "projection_pursuit"],
    12: ["svm", "flexible_discriminants"],
    13: ["prototype_methods", "nearest_neighbors"],
    14: ["clustering", "principal_components"],
    15: ["random_forests"],
    16: ["ensemble_learning"],
    17: ["graphical_models"],
    18: ["high_dimensional"],
}


def esl_dashboard_paths() -> list[Path]:
    return sorted(ESL_DASHBOARD_ROOT.glob("*/dashboard.py"), key=lambda path: int(path.parent.name.split("-")[0]))


def fml_dashboard_paths() -> list[Path]:
    return sorted(FML_DASHBOARD_ROOT.glob("*/dashboard.py"), key=lambda path: int(path.parent.name.split("-")[0]))


def iter_esl_page_cases() -> Iterator[tuple[Path, str]]:
    for dashboard_path in esl_dashboard_paths():
        ch_num = int(dashboard_path.parent.name.split("-")[0])
        for page_value in ESL_CHAPTER_PAGES[ch_num]:
            yield dashboard_path, page_value


def clear_chapter_modules() -> None:
    import sys

    for name in list(sys.modules):
        if any(name == root or name.startswith(f"{root}.") for root in CHAPTER_MODULE_ROOTS):
            del sys.modules[name]


def load_dl_helpers(chapter: int):
    """Import chapter plotting helpers from the Deep Learning textbook folder."""
    chapter_dirs = {
        2: CH2_DASHBOARD.parent,
        3: CH3_DASHBOARD.parent,
        4: CH4_DASHBOARD.parent,
        5: CH5_DASHBOARD.parent,
    }
    prepare_chapter_import(chapter_dirs[chapter])
    return importlib.import_module(f"ch{chapter}_helpers")


def prepare_chapter_import(chapter_dir: Path) -> None:
    import sys

    clear_chapter_modules()
    chapter_dir_str = str(chapter_dir.resolve())
    sys.path[:] = [path for path in sys.path if path != chapter_dir_str]
    sys.path.insert(0, chapter_dir_str)


def load_dashboard_module(path: Path):
    import sys

    clear_chapter_modules()
    chapter_dir = str(path.parent.resolve())
    if chapter_dir in sys.path:
        sys.path.remove(chapter_dir)
    sys.path.insert(0, chapter_dir)

    module_name = "dashboard_" + path.parent.as_posix().replace("/", "_").replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_esl_page_wiring(dashboard_path: Path, page_value: str) -> None:
    """Fast smoke: filters, shell, callbacks, and content import — no ML or TMDB."""
    chapter_dir = dashboard_path.parent
    ch_num = chapter_dir.name.split("-")[0]
    module = load_dashboard_module(dashboard_path)
    page = next(p for p in module.PAGES if p.value == page_value)

    assert page.build_filters() is not None
    shell = page.build_shell()
    assert shell is not None
    assert page.body_id in str(shell)

    prepare_chapter_import(chapter_dir)
    callbacks_mod = importlib.import_module(f"ch{ch_num}_pages.{page_value}.callbacks")
    content_mod = importlib.import_module(f"ch{ch_num}_pages.{page_value}.content")
    assert hasattr(callbacks_mod, "INPUTS")
    assert callable(content_mod.render_body)
    assert callable(callbacks_mod.register_callbacks)


def verify_esl_dashboard_loads(dashboard_path: Path) -> None:
    """Create the Dash app and ensure every declared page is wired."""
    module = load_dashboard_module(dashboard_path)
    ch_num = int(dashboard_path.parent.name.split("-")[0])
    declared = [p.value for p in module.PAGES]
    assert declared == ESL_CHAPTER_PAGES[ch_num]

    app = module.create_app()
    assert app.layout is not None
    assert len(app.callback_map) >= len(module.PAGES)
